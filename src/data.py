import random
import json
import os
from typing import Iterator, Optional

from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset


def _normalize_weights(sources: list[dict]) -> list[float]:
    raw = [max(0.0, float(s.get("weight", 1.0))) for s in sources]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(sources) for _ in sources]
    return [w / total for w in raw]


def _read_local_jsonl(path: str) -> Iterator[str]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get('text') or obj.get('content') or obj.get('data')
                if isinstance(text, str) and text.strip():
                    yield text
            except json.JSONDecodeError:
                continue


def _read_local_parquet(path: str) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        for col in ['text', 'content', 'data']:
            if col in table.column_names:
                for val in table[col].to_pylist():
                    if isinstance(val, str) and val.strip():
                        yield val
                break
    except Exception:
        pass


def _stream_local_corpus(data_dir: str, source_idx: int, seed: int, 
                          rank: int, world_size: int, 
                          worker_id: int, num_workers: int,
                          shuffle_buffer: int) -> Iterator[str]:
    """Stream local JSONL/Parquet files with sharding."""
    files = sorted([f for f in os.listdir(data_dir) 
                   if f.endswith('.jsonl') or f.endswith('.parquet')])
    if not files:
        return
    
    global_shard_id = rank * num_workers + worker_id
    total_shards = world_size * num_workers
    rng = random.Random(seed + source_idx * 997 + global_shard_id)
    
    doc_idx = 0
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        if fname.endswith('.jsonl'):
            reader = _read_local_jsonl(fpath)
        elif fname.endswith('.parquet'):
            reader = _read_local_parquet(fpath)
        else:
            continue
        
        for text in reader:
            if doc_idx % total_shards == global_shard_id:
                yield text
            doc_idx += 1


class PackedMixtureDataset(IterableDataset):
    """
    Streams multiple text/code/math corpora according to domain weights,
    tokenizes documents with EOS delimiters, and packs them into fixed-length
    `(input_ids, labels)` chunks of length `seq_len`.
    Supports both streaming from HF Hub and local JSONL/Parquet files (offline).
    """

    def __init__(
        self,
        sources: list[dict],
        tokenizer,
        seq_len: int = 2048,
        seed: int = 42,
        shuffle_buffer: int = 10000,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.sources = sources
        self.weights = _normalize_weights(sources)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.rank = rank
        self.world_size = world_size
        tokenizer_eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.default_eos_id = tokenizer_eos

    def _make_source_iterator(self, src: dict, source_idx: int):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        global_shard_id = self.rank * num_workers + worker_id
        total_shards = self.world_size * num_workers

        rng = random.Random(self.seed + source_idx * 997 + global_shard_id)

        # Synthetic dataset for smoke tests
        if src.get("dataset_id") == "synthetic":
            vocab_size = getattr(self.tokenizer, 'vocab_size', 1024)
            valid_tokens = list(range(1, min(vocab_size, 1024)))
            doc_idx = 0
            while True:
                doc_len = rng.randint(20, 100)
                tokens = [rng.choice(valid_tokens) for _ in range(doc_len)]
                if doc_idx % total_shards == global_shard_id:
                    yield tokens
                doc_idx += 1
                if not src.get("loop", True) and doc_idx >= 100:
                    break
            return

        # Local data directory (offline mode)
        local_dir = src.get("local_dir")
        if local_dir and os.path.exists(local_dir):
            yield from _stream_local_corpus(
                local_dir, source_idx, self.seed, 
                self.rank, self.world_size, worker_id, num_workers, self.shuffle_buffer
            )
            return

        # Streaming from HF Hub (online mode)
        config_name = src.get("config_name")
        if config_name == "":
            config_name = None
        ds = load_dataset(
            src["dataset_id"],
            name=config_name,
            data_dir=src.get("data_dir"),
            split=src.get("split", "train"),
            streaming=bool(src.get("streaming", True)),
        )
        if hasattr(ds, "shuffle") and self.shuffle_buffer > 0:
            ds = ds.shuffle(seed=self.seed + source_idx, buffer_size=self.shuffle_buffer)

        field = src.get("text_field", "text")
        doc_idx = 0
        while True:
            for row in ds:
                if doc_idx % total_shards == global_shard_id:
                    text = row.get(field)
                    if isinstance(text, str) and text.strip():
                        yield text
                doc_idx += 1
            if not src.get("loop", True):
                break

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + self.rank * 10007 + worker_id * 131)

        iterators = []
        active_weights = []
        is_synthetic_list = []
        eos_ids = []

        for idx, src in enumerate(self.sources):
            try:
                it = self._make_source_iterator(src, idx)
                iterators.append(it)
                active_weights.append(self.weights[idx])
                is_synthetic = src.get("dataset_id") == "synthetic"
                is_synthetic_list.append(is_synthetic)
                if is_synthetic:
                    eos_ids.append(1)
                else:
                    eos_ids.append(self.default_eos_id)
            except Exception as exc:
                print(f"[data] Skipping source {src.get('name', src['dataset_id'])}: {exc}")

        if not iterators:
            raise RuntimeError("No data sources could be initialized.")

        w_sum = sum(active_weights)
        norm_w = [w / w_sum for w in active_weights]

        token_buffer: list[int] = []
        while True:
            chosen_idx = rng.choices(range(len(iterators)), weights=norm_w, k=1)[0]
            chosen_it = iterators[chosen_idx]
            is_synthetic = is_synthetic_list[chosen_idx]
            try:
                item = next(chosen_it)
            except StopIteration:
                continue

            if is_synthetic:
                ids = item
            else:
                text = item
                ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            token_buffer.extend(ids)
            token_buffer.append(eos_ids[chosen_idx])

            while len(token_buffer) >= self.seq_len:
                chunk = token_buffer[: self.seq_len]
                token_buffer = token_buffer[self.seq_len :]
                x = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": x, "labels": x.clone()}