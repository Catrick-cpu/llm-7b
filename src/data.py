import random
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


class PackedMixtureDataset(IterableDataset):
    """
    Streams multiple text/code/math corpora according to domain weights,
    tokenizes documents with EOS delimiters, and packs them into fixed-length
    `(input_ids, labels)` chunks of length `seq_len`.
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
        # Use tokenizer EOS if available, but cap at model vocab size for safety
        tokenizer_eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        # We'll use a safe EOS per-source; default to tokenizer_eos
        self.default_eos_id = tokenizer_eos

    def _make_source_iterator(self, src: dict, source_idx: int) -> Iterator[str]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        global_shard_id = self.rank * num_workers + worker_id
        total_shards = self.world_size * num_workers

        rng = random.Random(self.seed + source_idx * 997 + global_shard_id)

        # Handle synthetic dataset for smoke tests
        if src.get("dataset_id") == "synthetic":
            # Yield token IDs directly within model vocab range
            # Use vocab range [1, vocab_size-1], reserve 0 for padding/special
            vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 1024
            valid_tokens = list(range(1, min(vocab_size, 1024)))
            doc_idx = 0
            while True:
                # Generate random document of 20-100 tokens
                doc_len = rng.randint(20, 100)
                tokens = [rng.choice(valid_tokens) for _ in range(doc_len)]
                if doc_idx % total_shards == global_shard_id:
                    yield tokens  # yield list of token IDs directly
                doc_idx += 1
                if not src.get("loop", True) and doc_idx >= 100:
                    break
            return

        ds = load_dataset(
            src["dataset_id"],
            name=src.get("config_name"),
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
        for idx, src in enumerate(self.sources):
            try:
                it = self._make_source_iterator(src, idx)
                iterators.append(it)
                active_weights.append(self.weights[idx])
                is_synthetic_list.append(src.get("dataset_id") == "synthetic")
            except Exception as exc:
                print(f"[data] Skipping source {src.get('name', src['dataset_id'])}: {exc}")

        if not iterators:
            raise RuntimeError("No data sources could be initialized.")

        w_sum = sum(active_weights)
        norm_w = [w / w_sum for w in active_weights]

        # Build per-source EOS IDs (synthetic uses a safe ID within model vocab)
        model_vocab_size = getattr(self.tokenizer, 'vocab_size', 1024)
        eos_ids = []
        for src in self.sources:
            if src.get("dataset_id") == "synthetic":
                eos_ids.append(1)  # Use token ID 1 as EOS for synthetic (within vocab)
            else:
                eos_ids.append(self.default_eos_id)

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