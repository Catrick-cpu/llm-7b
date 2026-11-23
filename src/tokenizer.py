import os
from typing import Iterable, Iterator, Optional

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|bos|>", "<|unk|>"]


def _stream_corpus_texts(sources: list[dict], max_chars: int) -> Iterator[str]:
    total_chars = 0
    for src in sources:
        if total_chars >= max_chars:
            break
        try:
            ds = load_dataset(
                src["dataset_id"],
                name=src.get("config_name"),
                data_dir=src.get("data_dir"),
                split=src.get("split", "train"),
                streaming=True,
            )
            field = src.get("text_field", "text")
            for row in ds:
                text = row.get(field)
                if isinstance(text, str) and len(text.strip()) > 0:
                    yield text
                    total_chars += len(text)
                    if total_chars >= max_chars:
                        break
        except Exception as exc:
            print(f"[tokenizer] Warning: could not stream {src.get('name', src['dataset_id'])}: {exc}")


def train_bpe_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    out_dir: str,
) -> PreTrainedTokenizerFast:
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|endoftext|>",
    )
    hf_tok.save_pretrained(out_dir)
    return hf_tok


def load_or_build_tokenizer(tok_cfg: dict, data_sources: list[dict]):
    explicit_path = tok_cfg.get("pretrained_or_path")
    if explicit_path and os.path.exists(explicit_path):
        tok = AutoTokenizer.from_pretrained(explicit_path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token or "<|endoftext|>"
        return tok

    out_dir = tok_cfg.get("tokenizer_out_dir", "/kaggle/working/tokenizer_65k")
    if os.path.exists(os.path.join(out_dir, "tokenizer.json")):
        tok = PreTrainedTokenizerFast.from_pretrained(out_dir)
        if tok.pad_token_id is None:
            tok.pad_token = "<|pad|>"
        return tok

    if tok_cfg.get("train_if_missing", False):
        max_chars = int(tok_cfg.get("sample_chars_for_training", 20_000_000))
        vocab_size = int(tok_cfg.get("vocab_size", 65536))
        corpus = _stream_corpus_texts(data_sources, max_chars=max_chars)
        try:
            return train_bpe_tokenizer(corpus, vocab_size=vocab_size, out_dir=out_dir)
        except Exception as exc:
            print(f"[tokenizer] Falling back to pretrained tokenizer due to: {exc}")

    fallback_name = tok_cfg.get("fallback_tokenizer", "gpt2")
    tok = AutoTokenizer.from_pretrained(fallback_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok