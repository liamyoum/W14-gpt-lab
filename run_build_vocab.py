# -*- coding: utf-8 -*-
"""NSMC Light baseline용 BPE vocabulary를 생성하는 CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a BPE vocabulary from NSMC LM text.")
    parser.add_argument("--train-path", type=Path, required=True, help="LM train text path.")
    parser.add_argument("--train-char-limit", type=int, default=500000, help="Character limit for vocab training.")
    parser.add_argument("--vocab-size", type=int, default=2000, help="BPE vocabulary size.")
    parser.add_argument("--output-path", type=Path, required=True, help="Where to save the vocabulary JSON.")
    parser.add_argument(
        "--sample-text",
        type=str,
        default="이 영화는 정말 좋았다",
        help="Round-trip smoke test sample after saving the vocab.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.train_path.exists():
        raise SystemExit(f"LM train text가 없습니다: {args.train_path}")

    corpus = args.train_path.read_text(encoding="utf-8")
    if args.train_char_limit > 0:
        corpus = corpus[: args.train_char_limit]
    if not corpus:
        raise SystemExit("vocab 학습용 corpus가 비어 있습니다.")

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(corpus)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output_path)

    # 저장 직후 load/encode/decode가 되는지 간단히 smoke test를 수행합니다.
    loaded = BPETokenizer(vocab_size=args.vocab_size)
    loaded.load(args.output_path)
    ids = loaded.encode(args.sample_text, add_bos_eos=True)
    decoded = loaded.decode(ids)

    print(f"saved vocab: {args.output_path}")
    print(f"vocab_size: {loaded.vocab_size}")
    print(f"sample_ids[:20]: {ids[:20]}")
    print(f"decoded_sample: {decoded}")


if __name__ == "__main__":
    main()
