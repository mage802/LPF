#!/usr/bin/env python3
"""Score Week-1 train samples with length, MTLD, and base answer NLL."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from common import (
    extract_int_list,
    load_model_and_tokenizer,
    per_example_answer_nll,
    read_jsonl,
    seed_everything,
    sha256_file,
    user_text,
    write_json,
    write_jsonl,
)


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="Qwen/Qwen2.5-0.5B-Instruct"
    )
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--adapter-path")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20250719)
    parser.add_argument("--mtld-threshold", type=float, default=0.72)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def mtld_direction(tokens: list[str], threshold: float) -> float:
    if not tokens:
        return 0.0
    factors = 0.0
    types: set[str] = set()
    start = 0
    for index, token in enumerate(tokens):
        types.add(token)
        length = index - start + 1
        ttr = len(types) / length
        if ttr <= threshold:
            factors += 1.0
            start = index + 1
            types = set()
    remainder = len(tokens) - start
    if remainder:
        remainder_ttr = len(types) / remainder
        denominator = max(1e-8, 1.0 - threshold)
        factors += max(0.0, (1.0 - remainder_ttr) / denominator)
    return len(tokens) / max(factors, 1e-8)


def mtld(text: str, threshold: float) -> float:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    if len(tokens) < 2:
        return float(len(tokens))
    return (
        mtld_direction(tokens, threshold)
        + mtld_direction(list(reversed(tokens)), threshold)
    ) / 2.0


def percentile_ranks(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    for rank, index in enumerate(ordered):
        result[index] = rank / (len(values) - 1)
    return result


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    rows = read_jsonl(args.input)
    if args.max_samples:
        rows = rows[: args.max_samples]

    model, tokenizer, device, dtype = load_model_and_tokenizer(
        args.model_name,
        revision=args.model_revision,
        adapter_path=args.adapter_path,
    )
    nll_rows = []
    for start in tqdm(
        range(0, len(rows), args.batch_size), desc="Base answer NLL"
    ):
        nll_rows.extend(
            per_example_answer_nll(
                model,
                tokenizer,
                rows[start : start + args.batch_size],
                args.max_length,
                args.batch_size,
                device,
            )
        )
    nll_by_id = {row["sample_id"]: row for row in nll_rows}

    scored = []
    for row in tqdm(rows, desc="Static difficulty"):
        nll = nll_by_id[row["sample_id"]]
        text = f"{user_text(row)}\n{row['output']}"
        total_tokens = int(row.get("total_tokens", 0))
        if total_tokens < 8:
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_text(row)},
                    {
                        "role": "assistant",
                        "content": str(row["output"]).strip(),
                    },
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
            total_tokens = len(extract_int_list(prompt))
        item = dict(row)
        item["total_tokens"] = total_tokens
        item["difficulty"] = {
            "length": float(total_tokens),
            "mtld": float(mtld(text, args.mtld_threshold)),
            "base_answer_nll": float(nll["mean_nll"]),
            "base_answer_tokens": int(nll["answer_tokens"]),
        }
        scored.append(item)

    for metric in ["length", "mtld", "base_answer_nll"]:
        ranks = percentile_ranks(
            [float(row["difficulty"][metric]) for row in scored]
        )
        for row, rank in zip(scored, ranks):
            row["difficulty"][f"{metric}_percentile"] = rank

    write_jsonl(args.output, scored)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "model_name": args.model_name,
        "model_revision_requested": args.model_revision,
        "model_revision_loaded": getattr(model.config, "_commit_hash", None),
        "adapter_path": args.adapter_path,
        "device": str(device),
        "dtype": str(dtype),
        "max_length": args.max_length,
        "mtld_threshold": args.mtld_threshold,
        "loss_scope": "assistant tokens only",
        "samples": len(scored),
        "domains": dict(Counter(str(row["domain"]) for row in scored)),
        "non_finite_nll": sum(
            not math.isfinite(row["difficulty"]["base_answer_nll"])
            for row in scored
        ),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
