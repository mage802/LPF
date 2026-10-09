from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "sample_id" not in row:
                raise ValueError(f"{path}:{line_number} is missing sample_id")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def user_text(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    extra = str(row.get("input", "")).strip()
    return f"{instruction}\n{extra}" if extra else instruction


def tokenize_supervised(
    row: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, Any]:
    prompt_messages = [{"role": "user", "content": user_text(row)}]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": str(row["output"]).strip()}
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages, tokenize=True, add_generation_prompt=False
    )
    prompt_ids = extract_int_list(prompt_ids)
    full_ids = extract_int_list(full_ids)
    common_prefix = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        common_prefix += 1
    answer_ids = full_ids[common_prefix:]
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        prompt_length = max_length - min(len(answer_ids), max_length)
    else:
        prompt_length = common_prefix
    if not full_ids:
        full_ids = [tokenizer.eos_token_id or 0]
        prompt_length = 0
    if prompt_length >= len(full_ids):
        prompt_length = len(full_ids) - 1
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    return {
        "input_ids": full_ids,
        "labels": labels,
        "sample_id": row["sample_id"],
        "total_tokens": len(full_ids),
        "answer_tokens": sum(x != -100 for x in labels),
    }



def extract_int_list(obj):
    if hasattr(obj, "tolist"):
        return extract_int_list(obj.tolist())
    if hasattr(obj, "ids"):
        return [int(i) for i in obj.ids]
    if hasattr(obj, "keys") and "input_ids" in obj:
        return extract_int_list(obj["input_ids"])
    if isinstance(obj, (list, tuple)):
        res = []
        for item in obj:
            if hasattr(item, "ids"):
                res.extend([int(i) for i in item.ids])
            elif isinstance(item, (list, tuple)):
                res.extend(extract_int_list(item))
            else:
                try:
                    res.append(int(item))
                except (TypeError, ValueError):
                    res.extend(extract_int_list(item))
        return res
    return [int(obj)]

def collate_supervised(
    items: list[dict[str, Any]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    width = max(len(extract_int_list(item["input_ids"])) for item in items)
    input_ids, labels, masks = [], [], []
    for item in items:
        pad = width - len(extract_int_list(item["input_ids"]))
        input_ids.append(extract_int_list(item["input_ids"]) + [pad_token_id] * pad)
        labels.append(extract_int_list(item["labels"]) + [-100] * pad)
        masks.append([1] * len(extract_int_list(item["input_ids"])) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


@torch.inference_mode()
def per_example_answer_nll(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    tokenized = [tokenize_supervised(row, tokenizer, max_length) for row in rows]
    output = []
    was_training = model.training
    model.eval()
    for start in range(0, len(tokenized), batch_size):
        chunk = tokenized[start : start + batch_size]
        batch = collate_supervised(chunk, tokenizer.pad_token_id)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        losses = F.cross_entropy(
            logits[:, :-1, :].float().transpose(1, 2),
            labels[:, 1:],
            reduction="none",
            ignore_index=-100,
        )
        valid = labels[:, 1:].ne(-100)
        for index, item in enumerate(chunk):
            token_count = int(valid[index].sum().item())
            if token_count == 0:
                nll_sum = float("nan")
                mean_nll = float("nan")
            else:
                nll_sum = float(losses[index][valid[index]].sum().item())
                mean_nll = nll_sum / token_count
            output.append(
                {
                    "sample_id": item["sample_id"],
                    "nll_sum": nll_sum,
                    "answer_tokens": token_count,
                    "mean_nll": mean_nll,
                }
            )
    if was_training:
        model.train()
    return output


def resolve_dtype(name: str | None = None) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_name: str,
    revision: str = "main",
    adapter_path: str | None = None,
    train_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target: str | list[str] = "all-linear",
    gradient_checkpointing: bool = False,
) -> tuple[Any, Any, torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=dtype
    ).to(device)
    if adapter_path:
        model = PeftModel.from_pretrained(
            model, adapter_path, is_trainable=train_lora
        )
    elif train_lora:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target,
            bias="none",
        )
        model = get_peft_model(model, config)
    if train_lora and gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    return model, tokenizer, device, dtype


def aggregate_nll(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        group = str(row[key])
        slot = groups.setdefault(
            group, {"nll_sum": 0.0, "answer_tokens": 0, "examples": 0}
        )
        slot["nll_sum"] += float(row["nll_sum"])
        slot["answer_tokens"] += int(row["answer_tokens"])
        slot["examples"] += 1
    result = {}
    for group, slot in sorted(groups.items()):
        mean_nll = slot["nll_sum"] / max(1, slot["answer_tokens"])
        result[group] = {
            "examples": slot["examples"],
            "answer_tokens": slot["answer_tokens"],
            "mean_nll": mean_nll,
            "ppl": float(np.exp(min(20.0, mean_nll))),
        }
    return result
