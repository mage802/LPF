#!/usr/bin/env python3
"""Low-cost deterministic coordinate sketches of LoRA gradients."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from upstream import ensure_week23_scripts

ensure_week23_scripts()
from common import collate_supervised, tokenize_supervised  # noqa: E402


def _stable_int(text: str) -> int:
    return int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
    )


def gradient_sketch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    device: torch.device,
    sketch_dim: int,
    coords_per_tensor: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int | None]]:
    """Backprop a tiny batch and hash sampled LoRA coordinates into a sketch."""
    model.train()
    model.zero_grad(set_to_none=True)
    items = [tokenize_supervised(row, tokenizer, max_length) for row in rows]
    items = [item for item in items if item["answer_tokens"] > 0]
    if not items:
        return torch.zeros(sketch_dim), {
            "loss": None,
            "full_tokens": 0,
            "answer_tokens": 0,
            "sampled_gradient_coordinates": 0,
        }
    batch = collate_supervised(items, tokenizer.pad_token_id)
    batch = {key: value.to(device) for key, value in batch.items()}
    loss = model(**batch).loss
    loss.backward()

    sketch = torch.zeros(sketch_dim, dtype=torch.float32, device=device)
    sampled_coordinates = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        flat = parameter.grad.detach().float().reshape(-1)
        count = min(coords_per_tensor, flat.numel())
        if count <= 0:
            continue
        name_seed = (_stable_int(name) + seed) % max(1, flat.numel())
        stride = max(1, flat.numel() // count)
        coordinate = (
            torch.arange(count, device=device, dtype=torch.long) * stride
            + name_seed
        ) % flat.numel()
        values = flat[coordinate]
        scale = (flat.numel() / count) ** 0.5
        values = values * scale

        hash_base = _stable_int(f"{seed}:{name}") % 2_147_483_647
        bucket = (
            torch.arange(count, device=device, dtype=torch.long) * 2654435761
            + hash_base
        ) % sketch_dim
        sign_bits = (
            torch.arange(count, device=device, dtype=torch.long) * 2246822519
            + hash_base
        ) % 2
        signs = sign_bits.float().mul_(2.0).sub_(1.0)
        sketch.scatter_add_(0, bucket, values * signs)
        sampled_coordinates += count

    norm = torch.linalg.vector_norm(sketch)
    if float(norm.item()) > 0:
        sketch = sketch / norm
    stats = {
        "loss": float(loss.detach().item()),
        "full_tokens": sum(int(item["total_tokens"]) for item in items),
        "answer_tokens": sum(int(item["answer_tokens"]) for item in items),
        "sampled_gradient_coordinates": sampled_coordinates,
        "sketch_norm_before_normalization": float(norm.item()),
    }
    model.zero_grad(set_to_none=True)
    return sketch.detach().cpu(), stats


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    denominator = float((left_norm * right_norm).item())
    if denominator <= 0:
        return 0.0
    return float(torch.dot(left, right).item() / denominator)
