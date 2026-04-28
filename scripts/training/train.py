"""Unified training entry point for both baseline FP16 and BitNet b1.58 models.

Usage:
    python scripts/training/train.py --config configs/bitnet_b158_125M.yaml
    python scripts/training/train.py --config configs/baseline_fp16.yaml

The config YAML controls every aspect of the run (model architecture,
optimizer, dataset, checkpointing).  See configs/ for annotated examples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

# Make the project root importable when running this file directly.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baseline.transformer import DecoderTransformer, ModelConfig
from models.bitnet.bitnet_b158 import BitNetTransformer
from utils.metrics_tracker import MetricsTracker


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def model_config_from_dict(d: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**{k: v for k, v in d.items() if k in ModelConfig.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_wikitext2(split: str, seq_len: int, tokenizer_name: str = "gpt2"):
    """Return a flat token tensor for the requested WikiText-2 split.

    Uses HuggingFace `datasets` + `transformers` (must be installed).
    First call downloads ~5 MB; subsequent calls use the cache.
    """
    from datasets import load_dataset  # type: ignore[import]
    from transformers import AutoTokenizer  # type: ignore[import]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=True)
    texts = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    tokens = tokenizer.encode(texts, add_special_tokens=False)
    return torch.tensor(tokens, dtype=torch.long)


class TokenDataset(torch.utils.data.Dataset):
    """Chunk a flat token tensor into (input, target) pairs of length seq_len."""

    def __init__(self, tokens: torch.Tensor, seq_len: int) -> None:
        self.tokens = tokens
        self.seq_len = seq_len
        # Drop the last incomplete chunk
        self.n_chunks = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


# ---------------------------------------------------------------------------
# Optimizer + LR schedule
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    lr = cfg.get("learning_rate", 3e-4)
    wd = cfg.get("weight_decay", 0.1)
    # Don't apply weight decay to embedding / norm / bias parameters.
    decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and p.ndim >= 2 and "emb" not in n
    ]
    nodecay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and (p.ndim < 2 or "emb" in n)
    ]
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": wd},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        eps=1e-8,
    )


def cosine_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    scale = min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return lr * scale


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    config: dict[str, Any],
    ckpt_dir: Path,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )
    print(f"  [ckpt] saved → {path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: dict[str, Any]) -> None:
    torch.manual_seed(cfg.get("training", {}).get("seed", 42))
    random.seed(cfg.get("training", {}).get("seed", 42))

    # ---- Directories -------------------------------------------------------
    out_dir = Path(cfg.get("output", {}).get("dir", "results/logs/run"))
    ckpt_dir = Path(cfg.get("output", {}).get("checkpoint_dir", str(out_dir / "checkpoints")))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Device ------------------------------------------------------------
    device_str = cfg.get("training", {}).get("device", "cpu")
    device = torch.device(device_str)
    dtype_str = cfg.get("training", {}).get("dtype", "float32")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    print(f"Device: {device}  dtype: {dtype_str}")

    # ---- Model -------------------------------------------------------------
    model_cfg_dict = cfg.get("model", {})
    model_config = model_config_from_dict(model_cfg_dict)

    model_type = model_cfg_dict.get("model_type", "baseline")
    if model_type == "bitnet_b158":
        model = BitNetTransformer(model_config)
    else:
        model = DecoderTransformer(model_config)

    model = model.to(device=device, dtype=dtype)
    n_params = model.num_parameters()
    print(f"Model: {model_type}  |  params: {n_params:,}  ({n_params/1e6:.1f}M)")

    # ---- Dataset -----------------------------------------------------------
    train_cfg = cfg.get("training", {})
    seq_len = model_config.max_seq_len
    tokenizer = cfg.get("dataset", {}).get("tokenizer", "gpt2")

    print("Loading WikiText-2 …")
    train_tokens = load_wikitext2("train", seq_len, tokenizer)
    val_tokens   = load_wikitext2("validation", seq_len, tokenizer)

    train_ds = TokenDataset(train_tokens, seq_len)
    val_ds   = TokenDataset(val_tokens, seq_len)

    batch_size = train_cfg.get("batch_size", 8)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device_str != "cpu"),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )

    # ---- Optimizer ---------------------------------------------------------
    optimizer = build_optimizer(model, train_cfg)

    max_steps       = train_cfg.get("max_steps", 10_000)
    warmup_steps    = train_cfg.get("warmup_steps", 500)
    grad_accum      = train_cfg.get("gradient_accumulation", 1)
    max_grad_norm   = train_cfg.get("max_grad_norm", 1.0)
    lr              = train_cfg.get("learning_rate", 3e-4)
    log_every       = train_cfg.get("log_every", 100)
    eval_every      = train_cfg.get("eval_every", 500)
    checkpoint_every= train_cfg.get("checkpoint_every", 1000)

    # ---- Metrics -----------------------------------------------------------
    tracker = MetricsTracker(out_dir)
    tracker.start()

    # ---- Training loop -----------------------------------------------------
    step = 0
    accum_loss = 0.0
    t0 = time.time()
    tokens_seen = 0

    model.train()
    data_iter = iter(train_loader)

    print(f"\nStarting training for {max_steps} steps …\n")

    while step < max_steps:
        # Fetch a batch (restart iterator when exhausted)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        # Forward + backward (gradient accumulation)
        _, loss = model(x, y)
        (loss / grad_accum).backward()
        accum_loss += loss.item() / grad_accum
        tokens_seen += x.numel()

        if (step + 1) % grad_accum == 0 or step == max_steps - 1:
            # Update LR
            current_lr = cosine_lr(step, max_steps, warmup_steps, lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            avg_loss = accum_loss
            accum_loss = 0.0

            elapsed = time.time() - t0
            tps = tokens_seen / elapsed if elapsed > 0 else 0.0

            if step % log_every == 0:
                print(
                    f"step {step:6d}/{max_steps}  loss={avg_loss:.4f}  "
                    f"lr={current_lr:.2e}  mem={tracker.peak_memory_mb:.0f}MB  "
                    f"tok/s={tps:.0f}"
                )
                tracker.log_step(step, avg_loss, tokens_per_sec=tps, lr=current_lr)

        # ---- Validation ----------------------------------------------------
        if step > 0 and step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"  [val] step {step}  val_loss={val_loss:.4f}  ppl={math.exp(min(val_loss,20)):.2f}")

        # ---- Checkpoint ----------------------------------------------------
        if step > 0 and step % checkpoint_every == 0:
            save_checkpoint(model, optimizer, step, avg_loss, cfg, ckpt_dir)

        step += 1

    # ---- Final validation --------------------------------------------------
    val_loss = evaluate(model, val_loader, device)
    print(f"\n[final] val_loss={val_loss:.4f}  ppl={math.exp(min(val_loss,20)):.2f}")

    # ---- BitNet weight stats (if applicable) -------------------------------
    if isinstance(model, BitNetTransformer):
        stats = model.ternary_weight_stats()
        print(f"[ternary] {stats}")

    # ---- Save artifacts ----------------------------------------------------
    summary_path = tracker.stop_and_save(model_cfg_dict)
    save_checkpoint(model, optimizer, step, val_loss, cfg, ckpt_dir)
    print(f"\nDone. Run summary → {summary_path}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Return mean cross-entropy loss over up to max_batches validation batches."""
    model.eval()
    total_loss = 0.0
    n = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline or BitNet b1.58 model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
