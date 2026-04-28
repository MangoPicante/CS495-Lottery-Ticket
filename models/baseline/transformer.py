"""Decoder-only transformer (GPT / LLaMA style) for the FP16/BF16/FP32 baseline.

Architecture choices:
  - Pre-layer RMSNorm (LLaMA style)
  - SwiGLU feed-forward (gate × up → down)
  - Learned positional embeddings (simpler than RoPE; sufficient for WikiText-2)
  - Causal self-attention via F.scaled_dot_product_attention
  - No bias in linear projections (matches LLaMA)
  - Weight-tied token embeddings and LM head

The same building blocks (RMSNorm, CausalSelfAttention, SwiGLUFeedForward,
TransformerBlock) are reused in bitnet_b158.py — the only difference is that
BitNet replaces nn.Linear with BitLinear in attention and FFN layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Hyper-parameters that define a model variant.

    Approximate parameter counts (including weight-tied embeddings):
      125M  → layers=12, hidden=768,  heads=12, ffn_size=2048
      700M  → layers=24, hidden=1536, heads=12, ffn_size=4096
      1.3B  → layers=16, hidden=2048, heads=16, ffn_size=8192
      3B    → layers=28, hidden=2560, heads=20, ffn_size=10240
    """

    vocab_size: int = 50257       # GPT-2 / tiktoken cl100k_base
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_size: int = 2048          # SwiGLU intermediate dim
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = False            # No bias in linear layers (LLaMA style)
    # Extra meta-data (not used by the model, stored in run_summary.json)
    model_type: str = "baseline"
    name: str = "baseline_125M"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    The linear_cls argument lets BitNet substitute BitLinear for nn.Linear
    while keeping everything else identical.
    """

    def __init__(self, config: ModelConfig, linear_cls: type = nn.Linear) -> None:
        super().__init__()
        assert config.hidden_size % config.num_heads == 0, (
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"num_heads ({config.num_heads})"
        )
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.dropout_p = config.dropout

        self.q_proj = linear_cls(config.hidden_size, config.hidden_size, bias=config.bias)
        self.k_proj = linear_cls(config.hidden_size, config.hidden_size, bias=config.bias)
        self.v_proj = linear_cls(config.hidden_size, config.hidden_size, bias=config.bias)
        self.o_proj = linear_cls(config.hidden_size, config.hidden_size, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Flash / memory-efficient causal attention (PyTorch ≥ 2.0)
        attn_dropout = self.dropout_p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=attn_dropout)

        x = x.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(x)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward block: down(silu(gate(x)) * up(x)).

    Uses three projections instead of two — the gate projection is the extra
    one compared to a standard two-layer MLP.
    """

    def __init__(self, config: ModelConfig, linear_cls: type = nn.Linear) -> None:
        super().__init__()
        self.gate_proj = linear_cls(config.hidden_size, config.ffn_size, bias=config.bias)
        self.up_proj   = linear_cls(config.hidden_size, config.ffn_size, bias=config.bias)
        self.down_proj = linear_cls(config.ffn_size, config.hidden_size, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: Norm → Attn → residual, Norm → FFN → residual."""

    def __init__(self, config: ModelConfig, linear_cls: type = nn.Linear) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn      = CausalSelfAttention(config, linear_cls)
        self.ffn_norm  = RMSNorm(config.hidden_size)
        self.ffn       = SwiGLUFeedForward(config, linear_cls)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class DecoderTransformer(nn.Module):
    """Decoder-only transformer for causal language modeling.

    Suitable as an FP16/BF16/FP32 baseline.  Pass linear_cls=BitLinear to
    BitNetTransformer instead of instantiating this directly.
    """

    def __init__(self, config: ModelConfig, linear_cls: type = nn.Linear) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.drop_emb  = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(config, linear_cls) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size)

        # LM head — weight tied to token embeddings (reduces parameters)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        pos = torch.arange(T, device=input_ids.device)
        x = self.drop_emb(self.token_emb(input_ids) + self.pos_emb(pos))

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.token_emb.weight.numel()
        return total

    @staticmethod
    def from_config(config: ModelConfig) -> "DecoderTransformer":
        return DecoderTransformer(config, linear_cls=nn.Linear)
