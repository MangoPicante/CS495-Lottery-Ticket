"""BitNet b1.58 — decoder-only transformer with ternary weights {-1, 0, 1}.

Identical architecture to models/baseline/transformer.py except that every
linear projection inside attention and FFN blocks is replaced with BitLinear.
Token embeddings and the LM head remain in full floating-point precision, as
specified in the original paper.

Reference: Ma et al. (2024) "The Era of 1-bit LLMs" arXiv:2402.17764
"""

from __future__ import annotations

import torch

from models.baseline.transformer import DecoderTransformer, ModelConfig
from models.bitnet.bitlinear import BitLinear


class BitNetTransformer(DecoderTransformer):
    """BitNet b1.58 model.

    Inherits the full architecture from DecoderTransformer but passes
    BitLinear as the linear_cls so all Q/K/V/O and FFN projections use
    ternary weights with 8-bit activation quantization.

    Token embeddings and the LM head use standard float (nn.Linear /
    nn.Embedding) — exactly as described in the paper.
    """

    def __init__(self, config: ModelConfig) -> None:
        # Override model_type for run logging
        config.model_type = "bitnet_b158"
        super().__init__(config, linear_cls=BitLinear)

    @staticmethod
    def from_config(config: ModelConfig) -> "BitNetTransformer":  # type: ignore[override]
        return BitNetTransformer(config)

    def ternary_weight_stats(self) -> dict[str, float]:
        """Return the fraction of weights at each ternary value {-1, 0, +1}.

        Useful for verifying that quantization is working correctly — a healthy
        distribution is roughly 1/3 each, skewed by the data distribution.
        """
        neg = zero = pos = total = 0
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                with torch.no_grad():
                    gamma = module.weight.abs().mean().clamp(min=1e-5)
                    w_q = (module.weight / gamma).clamp(-1, 1).round()
                    neg   += (w_q == -1).sum().item()
                    zero  += (w_q ==  0).sum().item()
                    pos   += (w_q ==  1).sum().item()
                    total += w_q.numel()
        if total == 0:
            return {}
        return {
            "frac_neg1": neg  / total,
            "frac_zero": zero / total,
            "frac_pos1": pos  / total,
            "total_weights": total,
        }
