"""BitLinear: ternary-weight linear layer for BitNet b1.58.

Weight quantization  — absmean normalization → RoundClip → {-1, 0, 1}
Activation quantization — per-token absmax → 8-bit integers
Gradient flow — Straight-Through Estimator (STE) through both quantizers

Reference: Ma et al. (2024) "The Era of 1-bit LLMs" arXiv:2402.17764
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with ternary weights {-1, 0, 1}.

    During training the quantization is *simulated* in floating point —
    weights are stored as float32/bf16 and quantized in each forward pass
    via STE so gradients flow back to the real-valued parameters.

    At inference the ternary weights could be packed to 2 bits and matmuls
    replaced with efficient integer kernels (not implemented here).

    Args:
        in_features:  input feature dimension
        out_features: output feature dimension
        bias:         add a learnable bias (default False, matches LLaMA style)
        act_bits:     activation quantization bit-width (default 8)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        act_bits: int = 8,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.act_bits = act_bits

    # ------------------------------------------------------------------
    # Quantizers
    # ------------------------------------------------------------------

    def _quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Ternary weight quantization with STE.

        γ = mean(|W|)               # global scale for this layer
        Ŵ = RoundClip(W / γ, -1, 1) # ternary: {-1, 0, 1}

        STE trick: return w + (Ŵ - w).detach()
          • forward  → evaluates to Ŵ (quantized)
          • backward → gradient of w (identity through quantizer)
        """
        gamma = w.abs().mean().clamp(min=1e-5)
        w_quant = (w / gamma).clamp(-1.0, 1.0).round()
        return w + (w_quant - w).detach()

    def _quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token absmax quantization to act_bits with STE.

        η = max(|x|) per token      # one scale per (batch × time) position
        x̂ = Clamp(x · Q / η, -Q, Q) # Q = 2^(act_bits-1) - 1

        STE: return x + (x̂ - x).detach()
        """
        Q = 2 ** (self.act_bits - 1) - 1  # 127 for 8-bit
        # x may be (B, T, C) or (B*T, C) — always quantize along last dim
        eta = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_quant = (x * Q / eta).clamp(-Q, Q).round()
        return x + (x_quant - x).detach()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self._quantize_weights(self.weight)
        x_q = self._quantize_activations(x)
        return F.linear(x_q, w_q, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, act_bits={self.act_bits}"
        )
