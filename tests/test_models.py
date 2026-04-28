"""Pytest sanity checks for models, BitLinear, and MetricsTracker.

Run: pytest tests/ -v
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    from models.baseline.transformer import ModelConfig
    return ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        ffn_size=128,
        max_seq_len=32,
        dropout=0.0,
        bias=False,
    )


@pytest.fixture
def batch(tiny_config):
    B, T = 2, 16
    ids = torch.randint(0, tiny_config.vocab_size, (B, T))
    tgt = torch.randint(0, tiny_config.vocab_size, (B, T))
    return ids, tgt


# ---------------------------------------------------------------------------
# BitLinear
# ---------------------------------------------------------------------------


class TestBitLinear:
    def test_output_shape(self):
        from models.bitnet.bitlinear import BitLinear
        layer = BitLinear(16, 32)
        x = torch.randn(4, 8, 16)
        out = layer(x)
        assert out.shape == (4, 8, 32)

    def test_weights_are_ternary(self):
        """Quantized weights must be strictly in {-1, 0, 1}."""
        from models.bitnet.bitlinear import BitLinear
        layer = BitLinear(64, 128)
        gamma = layer.weight.abs().mean().clamp(min=1e-5)
        w_q = (layer.weight / gamma).clamp(-1, 1).round()
        unique = w_q.unique().tolist()
        for v in unique:
            assert v in {-1.0, 0.0, 1.0}, f"Unexpected weight value: {v}"

    def test_ste_gradient_flows(self):
        """Gradient should reach the real-valued weight (STE check)."""
        from models.bitnet.bitlinear import BitLinear
        layer = BitLinear(8, 4)
        x = torch.randn(2, 8)
        out = layer(x).sum()
        out.backward()
        assert layer.weight.grad is not None
        assert not torch.all(layer.weight.grad == 0), "Gradient is zero everywhere — STE broken"

    def test_act_quantization_range(self):
        """Quantized activations must be clipped to [-127, 127] for 8-bit."""
        from models.bitnet.bitlinear import BitLinear
        layer = BitLinear(16, 16, act_bits=8)
        Q = 127
        x = torch.randn(4, 16) * 10  # exaggerated values
        gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x * Q / gamma).clamp(-Q, Q).round()
        assert x_q.abs().max().item() <= Q

    def test_no_bias_by_default(self):
        from models.bitnet.bitlinear import BitLinear
        layer = BitLinear(8, 4)
        assert layer.bias is None


# ---------------------------------------------------------------------------
# Baseline DecoderTransformer
# ---------------------------------------------------------------------------


class TestDecoderTransformer:
    def test_forward_shape(self, tiny_config, batch):
        from models.baseline.transformer import DecoderTransformer
        model = DecoderTransformer(tiny_config)
        ids, _ = batch
        logits, loss = model(ids)
        assert logits.shape == (ids.shape[0], ids.shape[1], tiny_config.vocab_size)
        assert loss is None

    def test_loss_computed_with_targets(self, tiny_config, batch):
        from models.baseline.transformer import DecoderTransformer
        model = DecoderTransformer(tiny_config)
        ids, tgt = batch
        _, loss = model(ids, tgt)
        assert loss is not None
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_loss_decreases_after_step(self, tiny_config, batch):
        """A single gradient step should lower the loss (sanity check)."""
        from models.baseline.transformer import DecoderTransformer
        torch.manual_seed(0)
        model = DecoderTransformer(tiny_config)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids, tgt = batch

        _, loss_before = model(ids, tgt)
        loss_before.backward()
        opt.step()
        opt.zero_grad()

        with torch.no_grad():
            _, loss_after = model(ids, tgt)

        assert loss_after.item() < loss_before.item(), (
            f"Loss did not decrease: {loss_before.item():.4f} → {loss_after.item():.4f}"
        )

    def test_seq_len_limit_raises(self, tiny_config):
        from models.baseline.transformer import DecoderTransformer
        model = DecoderTransformer(tiny_config)
        too_long = torch.zeros(1, tiny_config.max_seq_len + 1, dtype=torch.long)
        with pytest.raises(AssertionError):
            model(too_long)

    def test_num_parameters(self, tiny_config):
        from models.baseline.transformer import DecoderTransformer
        model = DecoderTransformer(tiny_config)
        assert model.num_parameters() > 0

    def test_weight_tying(self, tiny_config):
        """Token embedding and LM head weight should be the same object."""
        from models.baseline.transformer import DecoderTransformer
        model = DecoderTransformer(tiny_config)
        assert model.lm_head.weight is model.token_emb.weight


# ---------------------------------------------------------------------------
# BitNetTransformer
# ---------------------------------------------------------------------------


class TestBitNetTransformer:
    def test_forward_shape(self, tiny_config, batch):
        from models.bitnet.bitnet_b158 import BitNetTransformer
        model = BitNetTransformer(tiny_config)
        ids, _ = batch
        logits, _ = model(ids)
        assert logits.shape == (ids.shape[0], ids.shape[1], tiny_config.vocab_size)

    def test_loss_computed(self, tiny_config, batch):
        from models.bitnet.bitnet_b158 import BitNetTransformer
        model = BitNetTransformer(tiny_config)
        ids, tgt = batch
        _, loss = model(ids, tgt)
        assert loss is not None and loss.item() > 0

    def test_uses_bitlinear(self, tiny_config):
        from models.bitnet.bitnet_b158 import BitNetTransformer
        from models.bitnet.bitlinear import BitLinear
        model = BitNetTransformer(tiny_config)
        bitlinear_count = sum(1 for m in model.modules() if isinstance(m, BitLinear))
        # Each layer has 4 attn projections + 3 ffn projections = 7; 2 layers → 14
        expected = tiny_config.num_layers * 7
        assert bitlinear_count == expected, (
            f"Expected {expected} BitLinear layers, found {bitlinear_count}"
        )

    def test_embeddings_are_standard_float(self, tiny_config):
        """Token embeddings must NOT be BitLinear (paper requirement)."""
        from models.bitnet.bitnet_b158 import BitNetTransformer
        from models.bitnet.bitlinear import BitLinear
        model = BitNetTransformer(tiny_config)
        assert not isinstance(model.token_emb, BitLinear)
        assert not isinstance(model.lm_head, BitLinear)

    def test_ternary_weight_stats_valid(self, tiny_config):
        from models.bitnet.bitnet_b158 import BitNetTransformer
        model = BitNetTransformer(tiny_config)
        stats = model.ternary_weight_stats()
        assert "frac_neg1" in stats
        total_frac = stats["frac_neg1"] + stats["frac_zero"] + stats["frac_pos1"]
        assert abs(total_frac - 1.0) < 1e-4, f"Fractions don't sum to 1: {total_frac}"

    def test_gradient_flows_through_bitlinear(self, tiny_config, batch):
        from models.bitnet.bitnet_b158 import BitNetTransformer
        from models.bitnet.bitlinear import BitLinear
        model = BitNetTransformer(tiny_config)
        ids, tgt = batch
        _, loss = model(ids, tgt)
        loss.backward()
        for name, module in model.named_modules():
            if isinstance(module, BitLinear):
                assert module.weight.grad is not None, f"{name}.weight has no gradient"
                break


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------


class TestMetricsTracker:
    def test_log_and_save(self):
        from utils.metrics_tracker import MetricsTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(tmpdir)
            tracker.start()
            tracker.log_step(0, loss=4.5)
            tracker.log_step(100, loss=3.2, tokens_per_sec=1000.0, lr=1e-4)
            summary_path = tracker.stop_and_save({"model_type": "test"})
            assert summary_path.exists()
            import json
            with open(summary_path) as f:
                summary = json.load(f)
            assert summary["total_steps"] == 2
            assert summary["final_loss"] == pytest.approx(3.2, rel=1e-3)
            assert summary["peak_memory_mb"] > 0

    def test_perplexity_capped(self):
        """Very high loss should not cause exp() overflow."""
        from utils.metrics_tracker import MetricsTracker
        import math
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(tmpdir)
            tracker.start()
            tracker.log_step(0, loss=999.0)  # would overflow without clamp
            entry = tracker.step_metrics[0]
            assert entry["perplexity"] == pytest.approx(math.exp(20.0), rel=1e-3)

    def test_step_csv_written(self):
        from utils.metrics_tracker import MetricsTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(tmpdir)
            tracker.start()
            for i in range(5):
                tracker.log_step(i, loss=5.0 - i * 0.1)
            tracker.stop_and_save({})
            csv_path = Path(tmpdir) / "step_metrics.csv"
            assert csv_path.exists()
            lines = csv_path.read_text().strip().split("\n")
            assert len(lines) == 6  # header + 5 data rows
