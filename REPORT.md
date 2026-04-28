# REPORT.md — BitNet b1.58 Capstone Research Notes

**Author:** Sean Michael  
**Course:** CS 495 — Non-GPU LLM Training  
**Date:** April 27, 2026

---

## Phase 1 — Repository Study & Architecture Analysis

### 1. Absmean Quantization Function

BitNet b1.58 (Ma et al., 2024) trains transformer models with **ternary weights** drawn from {−1, 0, +1}. The quantization function used during the forward pass is:

#### Weight Quantization

Given a weight matrix **W** ∈ ℝ^(d_out × d_in):

```
γ = mean(|W|)                    # global scale factor for the layer
W̃ = W / (γ + ε)                  # normalize by absolute mean (ε = 1e-5)
Ŵ = RoundClip(W̃, −1, +1)         # round to nearest integer, then clip to {-1, 0, 1}
```

where `RoundClip(x, a, b) = max(a, min(b, round(x)))`.

The result is a ternary matrix: each weight is exactly −1, 0, or +1. This achieves 1.58 bits of information per weight (log₂(3) ≈ 1.58), giving the method its name.

**Implementation** (`models/bitnet/bitlinear.py`):
```python
gamma = w.abs().mean().clamp(min=1e-5)
w_quant = (w / gamma).clamp(-1.0, 1.0).round()
```

#### Activation Quantization

Activations are quantized per-token to 8-bit integers using absmax normalization:

```
η = max(|x_t|)                   # max absolute value for token position t
x̂_t = Clamp(x_t · Q_b / η, −Q_b, Q_b)   # Q_b = 2^(b-1)−1 = 127 for 8-bit
```

This gives one scale factor per token (rather than one per layer), which preserves outlier information at the token level.

---

### 2. Straight-Through Estimator (STE)

Rounding functions have zero gradient almost everywhere, which would make it impossible to train quantized models with standard backpropagation. The **Straight-Through Estimator** (Bengio et al., 2013) resolves this by substituting an identity gradient in the backward pass:

```
Forward:   f(x) = round(x)        # discrete, quantized output
Backward:  ∂L/∂x ≈ ∂L/∂f         # pretend round() is the identity
```

In PyTorch this is implemented using the `.detach()` trick:

```python
# forward evaluates to w_quant; backward gradient is w.grad (not w_quant.grad)
w_ste = w + (w_quant - w).detach()
```

**Why it works:** `w + (w_quant - w).detach()` evaluates to `w_quant` numerically, but because `(w_quant - w)` is detached, its gradient is treated as a constant zero. The only gradient-contributing term is `w`, so `∂(w_ste)/∂w = 1` — the identity.

**Justification:** The STE approximation holds well in practice when the quantization step is small relative to the parameter magnitudes and the training signal is strong. BitNet b1.58 uses a warmup phase to stabilize weights before quantization becomes sharp.

---

### 3. Published FP16 Baseline Results

The following results are drawn from published papers and serve as comparison targets. Our ternary models will be evaluated against these numbers on the **WikiText-2** perplexity metric.

#### LLaMA (Touvron et al., 2023) — Trained on 1T tokens (The Pile / CommonCrawl)

| Model     | Params | WikiText-2 PPL ↓ | HellaSwag | ARC-e | WinoGrande |
|-----------|--------|-------------------|-----------|-------|------------|
| LLaMA-7B  | 6.7B   | 5.68              | 76.1%     | 72.8% | 69.6%      |
| LLaMA-13B | 13B    | 5.09              | 79.9%     | 77.8% | 73.0%      |
| LLaMA-30B | 30B    | 4.10              | 82.9%     | 78.9% | 75.7%      |
| LLaMA-65B | 65B    | 3.53              | 84.2%     | 80.0% | 77.0%      |

#### StableLM (Stability AI, 2023) — Trained on 1.5T tokens

| Model        | Params | WikiText-2 PPL ↓ | Notes                          |
|--------------|--------|-------------------|--------------------------------|
| StableLM-3B  | 3B     | ~14.0             | Estimated; exact figure varies |
| StableLM-7B  | 7B     | ~7.5              | Similar to LLaMA-7B            |

#### BitNet b1.58 (Ma et al., 2024) — Trained on RedPajama (comparable token budget to LLaMA)

These are the results from the original paper that our training will attempt to reproduce at smaller scale:

| Model           | Params | WikiText-2 PPL ↓ | LLaMA PPL (same size) | HellaSwag |
|-----------------|--------|-------------------|-----------------------|-----------|
| BitNet b1.58 700M | 700M | 12.87             | 13.52                 | 52.3%     |
| BitNet b1.58 1.3B | 1.3B | 11.08             | 11.74                 | 55.1%     |
| BitNet b1.58 3B   | 3B   | 9.91              | 10.98                 | 59.4%     |
| BitNet b1.58 7B   | 7B   | 9.45              | 10.29                 | 63.0%     |

**Key finding:** By 3B parameters, BitNet b1.58 matches or surpasses FP16 LLaMA on perplexity and zero-shot tasks, while using ~3× less memory and running ~2.7× faster on CPU hardware (per the paper).

---

### 4. Finalized Model Sizes for This Project

Based on the Phase 1 analysis and the project's CPU-only constraint, we will train the following ternary model sizes:

| Config file                   | Params (approx) | Layers | Hidden | Heads | FFN   | Purpose                         |
|-------------------------------|-----------------|--------|--------|-------|-------|----------------------------------|
| `configs/bitnet_b158.yaml`    | ~125M           | 12     | 768    | 12    | 2048  | Phase 2 validation; fast to run  |
| `configs/bitnet_b158_700M.yaml` | ~700M         | 24     | 1536   | 12    | 4096  | Phase 3 primary experiment        |
| `configs/bitnet_b158_1B3.yaml`  | ~1.1B         | 16     | 2048   | 16    | 8192  | Phase 3 scaling experiment        |

A matching baseline config (`configs/baseline_fp16.yaml`, ~125M FP32) provides an apples-to-apples comparison for the smallest model. For larger sizes we rely on the published LLaMA numbers above.

**Rationale for size selection:**  
- 125M is small enough to train to convergence in hours on a single CPU, confirming that our implementation is correct before scaling up.  
- 700M is the smallest size for which the BitNet b1.58 paper reports competitive results vs. FP16 baselines — this is our primary experimental target.  
- 1.1B extends the scaling analysis one step further within practical time bounds.

---

## Phase 2 — Baseline Training (to be filled in after runs)

| Run | Model | Steps | Final Loss | Final PPL | Time (h) | Peak Mem (MB) |
|-----|-------|-------|------------|-----------|----------|---------------|
| —   | —     | —     | —          | —         | —        | —             |

---

## Phase 3 — BitNet Results (to be filled in after runs)

| Run | Model | Steps | Final Loss | Final PPL | Time (h) | Peak Mem (MB) | CO₂ (g) |
|-----|-------|-------|------------|-----------|----------|---------------|---------|
| —   | —     | —     | —          | —         | —        | —             | —       |

---

## Phase 4 — Cost Comparison (to be filled in)

*(See `results/plots/comparison/` for generated charts and `results/plots/comparison/comparison.csv` for the full table.)*

---

## References

1. Ma, S., et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764.
2. Wang, H., et al. (2023). *BitNet: Scaling 1-bit Transformers for Large Language Models.* arXiv:2310.11453.
3. Touvron, H., et al. (2023). *LLaMA: Open and Efficient Foundation Language Models.* arXiv:2302.13971.
4. Bengio, Y., Léonard, N., & Courville, A. (2013). *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* arXiv:1308.3432.
5. Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization.* NeurIPS 2019.
