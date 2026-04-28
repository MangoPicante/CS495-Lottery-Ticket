# Non-GPU LLM Training: Efficient Transformer Training with BitNet 📌

![Data Science Capstone](https://img.shields.io/badge/Data%20Science-Capstone-blue)
![BitNet](https://img.shields.io/badge/BitNet-Quantization-teal)
![CPU Training](https://img.shields.io/badge/CPU-Training-orange)

---

## Project Theme 📖

Modern LLM training demands expensive GPU clusters, limiting access and driving up energy costs. This project investigates whether **1-bit weight quantization (BitNet)** makes transformer pretraining viable on commodity CPUs — trading a modest accuracy penalty for dramatic reductions in memory, cost, and carbon footprint.

---

## Objectives 🎯

### Primary Research Questions

- Can 1-bit transformers scale to meaningful language tasks?
- How much accuracy is lost under extreme quantization vs. FP16/FP32 baselines?
- Are CPUs a viable substrate for LLM pretraining with BitNet?
- How does low-precision affect convergence stability?
- What workloads benefit most from low-precision training?

### Methodology

| Phase | Description |
|-------|-------------|
| 1 — Repository Study | Analyze BitNet architecture, quantization strategy, and weight binarization |
| 2 — Baseline Training | Train small transformer with FP16/FP32; record time, memory, accuracy |
| 3 — BitNet Implementation | Train equivalent model with 1-bit weights on CPU or low-resource hardware |
| 4 — Cost Comparison | Benchmark hardware cost, training time, energy, memory, and accuracy |
| 5 — Optimization | Explore hybrid precision, quantization schedules, and scaling laws |

### Deliverables

- **Research report** — quantization theory, BitNet architecture, benchmark results, environmental impact
- **Training pipelines** — baseline + BitNet scripts, configs, and reproducibility instructions
- **Benchmark dashboard** — loss curves, training time, memory, and accuracy comparisons
- **Systems cost analysis** — GPU vs. CPU dollar cost, energy usage, and carbon footprint proxy
- **Final presentation**

### Evaluation Metrics

| Metric | Tool |
|--------|------|
| Perplexity | WikiText-2 validation set |
| Zero-shot accuracy | lm-evaluation-harness (ARC, HellaSwag, WinoGrande…) |
| Training time | `MetricsTracker` (wall-clock) |
| Memory (CPU/GPU) | `psutil` + `pynvml` |
| Energy / CO₂ | `codecarbon` |
| Throughput | tokens/sec logged per step |

---

## 🧰 Tools & Technologies

- **Languages**: Python
- **ML Framework**: PyTorch
- **Model Reference**: [microsoft/BitNet](https://github.com/microsoft/BitNet)
- **Evaluation**: [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **Experiment Tracking**: TensorBoard / Weights & Biases
- **Carbon Tracking**: CodeCarbon
- **Data**: WikiText-2, RedPajama (standard NLP benchmarks)

---

## 🚀 How to Run

```bash
# 1. Clone and install
git clone <your-repo>
cd bitnet-capstone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify setup
python -m pytest tests/ -v

# 3. Train baseline (Phase 2)
python scripts/training/train.py --config configs/baseline_fp16.yaml

# 4. Train BitNet b1.58 (Phase 3)
python scripts/training/train.py --config configs/bitnet_b158.yaml

# 5. Compare results (Phase 4)
python scripts/benchmarking/compare_runs.py \
    --runs results/logs/baseline_fp16/run_summary.json \
            results/logs/bitnet_b158/run_summary.json \
    --output results/plots/comparison
```

---

## Project Structure

```
bitnet-capstone/
├── configs/                    # YAML experiment configs
│   ├── baseline_fp16.yaml      #   Standard FP16 transformer
│   └── bitnet_b158.yaml        #   BitNet b1.58 (ternary weights)
│
├── models/
│   ├── baseline/
│   │   └── transformer.py      # FP16/BF16/FP32 decoder-only transformer
│   └── bitnet/
│       ├── bitlinear.py        # BitLinear: absmean quantization + STE
│       └── bitnet_b158.py      # Full BitNet b1.58 model
│
├── scripts/
│   ├── training/
│   │   └── train.py            # Unified training entry point
│   ├── evaluation/             # (Phase 4) lm-eval-harness wrappers
│   └── benchmarking/
│       └── compare_runs.py     # Load JSON logs → plots + tables
│
├── utils/
│   └── metrics_tracker.py      # Time / memory / energy / perplexity tracker
│
├── results/
│   ├── checkpoints/            # Saved model weights
│   ├── logs/                   # run_summary.json + step_metrics.csv
│   └── plots/                  # Generated figures
│
├── tests/
│   └── test_models.py          # Pytest sanity checks
│
├── notebooks/                  # EDA, prototyping, dashboard exploration
├── docs/                       # Design notes, references
└── requirements.txt
```

---

## 📅 Timeline

| Week | Phase |
|------|-------|
| 1–2  | Phase 1 — Repository study & architecture review |
| 3–4  | Phase 2 — Baseline FP16 training & profiling |
| 5–6  | Phase 3 — BitNet b1.58 implementation & CPU training |
| 7–8  | Phase 4 — Cost comparison & benchmark dashboard |
| 9–10 | Phase 5 — Optimization & final report/presentation |

---

## Key Papers

- **BitNet b1.58**: Ma et al. (2024) — *The Era of 1-bit LLMs* — [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- **BitNet**: Wang et al. (2023) — *Scaling 1-bit Transformers* — [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)
- **LLaMA**: Touvron et al. (2023) — architecture reference

---

## 👥 Authors

Sean Michael · Prof. Dr. Pedro Albuquerque

---

> Any academic, research, or commercial usage must cite the original repository and authors.
>
> Data source: [microsoft/BitNet](https://github.com/microsoft/BitNet) · Standard NLP benchmarks for perplexity and accuracy evaluation
