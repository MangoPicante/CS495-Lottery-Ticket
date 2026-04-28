# PLAN.md

## Project Overview

**Title:** Non-GPU LLM Training: Efficient Transformer Training with BitNet
**Author:** Sean Michael
**Date:** April 27, 2026

### Description

A comparison between several ternary trained models (BitNet b1.58, weights in {-1, 0, 1}) and FP16 models whose performance and cost have already been publicly documented. Rather than training FP16 baselines from scratch, this project leverages existing benchmark results from the literature (e.g. LLaMA, StableLM) and focuses experimental effort on training and evaluating multiple ternary models of varying sizes — assessing whether 1-bit quantization makes transformer pretraining viable on commodity CPUs at different scales.

### Objectives

- Train several ternary (1.58-bit) transformer models of varying sizes using BitNet b1.58
- Evaluate each ternary model on perplexity, zero-shot accuracy, training time, memory footprint, and energy consumption
- Compare results against published FP16 baselines (LLaMA, StableLM) rather than training FP16 models from scratch
- Produce a cost-accuracy trade-off analysis across model sizes, including a carbon footprint proxy
- Determine at which scale ternary models close the accuracy gap with FP16, and whether CPUs remain a viable training substrate as size increases

---

## Tasks

### Phase 1 — Repository Study
- [ ] Clone and explore the [microsoft/BitNet](https://github.com/microsoft/BitNet) repository
- [ ] Read and annotate the BitNet b1.58 paper (Ma et al., 2024)
- [ ] Document the absmean quantization function and Straight-Through Estimator
- [ ] Summarize published FP16 baseline results (LLaMA, StableLM) to use as comparison targets
- [ ] Finalize model sizes to train (e.g. 700M, 1.3B, 3B)

### Phase 2 — Baseline Training
- [ ] Configure training environment (CPU or low-resource hardware)
- [ ] Set up `MetricsTracker` to record time, memory, energy, and perplexity per run
- [ ] Train first ternary model (smallest size) on WikiText-2
- [ ] Verify loss curves are stable and perplexity is reasonable
- [ ] Confirm checkpoint saving and log output are working correctly

### Phase 3 — BitNet Implementation
- [ ] Train remaining ternary model sizes (scaled up from Phase 2)
- [ ] Run lm-evaluation-harness on each model (ARC, HellaSwag, WinoGrande, etc.)
- [ ] Record perplexity on WikiText-2 validation set for each model
- [ ] Log training time, peak memory, and energy consumption per run
- [ ] Sanity-check results against Table 1 and Table 2 from the BitNet b1.58 paper

### Phase 4 — Cost Comparison
- [ ] Compile ternary model results alongside published FP16 baselines into a single table
- [ ] Run `compare_runs.py` to generate loss curves, bar charts, and memory plots
- [ ] Compute cost-accuracy trade-off (dollar cost proxy: time × hardware rate)
- [ ] Estimate carbon footprint using CodeCarbon output
- [ ] Produce final benchmark dashboard (plots + CSV)

### Phase 5 — Optimization & Writeup
- [ ] Experiment with at least one optimization (e.g. hybrid precision, quantization schedule)
- [ ] Document scaling observations — at what size do ternary models match FP16 accuracy?
- [ ] Write capstone research report
- [ ] Prepare final presentation slides
- [ ] Clean up repository for reproducibility (configs, instructions, results)