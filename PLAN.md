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