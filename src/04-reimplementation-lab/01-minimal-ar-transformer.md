# Lab 4.1 Minimal AR transformer

> **Notebook:** [`notebooks/01-minimal-ar-transformer.ipynb`](https://github.com/tuandung222/nemotron-diffusion-study/blob/main/notebooks/01-minimal-ar-transformer.ipynb)
>
> **Runtime:** CPU < 30 s · GPU < 5 s · **Params:** 0.81M

This is the baseline. A standard causal-LM (GPT-style) on a tiny character-level corpus. Every later lab modifies *exactly one thing* from this baseline, so we want it spelled out clearly.

## What you build

- A 4-layer, 128-dim, 4-head transformer with **causal self-attention** (`tril`-based mask).
- Position embeddings (learned, not RoPE, we want minimal moving parts).
- AdamW + cosine LR + grad clipping. 200 steps to converge on the toy data.

## What you measure

- Training and validation loss curves.
- Tokens-per-second of greedy AR generation.
- Generated text (qualitatively).

## What the next lab changes

- The **loss** becomes the MDLM loss (CE at masked positions instead of next-token CE).
- The **attention mask** becomes bidirectional (no `tril`).
- The architecture stays the same.

## Sanity outputs to expect

| Metric | Expected (CPU) |
|---|---|
| Final training loss | ~0.04 |
| Generated text from "The quick" | "The quick brown fox jumps over..." (recognisable corpus phrases) |
| AR tokens/s | 200–400 |

## Prerequisites

- Lectures 1.1, 1.6 (AR vs non-AR, comparison axes).
- Familiarity with PyTorch `nn.Module`, `F.scaled_dot_product_attention` or manual softmax.
