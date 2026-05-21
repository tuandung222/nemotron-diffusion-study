# Lab 4.4 — Tri-mode `generate()`

> **Notebook:** [`notebooks/04-tri-mode-generate.ipynb`](https://github.com/tuandung222/nemotron-diffusion-study/blob/main/notebooks/04-tri-mode-generate.ipynb)
>
> **Runtime:** CPU < 3 min · GPU < 30 s · **Params:** 0.83M

This lab exposes the three decoding modes through one `generate(mode=...)` API, and measures them side-by-side on the same prompts.

## What you build

```python
def generate(model, prompt, mode="ar", **kwargs):
    if mode == "ar":
        return generate_ar(model, prompt, **kwargs)
    elif mode == "block_diffusion":
        return generate_block_diffusion(model, prompt, **kwargs)
    elif mode == "self_speculation":
        return generate_self_speculation(model, prompt, **kwargs)
```

All three functions use the same `model` instance. The only differences are:
1. Which **attention mask** is fed to the forward (`causal` vs `dual-stream block-diff`).
2. The **decoding loop** structure (one token vs block vs draft+verify).

## What you measure

The notebook produces a comparison table:

| mode | NFE | TPF | wall-clock (s) | tokens/s |
|---|---|---|---|---|
| ar | 32 | 1.00 | 0.13 | 386 |
| block_diffusion | 5 | 6.40 | 0.03 | 1,509 |
| self_speculation | 64 | 0.50 | 0.30 | 166 |

**Interpretation.** Block-diffusion has the best TPF and wall-clock here because the model is well-trained on the joint loss; the threshold is permissive (0.3) so the accept rate is high. Self-speculation is *slower* — this is honest pedagogy: at this tiny scale, the draft and AR-verify rarely agree, so almost every cycle falls back to the AR-only fallback (commit 1 token per 2 forwards). In production NLD-8B the drafter and verifier are aligned via LoRA (Lecture 3.4), giving TPF ≈ 6.

## Why self-speculation underperforms here

The TinyJointModel is too small (0.8M params) to learn high-confidence diffusion predictions that match the AR head. When draft `argmax != verify_argmax`, we commit zero from the draft and fall back to one AR token. With K=8 and zero agreement, we do 2 forwards per 1 committed token → TPF 0.5.

This is the exact failure mode the LoRA drafter alignment from Lecture 3.4 fixes:

$$
\mathcal{L}_{\text{hybrid}} = \lambda_L \cdot \mathcal{L}_{\text{LM}}^{\text{draft}} + \lambda_K \cdot \text{KL}(p_{\text{verify}} \| p_{\text{draft}})
$$

with $\lambda_L = 1, \lambda_K = 0.5$. Lab 4.5 adds the KV cache; you could add the LoRA alignment as an exercise on top.

## What the next lab changes

- We add a **KV cache** so the AR generation doesn't recompute the prompt every step.
- We implement **cache cloning** (for branching draft / verify) and **cache cropping** (for partial-acceptance commits).

## Prerequisites

- Labs 4.1–4.3.
- Lectures 2.3 (self-speculation), 3.2 (tri-mode).
