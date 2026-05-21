# Lab 4.2 Adding diffusion masking and loss

> **Notebook:** [`notebooks/02-diffusion-masking-and-loss.ipynb`](https://github.com/tuandung222/nemotron-diffusion-study/blob/main/notebooks/02-diffusion-masking-and-loss.ipynb)
>
> **Runtime:** CPU < 90 s · GPU < 10 s · **Params:** 0.81M

This lab converts the AR transformer from Lab 4.1 into a **masked diffusion** LM. The architecture is unchanged, only the loss and the attention mask differ.

## What changes from Lab 4.1

| Component | Lab 4.1 (AR) | Lab 4.2 (MDLM) |
|---|---|---|
| Attention mask | Causal (`tril`) | Bidirectional (full attention) |
| Input | `x[:-1]` | `noisy_x` (random positions → `<MASK>`) |
| Target | `x[1:]` | `x` at masked positions |
| Loss | CE on all positions | CE at masked positions only |

## What you build

- The **`<MASK>` token** added to the vocabulary.
- A `diffusion_loss(model, x, mask_token_id)` function:
  1. Samples a per-sequence mask ratio $\rho \sim U(0.01, 0.99)$.
  2. Picks per-position mask flags by sampling $\text{Bernoulli}(\rho)$.
  3. Replaces masked positions with `mask_token_id`.
  4. CE at masked positions only.
- Two generation functions:
  - **Iterative denoising**: NFE = `gen_length` → TPF = 1.0 (no speedup over AR yet).
  - **Block-wise generation**: NFE = `gen_length / accepted_per_block` → **TPF ≈ 2** even with random training noise.

## What you measure

- Diffusion training loss curve (will plateau at higher value than Lab 4.1's AR loss, different objective).
- TPF for iterative vs block-wise generation.

## Mathematical note

The MDLM loss (Lecture 1.3) is

$$
\mathcal{L}_{\text{MDLM}} = \mathbb{E}_{\rho \sim U(0,1)} \frac{1}{|M_\rho|} \sum_{i \in M_\rho} -\log p_\theta(x_i \mid x_{\overline{M_\rho}})
$$

where $M_\rho$ is the set of masked positions at ratio $\rho$. This is what we implement.

## What the next lab changes

- We introduce the **dual-stream input**: concatenate `[noisy_x | clean_x]` so AR loss and diffusion loss can share one forward.
- We add the **structured attention mask** $M_{BD} \cup M_{OBC} \cup M_{BC}$.

## Sanity outputs to expect

| Metric | Expected (CPU) |
|---|---|
| Final diffusion loss | ~1.5 (plateau; small model + uniform mask ratio) |
| Iterative denoising TPF | 1.00 |
| Block-wise TPF | 1.5–2.5 |

## Prerequisites

- Lab 4.1.
- Lectures 1.2, 1.3 (discrete diffusion, MDLM loss).
