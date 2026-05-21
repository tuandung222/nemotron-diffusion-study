# Introduction

This book is a self-contained study guide for **diffusion language models (DLMs)**, **mixed AR–diffusion training**, and one specific concrete instantiation of those ideas: NVIDIA's **Nemotron-Labs-Diffusion (NLD)** model family, in particular the vision-language variant `nvidia/Nemotron-Labs-Diffusion-VLM-8B`.

## Who this is for

The intended reader is an **NLP / LLM data scientist** who is already comfortable with:

- The standard decoder-only transformer (multi-head attention, RoPE, GQA, KV cache, mixed-precision training).
- Autoregressive (AR) next-token training and inference, including techniques like batched generation, prefix caching, and speculative decoding.
- Common training tricks: cross-entropy with label smoothing, gradient checkpointing, ZeRO / FSDP, learning-rate schedulers, etc.

If you're in that bucket, you can skip transformer-101 explanations entirely. The book focuses on **what's actually different** in DLMs and how an AR-trained mental model needs to be updated.

## What this book is *not*

- It is **not** a generic "intro to diffusion models" written for vision researchers. It assumes you care about *text generation* and *LLM serving* economics. The continuous-image diffusion literature (DDPM, score matching, classifier-free guidance for images) is mentioned only where strictly necessary.
- It is **not** a paper survey. It picks a few key papers per topic and walks them in depth, then connects them to NLD specifically.
- It is **not** training code that you can drop into your cluster. The notebooks in Series 4 are pedagogical, small models, single-GPU friendly, designed to make the mechanics concrete. Production training of NLD-scale models is out of scope.

## Structure of the book

The book is organized as **four progressive series**:

### Series 1: Foundations of Diffusion Language Models

The minimum theory you need to understand why diffusion LMs exist at all and how their loss / inference differ from AR.

1. *From AR to non-AR.* Why parallel decoding matters: memory-bound vs compute-bound generation, KV cache reuse, tokens-per-forward (TPF) as the unit of efficiency.
2. *Discrete diffusion 101.* D3PM, the absorbing-state special case, forward/reverse processes, the ELBO that becomes practical.
3. *Masked Diffusion Models (MDM).* SEDD, MDLM, LLaDA, the modern simplified loss for masked-discrete diffusion and why it's just "weighted masked LM".
4. *Block diffusion.* Why full-sequence diffusion is inefficient at long contexts, and how block-wise factorization (SDAR, Block Diffusion / Arriola et al.) restores KV-cache reuse.
5. *Sampling strategies.* Confidence-based unmasking, ancestral sampling, remasking, learned samplers. The trade-off between tokens-per-forward and quality.
6. *AR vs Diffusion: a head-to-head.* When each wins, what their fundamental ceilings are, and how recent benchmarks land.

### Series 2: Mixed AR–Diffusion Language Models *(planned)*

How to train **one** model that supports both decoding modes, and what falls out of that ("self-speculation"):

1. The complementarity argument: AR priors + diffusion lookahead.
2. Joint loss, dual-stream input, structured attention masks (the M_BD / M_OBC / M_BC decomposition).
3. Self-speculation as DRAFT (diffusion) + VERIFY (AR) on shared KV cache.
4. Comparing self-speculation to MTP / EAGLE-3 / Medusa.
5. Speed-of-Light analysis: the theoretical TPF ceiling and how far current samplers are from it.

### Series 3: Nemotron-Labs-Diffusion deep dive *(planned)*

A code-level reading of NVIDIA's NLD release:

1. Model card and `config.json` walkthrough. What every flag means.
2. The `compute_block_mask` / `set_attention_mode` machinery: tri-mode implementation in PyTorch.
3. Joint AR + diffusion training pipeline: two-stage curriculum, global loss averaging, DP-rank varying masking.
4. Linear self-speculation with a LoRA-enhanced drafter (LK-hybrid distribution-matching loss).
5. Quadratic self-speculation via FlexAttention.
6. VLM extension: Pixtral encoder, asymmetric dual stream, exact-merge initialization.
7. Reading the benchmarks: how to interpret TPF, acceptance length, and real-device throughput across DGX Spark / H100 / GB200.

### Series 4: Re-implementation lab *(planned)*

A sequence of self-contained Jupyter notebooks. Each notebook starts from a tiny GPT-style baseline and adds one feature at a time until the result is a working tri-mode VLM in miniature.

The labs are *not* meant to reproduce NLD-8B's accuracy, they're meant to give you a hands-on understanding of the moving parts.

## Conventions

- All math uses inline LaTeX (`$...$` for inline, `$$...$$` for display) and is rendered with MathJax. If you're reading on GitHub directly, equations render natively; on the rendered mdBook site, MathJax handles everything.
- All code blocks are Python 3.11 / PyTorch 2.5+ unless stated otherwise. They're written for readability over performance; production kernels (FlashAttention, FlexAttention, fused triton kernels) are referenced but not re-implemented.
- Primary-source citations are inline, e.g. *(Nie et al., 2024, LLaDA)*. The full reading list lives in [Appendix B](./appendix/reading-list.md).
- "TPF" = tokens per forward pass; "NFE" = number of function evaluations (one full model forward). For AR, NFE = number of tokens produced; for diffusion, NFE < tokens produced.

## How to read

If you have time, read Series 1 in order; the lectures depend on each other. Series 2, 3, 4 each have a soft prerequisite on Series 1, but Series 3 in particular is also readable as a stand-alone if you skip to chapter 3.1 and refer back when needed.

Let's start with the obvious question: **why bother with non-AR decoding at all?**
