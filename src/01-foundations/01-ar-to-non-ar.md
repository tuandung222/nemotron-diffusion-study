# 1.1 - From AR to non-AR: why parallel decoding matters

> **Goal of this lecture.** Frame the problem that diffusion language models are trying to solve. By the end you should be able to say, in one paragraph, **why** anyone would give up the comfort of left-to-right next-token prediction in exchange for the headaches of non-AR training and inference - and to do so in terms of *concrete deployment economics*, not vibes.

Background assumed: you can describe what a KV cache is, what GQA does, why prefill is compute-bound and decode is memory-bound, and roughly how speculative decoding works. If any of that is fuzzy, the suggested re-reads are at the bottom of this page.

---

## 1. The arithmetic of AR decoding

Consider a transformer with $L$ layers, hidden size $d$, and $n_h$ attention heads with $n_{kv}$ key-value heads (so GQA group size $g = n_h / n_{kv}$). When you decode token $t+1$ given a prefix of length $t$:

- **Compute:** you run **one** forward pass that processes a single new query token at every layer.
- **Memory read:** you stream in the **entire model weights** (call this $W$ bytes) **and** the **prefix KV cache** ($K_{\text{cache}}$ bytes). On a modern decoder LLM, $W \gg K_{\text{cache}}$ for short contexts but they're comparable once the context grows past a few thousand tokens.
- **Memory write:** you append one KV per layer per head.

On NVIDIA hardware, this puts you firmly in the **memory-bound regime** for decode. The arithmetic intensity is roughly

$$
\text{AI}_{\text{AR-decode}} \;\approx\; \frac{\text{FLOPs per token}}{\text{bytes streamed per token}} \;\approx\; \mathcal{O}(1)\text{ to }\mathcal{O}(10),
$$

while a modern GPU's roofline ratio (peak TFLOPs / peak HBM TB/s) is in the **hundreds**. In practice, decode utilizes a few percent of the GPU's FLOP capacity.

A vivid restatement: **you paid to ship every parameter from HBM to SRAM, and got one (1) token out of it.** That's the core inefficiency.

### 1.1 Concurrency partially fixes this

If you batch $B$ independent requests together, the same weight load amortizes across $B$ tokens. As $B$ grows you eventually hit the compute-bound regime where utilization improves. This is why **cloud serving** generally does very well with vanilla AR - you have hundreds or thousands of concurrent users, and the batch absorbs the memory cost.

But concurrency does not help for the **single-user / low-concurrency** regime, which includes:

- On-device inference (phone, laptop, DGX Spark, robot).
- Long chain-of-thought generation where a single user is the entire batch.
- Streaming use cases where the user reads as the model writes and you can't safely re-batch them with somebody else.
- Latency-critical agents where time-to-first-token + time-per-token both matter and you can't wait for a batch to fill.

In all of those, the memory wall is the limit, and the only escape is to **get more than one token per weight load**. There are two known ways to do that:

1. **Speculative decoding.** Use a small "drafter" model to emit $k$ candidate tokens in parallel, then verify them with the target model in a single forward pass. If $m \le k$ tokens are accepted, you've produced $m+1$ tokens per target forward instead of one. Speculative decoding is *still* AR under the hood - the target model is run autoregressively - but each forward batches $k+1$ positions through the same weight load. (Leviathan & Kalman 2022; Chen et al. 2023; EAGLE-2/3.)

2. **Non-AR decoding.** Restructure the model so that **multiple positions are predicted in a single forward pass without needing a separate drafter**. This is the diffusion-LM route.

The rest of this book is about path (2), and eventually about how to combine (1) and (2) inside the same model.

---

## 2. What "tokens per forward" really measures

Let's nail down the unit of efficiency we'll use throughout the book.

**Definition.** *Tokens per forward (TPF)* is the average number of *committed output tokens* produced per call to the model. For AR decode, TPF = 1 by construction (one new query token per call, ignoring chunked prefill). For diffusion or self-speculation, TPF can be > 1.

A related quantity is **Number of Function Evaluations (NFE)** = total forward passes used to produce a sequence of length $N$:

$$
\text{NFE} \;=\; \frac{N}{\text{TPF}}.
$$

A diffusion model that produces $N = 256$ output tokens in $\text{NFE} = 64$ forwards has $\text{TPF} = 4$. The user perceives a $4\times$ wall-clock speedup *if and only if* each forward takes about the same wall-clock time as an AR decode step.

> ⚠️ **Subtlety.** TPF $> 1$ does not automatically mean speed. A diffusion forward may have a *longer* sequence length than an AR forward, which inflates per-forward latency. The honest metric is **wall-clock time per committed token**, and we'll compute that explicitly in lecture 1.6.

The NLD paper, the speed-of-light analysis, and almost every modern diffusion-LM benchmark report results in TPF; you should reflexively translate TPF into wall-clock by multiplying by per-forward latency.

---

## 3. The non-AR design space

Why didn't we have non-AR LMs all along? Mostly because the obvious idea - predict all tokens in parallel from a single context - destroys quality. The standard taxonomy of non-AR approaches:

### 3.1 Parallel decoding from a single forward (no iteration)

- *Non-Autoregressive Translation (NAT).* Gu et al. 2018. Predict every output token in parallel from the source encoding. Quality is significantly worse than AR because the model can't condition on its own previous predictions, and modeling distributions like $p(\mathbf{y} \mid \mathbf{x}) = \prod_t p(y_t \mid \mathbf{x})$ assumes token independence given $\mathbf{x}$, which is false for natural language.
- *Mixture of Experts with conditional independence assumptions* - same conceptual issue.

These methods get you TPF $= N$ in *one* forward, but at unacceptable quality.

### 3.2 Iterative refinement

- *Levenshtein Transformer*, *Mask-Predict* (Ghazvininejad et al., 2019), *CMLM*. Predict all tokens, mask the low-confidence ones, predict again, repeat. After $k$ iterations you get TPF $= N/k$. These work because conditioning improves with each round.

### 3.3 Multi-token prediction (MTP) heads

- *Medusa, EAGLE-1/2/3, DeepSeek MTP.* Attach $k$ small heads to the base LM that each predict tokens $t+1, t+2, \dots, t+k$ in parallel; verify against the base LM. This is essentially **speculative decoding without a separate drafter** - the drafter and verifier share most parameters.
- Strengths: trains cheaply on top of a frozen base. Weaknesses: the extra heads add parameters, the drafts are limited in lookahead by the head depth, and the heads have to be trained per-base-model.

### 3.4 Diffusion language models

- *D3PM* (Austin et al., 2021): discrete-state diffusion via a parameterized forward transition.
- *SEDD* (Lou et al., 2024): score entropy discrete diffusion.
- *MDLM* (Sahoo et al., 2024): "Simple and effective masked diffusion language models" - collapses everything to a weighted masked-LM loss.
- *LLaDA* (Nie et al., 2024): scales masked diffusion to 8B params.
- *Block diffusion* (Arriola et al., 2024): factorizes the sequence into blocks so KV cache is reusable.
- *SDAR* (2025), *Dream-7B*, *Nemotron-Labs-Diffusion* (Fu et al., 2026): production-grade diffusion (or AR+diffusion) LMs.

Diffusion LMs sit in between (3.1) and (3.2) conceptually: they iteratively refine, but the iterations are derived from a principled discrete-diffusion forward process, which gives them better convergence properties and a clean training objective.

This book is squarely about (3.4), and ultimately about combining (3.3) and (3.4) inside one model.

---

## 4. Where does the speed actually come from?

A common confusion: people say "diffusion LMs are faster than AR" without specifying *what* is faster.

Diffusion LMs (and mixed AR-diffusion LMs) are **not** faster per FLOP. If anything, they spend slightly more FLOPs because each forward processes a longer sequence (a whole block at once, or both noisy and clean views).

What they *do* is convert **memory-bound time** into **compute time**. Concretely:

- AR decode at batch 1: loads $W$ bytes, gets 1 token. Memory-bound.
- Diffusion block decode at batch 1: loads $W$ bytes, processes a block of $B$ noisy positions, commits some subset $k \le B$ of them. The memory cost is roughly the same as one AR step (the noisy block is small compared to weights), but the FLOP cost grows linearly in $B$ and we get $k$ tokens out.

So the *first* committed token is a touch slower than AR (because $B > 1$ token's worth of attention computation), but the *average* committed token is much faster. The model has been **moved from the memory-bound regime toward the compute-bound regime** - and that's exactly the regime where modern GPUs are good.

The NLD paper opens with this exact framing: *"Generation moved from a memory-bound regime toward a compute-bound regime. Model weights are loaded once and reused to compute multiple tokens during generation."* You should keep this in mind every time you see a diffusion-LM speedup claim.

---

## 5. Other reasons to want non-AR generation

Speed is the headline reason, but not the only one:

1. **Lookahead planning.** AR is myopic: the model commits to token $t$ before seeing what would have been a good token $t+5$. Diffusion training forces the model to reason about future tokens jointly during denoising, which the NLD paper claims (and ablates) yields *better* accuracy on tasks requiring forward planning, e.g. math reasoning. The mechanism is similar to "non-myopic decoding" arguments from beam search literature.

2. **Editable / infilling generation.** AR is left-to-right; if you want to edit a middle span you have to either resample everything to the right, or train a special infilling objective (FIM). Diffusion naturally supports arbitrary mask patterns, so infilling, inpainting, span-edit are first-class operations.

3. **Diverse sampling.** Iterative diffusion samplers can be tuned for diversity vs determinism via the unmask schedule, in a way that's more controllable than temperature on AR.

4. **Bidirectional context within a block.** Within a diffusion block, every position attends bidirectionally. For tasks where the natural "answer" is a *block* rather than a sequence (e.g. structured output, code snippet, JSON field set), bidirectional intra-block attention is genuinely useful.

Realistic caveats:

1. **Training cost.** Diffusion losses converge slower per token than AR loss, and you usually train *both* losses in a mixed regime (Series 2). NLD-8B was warm-started from Ministral3-8B base and continual-pretrained, not trained from scratch.
2. **Maturity of the inference stack.** vLLM/SGLang now support diffusion LMs, but the ecosystem is much younger than the AR stack. Many low-level tricks (chunked prefill, paged-attention prefetch, etc.) need new code paths.
3. **Long-tail quality.** Pure diffusion mode tends to underperform AR on tasks where left-to-right linguistic priors dominate, e.g. instruction following with strict format. Mixed AR-diffusion training (Series 2) closes most of this gap.

---

## 6. A worked example: TPF arithmetic at batch 1

Let's make the speed argument concrete. Take Nemotron-Labs-Diffusion-8B as a stand-in:

- Model: ~8B params, BF16 → $W \approx 16$ GB streamed per forward.
- AR decode at batch 1 on a single GPU with 1.5 TB/s HBM (B200-class): one forward takes $\sim 16 / 1500 \approx 10.7$ ms of memory streaming alone. Compute time is negligible. So we get $\sim 93$ tok/s from the memory bottleneck.
- Diffusion block decode with block_length = 32 and TPF $\approx 2.6$ (NLD's measured number): one forward takes ~12 ms (slightly more, because we're attending over 32 noisy positions + clean prefix). We commit $\sim 2.6$ tokens per forward, so effective rate $\approx 2.6 / 0.012 \approx 217$ tok/s.
- Linear self-speculation w/ LoRA at TPF $\approx 6.0$: each "cycle" is a draft forward + a verify forward $\approx 24$ ms total. We commit $\sim 6$ tokens per cycle → $\sim 250$ tok/s, with bigger improvements at larger blocks.

These numbers roughly match the paper's measured 2.7× / 3.3× speedups on DGX Spark and GB200. The point is not the exact constants; it's that **the speedup scales with how memory-bound the AR baseline is**. If your AR baseline is already at batch 256 saturating compute, diffusion gives you very little. If your AR baseline is at batch 1 saturating HBM bandwidth, diffusion gives you a lot.

This is the **per-user vs system throughput** trade-off that NLD Fig. 1(c) plots. We'll return to that figure in Series 3.

---

## 7. Setting up the rest of Series 1

We've established:

- Decode at low concurrency is memory-bound; the goal is more tokens per weight load.
- There are two clean strategies: speculative decoding (multi-token prediction, externally drafted) and non-AR / diffusion decoding.
- Diffusion LMs sit in the non-AR camp; their speedup converts memory time into compute time.
- A unified AR + diffusion model can combine both strategies (this is NLD's "self-speculation").

The next four lectures fill in the diffusion side:

- **1.2** - what does it actually mean to "diffuse" a discrete sequence? We need the math before we can talk about training objectives.
- **1.3** - modern simplified masked-diffusion objectives (the ones NLD actually uses).
- **1.4** - why block-wise diffusion, and how it makes KV cache reuse work again.
- **1.5** - sampling algorithms (the part that determines real TPF).
- **1.6** - head-to-head with AR LMs on the dimensions that matter for production.

Then Series 2 will explain how to glue AR and diffusion together.

---

## Suggested re-reads (if any of section 1 was fuzzy)

- *FlashAttention-2* (Dao, 2023) §2 - clearest published explanation of arithmetic intensity for attention.
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., 2023) §3 - KV-cache memory accounting.
- *Speculative Decoding* (Leviathan et al., 2022; Chen et al., 2023) - the canonical speculative-decoding papers; necessary background for self-speculation in Series 2.
- *EAGLE-3* (Li et al., 2024) - modern MTP-style speculative decoding, which NLD explicitly compares against.

## Exercises

1. Compute the AR decode rate, in tokens per second, for an 8B BF16 model at batch 1, batch 16, and batch 256, on:
   - A100 80GB (HBM bandwidth $\approx 2$ TB/s).
   - H100 SXM (HBM3 bandwidth $\approx 3.35$ TB/s).
   - B200 (HBM3e bandwidth $\approx 8$ TB/s).
   Assume the KV cache for a 4K-token prefix is roughly 1 GB. At what batch size does the regime switch from memory-bound to compute-bound on each device?

2. If a diffusion LM commits TPF $= k$ tokens per forward but each forward costs $1 + \alpha \cdot k$ times as much wall-clock as an AR decode (because of the longer sequence per forward), at what $\alpha$ does the diffusion LM stop being faster than AR? Express your answer as a function of $k$.

3. Why doesn't naive parallel decoding (NAT-style) work for autoregressive language modeling, in terms of the chain-rule factorization of $p(\mathbf{x})$? Be precise.

Answers to (1)–(3) are sketched in [Appendix B](../appendix/reading-list.md#exercise-solutions).
