# 2.5 Speed-of-Light analysis: the theoretical ceiling of parallel decoding

> **Goal of this lecture.** Compute the *upper bound* on tokens-per-second for any decoder-only LM as a function of (model size, batch size, GPU memory bandwidth, GPU FLOPs). Compare against NLD's measured numbers to compute the **SOL utilisation** and identify what's left on the table. By the end you should be able to: (i) state the SOL formula in the memory-bound and compute-bound regimes, (ii) compute SOL for an arbitrary (model, GPU, batch) combination, (iii) read NLD's tech-report tables and immediately identify which numbers are SOL-limited vs algorithm-limited.

Background assumed: Lecture 1.1 (memory-bound vs compute-bound), Lecture 2.3 (self-speculation cycle accounting). Comfort with roofline-style arithmetic.

---

## 1. The roofline model in one paragraph

Every dense matmul on a GPU is either *memory-bound* (the GPU can finish the FLOPs faster than HBM can stream the operand bytes, so the actual throughput equals HBM bandwidth × bytes-per-FLOP) or *compute-bound* (the operands are small enough to stay in SRAM/L2, so throughput equals peak FLOPs). The transition between regimes happens at a critical *arithmetic intensity* $I^*$ given by:

$$
I^* \;=\; \frac{\text{peak FLOPs/s}}{\text{HBM bandwidth (bytes/s)}}.
$$

On a B200 with 4500 TFLOPs/s (BF16) and 8 TB/s HBM, $I^* \approx 560$ FLOPs/byte. On an A100 with 312 TFLOPs/s and 2 TB/s, $I^* \approx 156$. On an H100 SXM with 989 TFLOPs/s and 3.35 TB/s, $I^* \approx 295$.

If your workload has $I < I^*$, you're memory-bound and adding compute units won't help. If $I > I^*$, you're compute-bound and adding bandwidth won't help.

This is the entire physics of LM inference economics.

---

## 2. AR baseline: where does it sit on the roofline?

Consider AR decoding at batch size $b$, model size $W$ parameters, hidden dim $d$.

### 2.1 FLOPs per token

One forward of the model takes $\approx 2W$ FLOPs (the rough roofline approximation for transformer inference, ignoring attention's $O(L^2)$ tail for short sequences). At batch $b$, that's $2 W b$ FLOPs per *forward*, which gives $b$ tokens per forward (one per sequence), so:

$$
\text{FLOPs per token (AR)} \;=\; \frac{2 W b}{b} \;=\; 2W.
$$

Independent of batch size. Curious result, but it's because we're amortising the same weight load across $b$ sequences.

### 2.2 Bytes streamed per token

Per forward, we stream:

- The model weights: $W$ params × 2 bytes (BF16) = $2W$ bytes.
- The KV cache for length $L$: this grows linearly with $L$, but let's ignore it for the moment (typical short-context inference has KV cache < weights).

So bytes per forward $\approx 2W$, giving:

$$
\text{Bytes per token (AR)} \;=\; \frac{2W}{b}.
$$

### 2.3 Arithmetic intensity of AR decoding

$$
I_{\text{AR}} \;=\; \frac{\text{FLOPs per token}}{\text{Bytes per token}} \;=\; \frac{2W}{2W / b} \;=\; b.
$$

Astonishingly clean: **AR decoding has arithmetic intensity equal to the batch size**. So:

- At $b = 1$: $I = 1$, deeply memory-bound. Compute is wasted; you're paying for bandwidth.
- At $b = I^*$: critical regime. On B200, that's $b \approx 560$.
- At $b \gg I^*$: compute-bound. Throughput scales with FLOPs.

For practical 8B-class models on a single B200, $I^* \approx 560$, so you need batch ~600 to leave the memory-bound regime. Most production deployments run at $b = 4$–$32$, so most production inference is **deeply memory-bound**. This is the regime where parallel decoding helps.

### 2.4 Speed-of-Light for AR

In the memory-bound regime ($b \ll I^*$), AR throughput is:

$$
\text{Tokens/s (AR, SOL)} \;=\; \frac{\text{HBM bandwidth}}{\text{bytes per token}} \;=\; \frac{B_{\text{HBM}}}{2W / b} \;=\; \frac{b \cdot B_{\text{HBM}}}{2W}.
$$

Plug in numbers for an 8B BF16 model ($W = 8 \times 10^9$, $2W = 16$ GB) on different GPUs:

| GPU | HBM BW | $b = 1$ | $b = 16$ | $b = 256$ |
|---|---|---|---|---|
| A100 80GB | 2.0 TB/s | 125 tok/s | 2,000 tok/s | 32,000 tok/s |
| H100 SXM | 3.35 TB/s | 209 tok/s | 3,344 tok/s | 53,500 tok/s |
| B200 | 8.0 TB/s | 500 tok/s | 8,000 tok/s | 128,000 tok/s |
| DGX Spark | 0.4 TB/s | 25 tok/s | 400 tok/s | 6,400 tok/s |

These are the **AR SOL numbers**. Any AR decoder on these GPUs that runs faster than these numbers is doing something physically impossible (or using a smaller model than 8B).

In the compute-bound regime ($b \gg I^*$):

$$
\text{Tokens/s (AR, SOL, compute-bound)} \;=\; \frac{\text{peak FLOPs}}{2W}.
$$

E.g., B200: $4500 \times 10^{12} / (2 \times 8 \times 10^9) = 281{,}250$ tok/s. But this requires $b \gtrsim 560$, which on B200 is feasible only at very large memory budgets.

---

## 3. Parallel-decoding SOL

Parallel decoding (self-speculation, EAGLE, MTP, …) all share one structural property: **each cycle costs roughly $c$ forwards, and commits roughly $\mathbb{E}[m]$ tokens**.

The SOL for parallel decoding in the memory-bound regime is:

$$
\text{Tokens/s (SOL, parallel)} \;=\; \frac{b \cdot B_{\text{HBM}}}{c \cdot 2W} \cdot \mathbb{E}[m] \;=\; \frac{\text{TPF}}{1} \cdot \text{Tokens/s (AR, SOL)},
$$

where TPF = $\mathbb{E}[m] / c$. So **parallel decoding's SOL is exactly AR's SOL multiplied by TPF**. There is no "extra" speedup beyond what TPF gives you.

This is a critical observation: TPF measures how close you are to AR SOL × TPF, not how close you are to some other physical ceiling. If TPF is 6, you cannot beat AR SOL × 6 unless you're in a different regime.

### 3.1 Two important corollaries

1. **TPF saturates at a finite value.** TPF is bounded by the acceptance probability and the block size (Lecture 2.3 §3.3). No algorithmic improvement can give you arbitrarily high TPF.

2. **Beyond TPF, the only way to go faster is to lower the per-token byte cost.** That means quantization (lower bits per weight), or smaller models, or sparser models. NLD's tech report has a brief discussion of FP8 inference for self-speculation in §7; the headline is that FP8 doubles tokens/s in the memory-bound regime by halving the bytes per weight.

### 3.2 The full SOL formula

Putting it all together, the SOL for any decoding algorithm on any GPU at batch $b$ is:

$$
\boxed{
\text{Tokens/s (SOL)} \;=\;
\begin{cases}
\frac{b \cdot B_{\text{HBM}}}{B_W \cdot W} \cdot \text{TPF} & \text{if } b < I^* / \text{TPF} \quad \text{(memory-bound)} \\
\frac{\text{peak FLOPs}}{B_W \cdot W} \cdot \text{TPF} & \text{if } b > I^* / \text{TPF} \quad \text{(compute-bound)}
\end{cases}
}
$$

where $B_W$ is bytes-per-weight (2 for BF16, 1 for FP8, 0.5 for INT4).

The crossover batch size is $I^* / \text{TPF}$ for parallel decoding, lower than AR's crossover at $I^*$. So **parallel decoding pushes you into the compute-bound regime at smaller batch sizes**.

For NLD self-speculation on B200 with TPF $\approx 6$, the crossover is at $b \approx 93$. Past that point, the TPF advantage starts disappearing because you're no longer memory-bound. This matches NLD's empirical observation that self-speculation's speedup drops from 5–6× at b=1 to ~1.5× at b=256.

---

## 4. SOL utilisation: how close to ceiling does NLD get?

Computed numbers from NLD's tech report, normalised against the SOL formula:

### 4.1 At batch 1

| Hardware | NLD AR (tok/s) | NLD self-spec linear (tok/s) | AR SOL (tok/s) | Self-spec SOL (tok/s) | AR util | Self-spec util |
|---|---|---|---|---|---|---|
| B200 | 280 | 1680 | 500 | 3000 | 56% | 56% |
| H100 | 120 | 740 | 209 | 1255 | 57% | 59% |
| A100 | 70 | 410 | 125 | 750 | 56% | 55% |
| DGX Spark | 15 | 80 | 25 | 150 | 60% | 53% |

NLD reaches **roughly 55% SOL utilisation across hardware**. This is a consistent number, suggesting the bottleneck is somewhere other than the raw HBM bandwidth.

### 4.2 What costs the missing 45%?

The standard accounting for non-SOL overhead in inference:

1. **KV-cache attention.** At long context, the KV cache becomes a non-trivial fraction of the bytes streamed per token. Each new query reads the entire history's KV. This is $O(L \cdot b \cdot d)$ extra bytes per token. For $L = 2048$, $b = 1$, $d = 4096$, that's ~33 MB per token, small relative to model weight load (16 GB), so usually ≤ 5% overhead. NLD's evals run at moderate context lengths so this is bounded.

2. **Activation memory.** The forward pass writes intermediate activations to HBM for FlashAttention-style attention computation. ~5–10% of streamed bytes.

3. **Layer norm + small ops.** RMSNorm, Rotary embeddings, the embedding lookup, none individually large but together ~3% of total bytes.

4. **Sampling / softmax / postprocessing.** Top-k selection, argmax, threshold checks, ~2% of forward time.

5. **Kernel launch overhead.** ~3% on B200 (kernel launches are not free even at zero-cost).

6. **Memory-allocator overhead.** Especially during KV-cache appends; ~5%.

7. **Inter-mode switch cost.** Self-speculation alternates between AR mode and diffusion mode within a cycle. The structured attention mask is rebuilt and `set_attention_mode` is called twice per cycle. ~5–10% overhead.

Sum: ~30–45%. NLD's measured 55% utilisation is consistent with this accounting.

> **What this implies.** Getting from 55% to 70% utilisation requires meaningful engineering work: fused mode-switch kernels, persistent KV-cache layouts, careful FlashAttention tuning. Past 70%, you start needing speculative scheduling and warp-level optimisation. NLD's tech report acknowledges these as future work.

### 4.3 The TPF vs utilisation trade-off

A surprising observation: as TPF goes up (linear → quadratic self-speculation), SOL utilisation goes *down* in the NLD numbers:

| Mode | TPF | Utilisation |
|---|---|---|
| AR | 1.0 | 56% |
| Block-diffusion | 2.8 | 52% |
| Self-spec linear | 6.0 | 56% |
| Self-spec quadratic | 8.0 | 49% |

This reflects two things:

1. **Higher TPF means tighter per-cycle scheduling**, so per-token overhead is amortised over more tokens but per-cycle overhead is larger.
2. **Quadratic self-speculation has more mode switches** and more KV-cache slicing operations, raising the constant-factor overhead per cycle.

In other words: improving TPF doesn't free your engineering team from optimising the constant factors. If anything, it raises the bar.

---

## 5. SOL across hardware: what's the *right* GPU for NLD?

Each GPU has a different sweet-spot.

### 5.1 DGX Spark (consumer / edge)

Bandwidth: 0.4 TB/s (DDR5, no HBM). Peak FLOPs: ~25 TFLOPs (BF16).

$I^* = 25 / 0.4 = 62.5$, so memory-bound up to $b \approx 60 / \text{TPF}$. At TPF = 6, crossover is $b \approx 10$.

The DGX Spark use case is **on-device single-user**: $b = 1$, latency-critical. NLD self-speculation reports 80 tok/s here, 3.3× the AR baseline. SOL is 150 tok/s, so utilisation is ≈ 53%.

This is the regime where NLD's batch-1 advantage matters most. EAGLE-3 reports 50 tok/s on the same hardware (drafter HBM tax bites hard at this bandwidth).

### 5.2 H100 single (cloud, small-batch)

Bandwidth: 3.35 TB/s. Peak FLOPs: 989 TFLOPs. $I^* \approx 295$.

Crossover at TPF = 6 is $b \approx 49$. The H100 batch-1 use case (LLM API at low concurrency) hits 740 tok/s with NLD self-spec, against 209 tok/s SOL for AR, 3.5× speedup over AR baseline.

### 5.3 GB200 NVL72 (cloud, high-batch)

Bandwidth: 8 TB/s. Peak FLOPs: 4500 TFLOPs. $I^* \approx 560$.

Crossover at TPF = 6 is $b \approx 93$. At $b = 256$ on NVL72, the per-cycle compute is already past the crossover, so the TPF speedup is ~1.4× over AR baseline rather than ~6×.

The GB200 NVL72 use case is **cloud high-concurrency**: $b = 64$–$256$. NLD's speedup here is modest, in part because the AR baseline is already running at near-SOL.

### 5.4 The right deployment per GPU

| GPU class | Best workload | Best mode | Expected speedup |
|---|---|---|---|
| DGX Spark | Single-user on-device | Self-spec linear | 3.3× over AR |
| H100 single | Low-concurrency API | Self-spec linear | 3.5× |
| GB200 NVL72 | High-batch cloud | AR (b ≥ 64) | 1.4× |
| B200 (general) | Mixed | Self-spec at low b, AR at high b |, |

Pattern: NLD is most valuable on **low-bandwidth, low-batch** hardware. The higher the bandwidth or batch size, the smaller the NLD advantage.

---

## 6. The implications for model design

A higher-level question: given that parallel decoding's SOL is fundamentally bounded, is there a regime where smarter algorithms could beat the SOL?

Yes, by changing one of the parameters:

1. **Lower $W$ (smaller model).** A 3B model has half the bytes-per-forward of an 8B, so it goes 2× faster at SOL. This is why distillation and small-model approaches remain competitive at inference time, even with much worse accuracy.

2. **Lower $B_W$ (quantization).** FP8 halves bytes-per-weight; INT4 quarters them. NLD's tech report §7 reports 1.9× speedup from FP8 inference (close to the theoretical 2×). The marginal accuracy loss is < 0.5 point on most evals.

3. **Sparse / MoE.** Only a fraction of weights stream per forward. With 2-of-8 expert routing, only 25% of weights stream, giving 4× theoretical speedup. Real-world MoE inference reports 2–3× over dense AR for the same active-parameter count.

4. **Lower context length.** KV cache scales with context. Truncating context lowers the bytes-per-token after the model weight load.

NLD is essentially using approach (1) implicitly via "more tokens per HBM load" (TPF) rather than shrinking the model. The two are multiplicative: a 4B NLD with TPF 6 would have SOL of $2 \cdot 4 \cdot 6 = 48 \times$ the AR baseline of a 4B model.

The lesson: **SOL is not a hard ceiling; it's a ceiling per-GPU-per-model-per-batch-size**. Changing those parameters changes the ceiling.

---

## 7. What this means for the rest of the book

We now have the analytical machinery to read NLD's published benchmark tables critically. Specifically, for any benchmark row we see in Series 3 (Lecture 3.7), we can:

1. Compute the SOL for that (model, GPU, batch) tuple.
2. Compute the utilisation (measured / SOL).
3. Identify whether NLD is bottlenecked by algorithm (low TPF) or by engineering (low utilisation).

We'll do this exercise systematically in Lecture 3.7. The answer will guide whether the "right" optimisation to invest in is bigger drafters, smarter quadratic self-speculation, or low-level kernel work.

---

## 8. Exercises

1. Compute the SOL utilisation for an 8B BF16 model on an A100 80GB at batch 16, given measured 1450 tok/s with NLD self-speculation (TPF = 5.5). State the bottleneck regime.

2. Predict the FP8 SOL for the same setup as (1). How does utilisation change if you assume the same algorithmic overhead?

3. The crossover batch size between memory-bound and compute-bound is $I^* / \text{TPF}$. Derive this for a model with explicit FLOPs/weight ratio $\rho$ (not all 2W). What value of $\rho$ would make the crossover happen at $b = 1$?

4. KV cache becomes a meaningful fraction of bytes-streamed at long context. Derive the critical context length at which KV bytes equal weight bytes for an 8B model with $d = 4096$, GQA group size 8.

5. Suppose someone claims their LM does 100 tok/s on a CPU with 80 GB/s memory bandwidth at batch 1, for an 8B BF16 model. Is this physically possible? Show the arithmetic.

Solutions to (1), (4), (5) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 9. Further reading

- **Williams, Waterman, Patterson** (2009). *Roofline: An Insightful Visual Performance Model for Floating-Point Programs*. The canonical roofline model paper.
- **Pope, Wagh, et al.** (2023). *Efficiently Scaling Transformer Inference*. Google's analysis of transformer inference economics; §3 has the AR arithmetic-intensity formula we use here.
- **Patel, Park, et al.** (2024). *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*. The prefill-vs-decode roofline analysis. Good complement to this lecture.
- **NVIDIA Hopper Architecture Whitepaper** (2024). For peak bandwidth and FLOPs numbers on H100/H200/B200/GB200 NVL72.
- **Nemotron-Labs-Diffusion Tech Report** (Fu et al., 2026). §7 (deployment economics) and Tables 7–9 (per-GPU benchmarks).

End of Series 2. Series 3 takes the abstract machinery from Series 1+2 and reads the actual NLD source code line by line, starting with the model card and `config.json`.
