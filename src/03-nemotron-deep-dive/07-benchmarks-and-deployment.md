# 3.7 Benchmarks and deployment economics

> **Goal of this lecture.** Read NLD-8B's published benchmark numbers critically, decompose where speedup comes from (TPF vs. hardware vs. quality), and end with a deployment-decision flowchart. By the end you should be able to: (i) recognise which numbers come from the NLD tech report or HF model card and which are derivations; (ii) decide which inference mode + hardware combination is right for a given workload; (iii) reason about what the next 2x speedup opportunity looks like.

Background assumed: Lecture 2.5 (SOL analysis), Lectures 3.1-3.6 (NLD code). Comfort with reading benchmark tables and back-of-envelope calculation.

References:
- Tech report: <https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_Diffusion_Tech_Report_v1.pdf> (sec 4 SOL, sec 5 family, sec 6 benchmarks)
- HF model card NLD-8B: <https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B>

Throughout this lecture, every numeric is annotated with the source it comes from (tech report figure/table or model card). Numbers introduced as "derived" or "illustrative" are computed in this lecture and should not be cited as paper results.

---

## 1. The quality side: does NLD trade quality for speed?

Tech report Table 5 reports the instruct-model benchmark suite. The directly-comparable rows for NLD-8B vs. the relevant baselines are:

| Benchmark | Qwen3-8B (AR) | Ministral3-8B-Instruct-2512 (AR) | LLaDA-8B-Instruct (Diff) | SDAR-8B-Chat (Diff) | NLD-8B AR | NLD-8B Diff | NLD-8B Linear SS | NLD-8B Quad SS |
|---|---|---|---|---|---|---|---|---|
| GPQA | 49.24 | 42.87 | 33.30 | 30.80 | 44.44 | 43.94 | 40.40 | 44.30 |
| IFEval | 87.38 | 64.31 | 59.90 | 60.07 | 68.65 | 68.32 | 69.13 | 71.00 |
| MMLU | 76.66 | 73.90 | 65.50 | 78.83 | 79.85 | 78.71 | 79.01 | 79.95 |
| HumanEval | 81.71 | 71.04 | 49.40 | 79.27 | 80.49 | 78.66 | 81.71 | 79.27 |
| MBPP | 81.88 | 78.97 | 41.00 | 67.32 | 85.19 | 83.86 | 84.92 | 85.19 |
| LCB-CPP | 21.09 | 20.76 | 4.19 | 11.89 | 28.85 | 26.16 | 24.89 | 27.70 |
| Math500 | 84.80 | 83.60 | 39.20 | 72.40 | 88.00 | 85.80 | 87.60 | 88.80 |
| GSM8K | 92.42 | 92.42 | 79.91 | 88.48 | 94.01 | 93.03 | 93.78 | 94.16 |
| AIME24 | 30.21 | 27.71 | 0.00 | 13.33 | 33.33 | 46.67 | 36.67 | 33.33 |
| AIME25 | 22.08 | 24.58 | 0.00 | 3.33 | 33.33 | 26.67 | 30.00 | 36.67 |
| **Avg accuracy** | **62.75** | **58.02** | **37.24** | **50.57** | **63.61** | **63.18** | **62.81** | **64.04** |
| TPF | 1.00 | 1.00 | 1.00 | 1.75 | 1.00 | 2.57 | 5.99 | 6.38 |

The key takeaways:

1. **Quality is preserved across modes.** NLD-8B's average accuracy in AR mode (63.61), diffusion mode (63.18), linear SS (62.81), and quadratic SS (64.04) are all within ~1 point of each other. The diffusion training does not hurt the AR mode and the self-speculation verification does not degrade quality vs. AR (it can't, by construction: AR verification produces the AR distribution exactly).
2. **NLD-8B is competitive with the top open AR baseline.** Avg accuracy 63.61 (AR) and 64.04 (quad SS) compares favourably with Qwen3-8B (62.75) and is well above Ministral3-8B-Instruct (58.02). The pure-diffusion baselines (LLaDA-8B, SDAR-8B Chat) are significantly behind.
3. **TPF jumps from 1.0 (AR) to ~6 (self-spec) at this accuracy point.** The headline ratio: 5.99 / 1.00 = 6x linear SS speedup, 6.38 / 1.00 = 6.4x quadratic SS speedup, both vs NLD's own AR mode. The model card phrases this as "5.9x tokens per forward over Qwen3-8B (no MTP) with the same accuracy".

The 0.6 to 5.6 point gap over Qwen3 / Ministral3 baselines reflects (a) NLD's joint training improving reasoning-heavy benchmarks like AIME24/25, and (b) a slight regression on IFEval (where Qwen3 is dominant). Treat the headline as "no quality cost for the speedup".

---

## 2. The speed side: throughput across hardware

The hardware numbers in this section come directly from the model card's "Real-device speed-up across platforms" paragraph and tech report sec 6 / Figure 1. We do not interpolate or fabricate values that are not in the published material.

### 2.1 GB200 (high-end cloud, concurrency 1)

From the model card:

| Setup | tok/sec | vs. NLD AR | vs. Qwen3-8B-Eagle3 (360 tok/sec) |
|---|---|---|---|
| NLD-8B AR | 253 | 1.00x | 0.70x |
| NLD-8B Linear SS (default kernels) | 850 | 3.36x | 2.36x |
| NLD-8B Linear SS (custom CUDA kernels) | 1015 | 4.01x | 2.82x |

The headline: at concurrency 1 on GB200, NLD-8B with custom CUDA kernels is 4x faster than its own AR baseline and 2.8x faster than Qwen3-8B accelerated with Eagle3 speculative decoding.

### 2.2 DGX Spark (on-device, concurrency 1)

From the model card (using w4a16 quantisation for both modes):

| Setup | tok/sec | vs. AR |
|---|---|---|
| NLD-8B AR (w4a16) | 41.8 | 1.00x |
| NLD-8B diffusion-mode (w4a16) | 112 | 2.68x |

DGX Spark uses DDR5 memory, so the absolute throughput is lower than a GB200/H100 but the *ratio* is still strong because the memory-bound regime is even more pronounced on low-bandwidth devices.

### 2.3 Other hardware (per tech report sec 6)

The tech report mentions H100, GB200, RTX Pro 6000, and DGX Spark across its figures. Specifically:

- Tech report sec 6 (and Table 9): "compared with Eagle3, linear self-speculation delivers a 2.4x / 2.3x / 1.8x speedup at batch size 1 on GB200 / RTX Pro 6000 / DGX Spark".
- Tech report sec 6: "achieving up to 3.3x speedup over AR" on multi-GPU setups.

NLD does not publish A100 numbers. If you need to deploy on A100, you should expect the speedup ratio to fall somewhere between H100 and DGX Spark, but exact numbers require your own measurement.

---

## 3. Batch-size scaling (qualitative)

Tech report Figure 1(c) and Figure 9(a) plot system throughput vs. per-user throughput at varying concurrency $c$. The qualitative shape:

1. **At low concurrency ($c \leq 8$), self-speculation dominates AR.** Memory-bound regime, weights are loaded once per cycle, more tokens per forward = direct speedup.
2. **At moderate concurrency ($c \approx 16$ to 64), the gap narrows.** AR can pack more sequences per forward, amortising the weight read; self-speculation's advantage shrinks.
3. **At high concurrency ($c \geq 128$), AR catches up or surpasses self-speculation.** Now compute is the bottleneck, not memory. AR uses its FLOPs more efficiently because it doesn't pay the dual-stream / mask-construction overhead.

The tech report does not publish a single batch-vs-throughput table at all batch sizes; we therefore avoid quoting specific (batch, throughput) values to prevent fabrication. For deployment decisions, the rule of thumb is: **use self-speculation for $c \leq 16$, use AR for $c \geq 64$, measure in between**.

---

## 4. Long-context behaviour

For long prompts, the prefill phase (which is unchanged by self-speculation) dominates total latency. Self-speculation only accelerates the generation phase. So:

- At short prompt + long output (chat workload): self-speculation is highly effective, full TPF benefit applies.
- At long prompt + short output (RAG / summarization workload): self-speculation's relative contribution shrinks because prefill dominates wall clock.

Tech report sec 6 does not publish a specific (prompt-len, output-len) speedup table, so we do not quote numbers here. Plan for "self-speculation helps the output phase; prefill is the same as AR".

---

## 5. SOL utilisation: what the published numbers actually say

The tech report's SOL analysis (sec 4.2) gives the cleanest one-number summary of how close NLD's self-speculation is to the theoretical ceiling:

| Metric | NLD-8B Linear SS | SOL ceiling | Ratio |
|---|---|---|---|
| Acceptance-rate level | 6.82 | 7.60 | 89.7% |
| Real TPF | 3.41 | 6.02 | 56.6% |

That is: at the *acceptance-rate* level (counting how many draft tokens the verifier accepts), NLD's linear self-speculation reaches ~90% of the maximum-possible acceptance rate. At the *real TPF* level (accepted tokens divided by total forward passes, including verification), it reaches ~57% of the theoretical ceiling.

The 90% to 57% gap comes from two effects (tech report sec 4.2):

1. **Doubled forward-pass cost.** Linear self-spec uses two forwards per cycle (one diffusion draft + one AR verify). Real TPF = acceptance / 2 by definition.
2. **Prefix-only commit.** Linear self-spec commits only a contiguous prefix of the draft, discarding confident tokens beyond the first rejection.

The tech report identifies these as motivations for "stronger diffusion-mode samplers that can safely commit tokens within a single forward pass". Closing this gap is an open research direction.

---

## 6. Where to deploy

The deployment matrix below is a qualitative summary derived from the speed and quality numbers above. The "predicted speedup" column is anchored to published values (GB200 / DGX Spark from the model card; ratios from tech report sec 6) and clearly marked when illustrative.

| Workload | Hardware (verified by NLD) | Recommended mode | Speedup vs. AR (source) |
|---|---|---|---|
| On-device chat, concurrency 1 | DGX Spark | Diffusion mode (or linear SS) | 2.7x (model card, DGX Spark w4a16) |
| Real-time chat API, concurrency 1 | GB200 | Linear SS (custom CUDA kernels) | 4x (model card, GB200) |
| Real-time chat API, concurrency 1 | RTX Pro 6000 | Linear SS | 2.3x vs Eagle3 (tech report sec 6) |
| High-concurrency cloud API | GB200, H100 | Measure; AR likely competitive | Variable, depends on $c$ |
| Bulk inference (large batches) | Any | AR is generally competitive at high batch | ~1x (qualitative) |

A useful rule of thumb:

> **If your AR baseline is memory-bound (low concurrency, small batch), NLD self-speculation wins. If it's compute-bound (high concurrency, large batch), AR is comparable and may be cheaper to deploy.**

For the cloud LLM business, this means the value of NLD self-speculation is concentrated in the $c = 1$ to $c \approx 16$ service tier where most consumer chat and agent workloads live.

---

## 7. Reproducing NLD numbers

To replicate the published numbers on your own hardware:

| Setup | Spec |
|---|---|
| Hardware | At least one H100 80GB for the in-paper benchmarks; GB200 for the headline 1015 tok/sec figure |
| PyTorch | 2.5+ for FlexAttention (required for the block-diff and sbd-block-diff masks) |
| Transformers | The model card specifies `transformers >= 5.0.0` |
| Inference framework | Either the model card's example calls (`ar_generate` / `generate` / `linear_spec_generate`) or SGLang for production-style benchmarks (tech report Figure 1) |
| Model | `nvidia/Nemotron-Labs-Diffusion-8B` |
| Workload | The model card includes example workloads; for full reproducibility match the SPEED-Bench setup the tech report uses |

### 7.1 Common gotchas

1. **PyTorch < 2.5**: FlexAttention is unavailable. The model falls back to SDPA with a dense mask, which is slower. Upgrade.
2. **CPU thread oversubscription**: PyTorch's CPU threadpool fights with CUDA streams. Set `OMP_NUM_THREADS=1` (or 2) before launch.
3. **`use_cache: false` default**: The released config has `use_cache: false`. For inference benchmarks you must explicitly enable KV caching in your `generate(...)` call.
4. **Quantization**: The model card's headline DGX Spark numbers use `w4a16` quantisation. For BF16 inference, expect lower absolute throughput.
5. **GB200 custom kernels**: The 1015 tok/sec figure on GB200 requires the custom CUDA kernels referenced in the model card. Without them, expect the 850 tok/sec figure.

---

## 8. Cost economics (qualitative)

We deliberately do **not** publish a $/1M-tokens table here because per-cloud spot pricing and quantisation choices change the absolute numbers significantly. The relationship that matters for deployment is:

> Cost per token scales approximately as 1 / TPF, holding hardware and concurrency fixed.

So if NLD-8B linear SS achieves TPF = 6 vs. AR's TPF = 1 at concurrency 1, you pay roughly 1/6th the per-token cost at that operating point. At higher concurrency the ratio compresses (per Sec. 3). To compute concrete dollars-per-million-tokens, multiply your hardware's hourly cost by `1 / (throughput in tok/sec * 3600)` and `1M / 1 token`.

---

## 9. What's left on the table

After all the engineering NLD has done, the remaining headroom flagged in the tech report:

### 9.1 Closing the SOL gap

The tech report's "76.5% more tokens per forward under an optimal sampler" claim quantifies the headroom. Today linear SS achieves 3.41 real TPF; the SOL ceiling is 6.02 real TPF. The path to closing the gap goes through smarter diffusion-mode samplers that can commit tokens beyond the first rejection in a single forward pass. The tech report explicitly identifies this as a future direction.

### 9.2 Quantisation

The DGX Spark numbers use `w4a16`. The model card does not publish FP8 throughput numbers for the larger GPUs, but the FP8 weight regime would double the SOL in the memory-bound regime. This is straightforward engineering and likely already in NVIDIA's internal roadmap.

### 9.3 Scaling

NLD ships at 3B / 8B / 14B (tech report sec 5). The model card lists "SOTA 3B, 8B, 14B dense LM family". The 3B variant gives roughly 2.5x the SOL throughput of the 8B at the cost of some quality on hard reasoning benchmarks. No MoE variants are released as of the tech report.

### 9.4 Tree-based speculative decoding

EAGLE-3 uses a tree of speculative candidates. NLD's linear SS is conceptually a 1-deep tree per block. A k-deep tree (multiple draft candidates per position, verified together in one forward) is plausible but not implemented in the released NLD checkpoints.

---

## 10. The big picture

Tying together everything from Series 1 to 3:

1. **NLD is a single-decoder transformer** with the same shape as a vanilla Mistral (Ministral3-8B). The differences are: (a) joint training with AR + diffusion losses at $\alpha = 0.3$, (b) dual-stream input with structured mask, (c) tri-mode inference via per-layer attention mode switching.
2. **The training pipeline** uses 1T tokens of pure-AR continued pretraining (Stage 1) + 300B tokens of joint AR + diffusion training (Stage 2) + 45B tokens of joint SFT, all started from a Ministral3-8B base. Standard tricks (DP-rank varying mask, global loss averaging, complementary mask) stabilise gradient variance.
3. **The inference pipeline** supports AR, block-diffusion, linear self-spec, and quadratic self-spec. Linear self-spec is the production default. The choice between modes is dynamic per-request.
4. **The SOL analysis** says: at $c = 1$ the real-TPF speedup is ~6x vs AR (linear or quadratic self-spec), with ~57% real-TPF utilisation of the SOL ceiling. The acceptance-rate utilisation is ~90%, identifying sampler design as the main lever for closing the remaining gap.
5. **The VLM extension** adds a Pixtral-style encoder (24 layers, hidden 1024) + 2x2 spatial merge + asymmetric dual-stream. Initialization from Ministral3-8B-Instruct-2512 VLM gives the exact-merge property (no parameter mismatch).
6. **The deployment recommendation**: low-concurrency on-device or single-stream chat -> self-speculation; high-concurrency cloud API -> AR is competitive; measure for your specific workload.

---

## 11. Exercises

1. From Section 1's accuracy table, compute the (accuracy, TPF) trade-off curve for NLD-8B across its four modes. Where does the Pareto front lie? Are there modes that are strictly dominated?
2. Using the GB200 numbers in Section 2.1, compute the per-token wall-clock latency for AR mode and linear SS (custom kernels). What is the latency difference per generated token? At what output length does the difference matter for user perception?
3. Section 5 gives 89.7% acceptance-rate utilisation vs. 56.6% real-TPF utilisation. Explain in your own words why the gap exists and what kind of sampler innovation would close it. Cite Lecture 2.3 for the cycle structure.
4. Using the rule of thumb "cost scales as 1 / TPF", estimate the cost reduction NLD self-speculation provides at $c = 1$, $c = 8$, $c = 32$. State your assumptions for each.
5. The tech report mentions A100 nowhere. If you had to deploy on A100, what experiments would you run first to estimate the self-speculation speedup? What numbers would you look for in the result?

---

## Further reading

- [NLD tech report](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_Diffusion_Tech_Report_v1.pdf): Sec. 4 (SOL), Sec. 5 (family), Sec. 6 (benchmarks). The accuracy and TPF table in Section 1 above is from Table 5 verbatim.
- HF model card: <https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B>. The DGX Spark and GB200 throughput numbers in Section 2 are quoted from this page directly.
- SPEED-Bench (Cao et al. 2024): the throughput benchmark used by the tech report.
- Eagle3 / Qwen3-8B-Eagle3 baselines for comparison.
