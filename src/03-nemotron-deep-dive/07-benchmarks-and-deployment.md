# 3.7 — Benchmarks and deployment economics

> **Goal of this lecture.** Read NLD's published benchmark tables critically. Compute SOL utilisation for every published number, decompose where speedup comes from (TPF vs hardware vs quality), and end with a deployment-decision flowchart. By the end you should be able to: (i) reproduce any of NLD's tech-report numbers from your own measurements, (ii) decide which inference mode + hardware combination is right for a given workload, and (iii) identify the next 2× speedup opportunity for production.

Background assumed: Lecture 2.5 (SOL analysis), Lectures 3.1–3.6 (NLD code). Comfort with reading benchmark tables and back-of-envelope calculation.

References: NLD Tech Report §5 (benchmarks) and §7 (deployment), augmented with our SOL framework.

---

## 1. The quality side: how good is NLD?

Before talking speed, let's check that NLD isn't trading quality for tokens-per-second. Tech report Tables 1–2 (consolidated):

| Benchmark | Ministral3-8B | NLD-8B AR mode | NLD-8B self-spec | Notes |
|---|---|---|---|---|
| MMLU (5-shot) | 73.2 | 73.5 | 73.5 | Self-spec matches AR by design |
| MATH (0-shot) | 47.0 | 49.5 | 49.3 | Joint training improves MATH |
| HumanEval | 66.5 | 67.4 | 67.4 | Small code improvement |
| GSM8K | 81.5 | 82.0 | 81.8 | Slight regression in self-spec |
| WinoGrande | 76.5 | 76.7 | 76.8 | Reasoning preserved |
| TruthfulQA | 49.5 | 50.1 | 49.9 | Within noise |
| IFEval | 81.0 | 80.5 | 80.5 | Within noise |
| Avg | 67.9 | 68.5 | 68.5 | +0.6 over Ministral3 |

The key takeaways:

1. **Quality is preserved or improved.** NLD's AR mode matches or slightly beats Ministral3, and self-spec matches AR. No quality cost for the speedup.
2. **MATH and HumanEval improve.** The joint training acts as a quality regulariser on reasoning-heavy benchmarks. The most plausible mechanism: the bidirectional intra-block attention during training helps the model see relationships between tokens that AR-only training misses.
3. **No degradation on knowledge benchmarks.** MMLU is unchanged — the LM didn't lose factual recall during joint training.

The 0.6-point average improvement over Ministral3 is small but consistent. The headline is "no harm done, plus a free 5–6× speedup".

---

## 2. The speed side: throughput across hardware

The core deployment numbers. Tech report Table 6 (extracted):

### 2.1 B200 (high-end cloud, b=1)

| Mode | TPF | Tokens/s | SOL Tokens/s | SOL utilisation |
|---|---|---|---|---|
| AR | 1.0 | 280 | 500 | 56% |
| Block-diffusion | 2.8 | 780 | 1400 | 56% |
| Linear self-spec | 6.0 | 1680 | 3000 | 56% |
| Quadratic self-spec | 8.0 | 2240 | 4000 | 56% |

The 56% SOL utilisation is constant across modes — confirming that NLD's per-cycle engineering overhead is mode-independent, just multiplied by TPF.

### 2.2 H100 (mid-tier cloud, b=1)

| Mode | TPF | Tokens/s | SOL Tokens/s | SOL utilisation |
|---|---|---|---|---|
| AR | 1.0 | 120 | 209 | 57% |
| Block-diffusion | 2.7 | 320 | 564 | 57% |
| Linear self-spec | 5.5 | 740 | 1295 | 57% |
| Quadratic self-spec | 7.0 | 920 | 1463 | 63% |

Similar 57% utilisation. Quadratic self-spec gets a small bump (63%) on H100 because the FLOPs-per-cycle increase fits better in the H100's larger SRAM.

### 2.3 A100 (older cloud, b=1)

| Mode | TPF | Tokens/s | SOL Tokens/s | SOL utilisation |
|---|---|---|---|---|
| AR | 1.0 | 70 | 125 | 56% |
| Block-diffusion | 2.6 | 175 | 325 | 54% |
| Linear self-spec | 5.0 | 410 | 750 | 55% |
| Quadratic self-spec | 5.8 | 460 | 725 | 63% |

Slightly lower TPF on A100 because the smaller SRAM limits how many candidate positions can be in flight at once.

### 2.4 DGX Spark (on-device)

| Mode | TPF | Tokens/s | SOL Tokens/s | SOL utilisation |
|---|---|---|---|---|
| AR | 1.0 | 15 | 25 | 60% |
| Linear self-spec | 5.0 | 80 | 150 | 53% |

DGX Spark (consumer/edge with DDR5 memory) gets the highest *relative* speedup: AR at 15 tok/s is unusable for chat, but self-spec at 80 tok/s is. The absolute numbers are low, but the wall-clock difference (~83 ms/token for AR vs ~13 ms/token for self-spec) is what matters at b=1.

---

## 3. Batch-size scaling

Tech report Table 7 reports per-mode throughput at various batch sizes on B200:

| Batch | AR | Block-diff | Linear self-spec | Quadratic self-spec |
|---|---|---|---|---|
| 1 | 280 | 780 | 1680 | 2240 |
| 8 | 2160 | 5040 | 8500 | 8800 |
| 16 | 4100 | 8400 | 12000 | 11500 |
| 32 | 7600 | 11500 | 14500 | 12000 |
| 64 | 13500 | 14000 | 15500 | 13000 |
| 128 | 19000 | 17000 | 16500 | 14000 |
| 256 | 24000 | 21000 | 19000 | 16500 |
| 512 | 28000 | 24500 | 22000 | 18000 |

Three patterns:

1. **AR scales linearly with batch in the memory-bound regime** (b ≤ 32), then sublinearly as it transitions to compute-bound. By b=512 it's at 28K tok/s vs 500 tok/s × 512 = 256K SOL — so 11% utilisation. (The drop reflects compute-boundedness.)

2. **Self-spec's relative speedup decreases with batch.** At b=1 it's 6× AR; at b=64 it's 1.15× AR. By b=128, AR is outright faster. The crossover happens when AR enters compute-bound regime (no more memory-headroom for self-spec to exploit).

3. **Quadratic loses to linear at high batch.** At b=8 it's the best mode, but by b=32 linear self-spec wins, and by b=64 quadratic is the worst of all options. Compute overhead bites hard.

The **deployment lesson**: pick the inference mode per workload, not per model. For batch-1 service tier, use quadratic self-spec. For batch-32+ service tier, use AR (or possibly linear self-spec). NLD's `set_attention_mode` API makes this dynamically switchable.

---

## 4. Long-context scaling

Tech report Table 8 reports performance at varying prompt/output lengths:

| Prompt len | Output len | AR (s) | Self-spec (s) | Speedup |
|---|---|---|---|---|
| 128 | 64 | 0.23 | 0.07 | 3.3× |
| 128 | 256 | 0.91 | 0.21 | 4.3× |
| 128 | 1024 | 3.66 | 0.62 | 5.9× |
| 1024 | 256 | 1.05 | 0.31 | 3.4× |
| 4096 | 256 | 1.42 | 0.62 | 2.3× |
| 16384 | 256 | 3.21 | 1.95 | 1.6× |
| 65536 | 256 | 11.30 | 8.20 | 1.4× |

Observations:

- **Self-spec speedup grows with output length.** More tokens → cycle cost amortises better → speedup approaches SOL × TPF.
- **Self-spec speedup shrinks with prompt length.** Longer prompts → more time spent in prefill (which is unchanged by self-spec) → speedup ratio compressed.
- **At 65K context with 256 output**, the speedup drops to 1.4×. The prefill dominates total time, and self-spec only helps the small generation phase.

For long-context workloads (e.g., document summarisation), the self-spec speedup is modest. For chat workloads (short prompts, longer outputs), self-spec is highly effective.

---

## 5. Decoding the SOL utilisation pattern

The consistent 55–60% SOL utilisation across modes and hardware is striking. Let's decompose where the remaining ~40% goes (combining the analysis from Lecture 2.5):

| Source | Estimated fraction |
|---|---|
| KV-cache attention reads | 5–8% |
| Activation memory traffic | 5–10% |
| RMSNorm + RoPE + small ops | 3–5% |
| Sampling / softmax / postprocess | 2–4% |
| Kernel launch overhead | 2–4% |
| Memory allocator (KV cache append) | 3–5% |
| Mode-switch cost (self-spec only) | 5–10% |
| **Total overhead** | **30–45%** |
| **Resulting utilisation** | **55–70%** |

The "mode-switch cost" is the one that varies between modes:

- AR mode: 0 mode switches per token → no mode-switch overhead → ~60% utilisation.
- Self-spec: 2 mode switches per cycle → mode-switch overhead matters → ~55% utilisation.

To increase utilisation: **fuse the mode switch into the per-layer kernel** so the attention mask is rebuilt without a sync. This is a known engineering opportunity for the NLD team.

Other opportunities:

- **Reduce KV-cache write bandwidth** via FP8 KV cache. Halves the per-token KV traffic.
- **Persistent KV layout** that avoids reallocation. Saves the allocator overhead.
- **Compiled forward pipeline** (torch.compile + CUDA Graphs). Saves kernel launch overhead.

A combination of these could push utilisation to ~75%, giving another ~30% wall-clock improvement on top of the existing self-spec speedup.

---

## 6. Where to deploy

The deployment matrix from this analysis:

| Workload | Hardware | Mode | Predicted speedup vs AR |
|---|---|---|---|
| On-device chat, low concurrency | DGX Spark / M-series | Linear self-spec (no LoRA) | 5× |
| Real-time chat API (low concurrency) | H100 | Quadratic self-spec | 7× |
| Real-time chat API (high concurrency) | H100 / B200 | Linear self-spec | 4–5× |
| Bulk inference (b ≥ 64) | B200 / GB200 | AR (b ≥ 64 outperforms self-spec) | 1× |
| Long-document Q&A (prefill dominated) | H100+ | Self-spec for output only | 1.4× |
| Code completion (strict format, long IDE prompt) | H100 | AR + small Medusa | 1.8× |
| Multi-turn chat (cached KV) | H100 | Linear self-spec | 5× |

A useful rule of thumb:

> **If your AR baseline is memory-bound, NLD self-spec wins. If it's compute-bound, AR is comparable.**

For the cloud LLM business, this means the value of NLD is concentrated in the b=1–8 service tier — where most consumer chat and most agent workloads live. High-batch batch-inference (e.g., overnight bulk processing) gets little benefit.

---

## 7. Reproducing NLD numbers

To replicate the published tech-report numbers yourself, you need:

| Setup | Spec |
|---|---|
| Hardware | At least one H100 80GB; B200 for top-end |
| PyTorch | 2.5+ for FlexAttention |
| Transformers | 4.57+ |
| Inference framework | Custom (NLD ships its own `generate`) |
| Model | `nvidia/Nemotron-Labs-Diffusion-8B` |
| Workload | Synthetic prompts of varying length |

The released model card includes a basic benchmark script (`benchmark.py`) that reproduces b=1 throughput numbers. Running it on H100 SXM should give ~120 tok/s in AR mode and ~740 tok/s in linear self-spec mode — matching Table 6.

For the high-batch numbers, you'll need a multi-GPU setup with FSDP or tensor parallelism. NLD's released code supports both (FSDP for training; TP for inference). The configuration is in `accelerate/launcher.yaml`.

### 7.1 Common gotchas

If your numbers don't match the tech report:

1. **PyTorch < 2.5**: FlexAttention is unavailable. The model falls back to SDPA with a dense mask, which is ~30% slower. Upgrade.

2. **CPU thread oversubscription**: PyTorch's CPU threadpool fights with CUDA streams. Set `OMP_NUM_THREADS=1` (or 2) before launch.

3. **CUDA graphs disabled**: NLD's `generate` uses CUDA graphs for the per-cycle forward. If disabled (e.g., `torch.cuda.graphs_enabled = False`), throughput drops 10–15%. Re-enable.

4. **Quantization unloading**: If you loaded with `bnb_8bit`, the dequantization adds per-forward overhead. Use raw BF16 for benchmarks (or FP8 if you have Hopper).

5. **KV cache misallocated**: A common bug — the dynamic cache reallocates each cycle. Use `StaticCache` or pre-allocate to `max_position_embeddings + max_new_tokens`.

---

## 8. Cost economics for production

Pricing at H100 spot rates of ~$2/hour (3.6K seconds), AR vs self-spec for an 8B model:

| Workload | AR cost per 1M tokens | Self-spec cost per 1M tokens | Savings |
|---|---|---|---|
| Chat, b=1 | $4.63 | $0.77 | 83% |
| Chat, b=16 | $0.29 | $0.18 | 38% |
| Chat, b=64 | $0.07 | $0.06 | 14% |
| Bulk, b=256 | $0.018 | $0.018 | 0% |

The headline: **at b=1, self-speculation is 6× cheaper per token**. This is what matters for low-concurrency services (e.g., agent-style inference, voice agents, on-device).

For high-concurrency cloud APIs (b ≥ 64), the cost difference is negligible. Self-spec's value is in the **latency**, not the cost-per-token.

---

## 9. What's left on the table

After all the engineering NLD has done, the remaining opportunities (Lecture 2.5 §6 expanded):

### 9.1 FP8 weights and KV cache

NLD-8B in BF16 is 17 GB. In FP8, it would be 8.5 GB. The SOL doubles in the memory-bound regime.

NLD's tech report §7 has preliminary FP8 numbers (Table 9 supplementary): ~1.85× speedup, vs the theoretical 2×. The remaining gap is engineering overhead. Production-quality FP8 inference would push self-spec to ~12 TPF at b=1 (effective; vs 8 quadratic in BF16).

### 9.2 Sparse / MoE variants

A 16x4B MoE NLD (16 experts, 4 active per token) would have effective parameter count ~16B but compute cost ~4B. The SOL scaling: 4× fewer bytes per active forward, 2× more in expert routing, net ~2× speedup over a dense 8B.

NLD's roadmap mentions MoE variants but no release as of the tech report.

### 9.3 Smaller models

The 3B variant of NLD (mentioned but not released as of late 2025) would be ~2.5× the SOL throughput at the cost of ~5 quality points on MATH. For some workloads (chat, code completion), this is acceptable.

### 9.4 Speculative decoding meta-tree

EAGLE-3 uses a tree of speculative candidates. NLD's linear self-spec is "1-deep tree". A "k-deep tree" — multiple draft candidates per position, with verification of the whole tree in one forward — could push TPF to ~10. But the implementation is complex and the gain marginal vs quadratic self-spec.

### 9.5 KV-cache compression

Vector quantization on the KV cache (e.g., GQVA, KIVI) reduces per-token KV bandwidth. Combined with FP8 weights, this could push effective SOL ~1.4× higher.

---

## 10. The big picture

Tying together everything from Series 1–3:

1. **NLD is a single-decoder transformer** with the same shape as a vanilla Mistral. The differences are: (a) joint training with AR + diffusion losses, (b) dual-stream input with structured mask, (c) tri-mode inference via `set_attention_mode`.

2. **The training pipeline** uses 2-stage curriculum (pretrain + SFT), DP-rank varying mask, complementary masks, and global loss averaging. Total compute: ~35K H100-hours, plus optional ~1 H100-hour for the LoRA adapter.

3. **The inference pipeline** supports AR, block-diffusion, linear self-spec, and quadratic self-spec. Linear self-spec is the production default. The choice between modes is dynamic per-request.

4. **The SOL analysis** says: at b=1 the speedup is ~5-6× vs AR (linear) or ~7-8× (quadratic). Utilisation is ~55%, leaving room for ~30% more wall-clock improvement via engineering.

5. **The VLM extension** adds a Pixtral encoder + 2x2 spatial merge + asymmetric dual-stream. The "exact-merge" initialization from Pixtral 12B makes the VLM converge in ~70B tokens of joint training.

6. **The deployment recommendation**: low-batch low-concurrency → quadratic self-spec; standard chat API → linear self-spec; high-batch bulk → AR mode. The mode-switching is one API call.

The fundamental insight that makes this work: **discrete diffusion's bidirectional drafting and AR's reliable verification fit together in a single weight matrix**. By training them jointly with the right attention mask, you get both capabilities for the price of one. The rest is engineering.

---

## 11. Exercises

1. The SOL utilisation is consistently ~55–60% across NLD's modes. Decompose the missing ~40% using the framework from §5. Identify the two single-largest sources of overhead.

2. The crossover from "self-spec wins" to "AR wins" happens around b=64. Derive this from the SOL formula and the per-cycle wall-clock numbers.

3. For an FP8 NLD-8B with self-spec, predict the b=1 throughput on B200 and the SOL utilisation. State assumptions.

4. The "memory-bound to compute-bound" crossover formula $b^* = I^* / \text{TPF}$ applies to NLD. For an NLD MoE variant with 4 active experts of 2B each, what's $b^*$ on B200?

5. Suppose a user runs a 32K-prompt / 256-output workload. From the long-context table (§4), predict the cost-per-million-tokens at b=1 on H100 ($2/hour). Compare against the same workload on a 100% AR baseline.

Solutions to (1), (3), (5) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 12. Further reading

- **NLD Tech Report §5** (benchmarks), §6 (LoRA), §7 (deployment economics) — the canonical references.
- **vLLM blog: "Speculative decoding in vLLM"** — for how vLLM integrates speculative-decoding pipelines (NLD self-spec is on the roadmap).
- **NVIDIA "How to optimise LLM inference"** whitepaper — for the kernel-level optimisations that get from 55% to 75% SOL utilisation.
- **MLPerf inference benchmarks** — for standardised throughput comparisons across LMs and hardware.

End of Series 3. Series 4 is the hands-on lab: six Jupyter notebooks that reimplement everything we've discussed at a 3M-parameter toy scale. Each notebook runs on CPU in under 15 minutes.
