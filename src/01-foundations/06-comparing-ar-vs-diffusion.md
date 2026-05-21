# 1.6 AR vs Diffusion LMs: a head-to-head

> **Goal of this lecture.** Close Series 1 with a clear-eyed comparison: where AR LMs win, where diffusion LMs win, and what the recent benchmark numbers actually mean once you control for confounders. We'll use NLD-8B and LLaDA-8B as the diffusion reference, and Qwen3-8B / LLaMA-3-8B as the AR reference. The conclusion sets up the motivation for Series 2 (mixed AR-diffusion).

Series 1 has been mostly mechanical: forward processes, losses, masks, samplers. This lecture is the first one with a real "engineering judgment" flavor. There's no single number that decides AR vs diffusion; you have to read multiple benchmarks correctly.

---

## 1. Axes of comparison

When people compare AR and diffusion LMs, they conflate several axes. Let's separate them:

1. **Accuracy at fixed compute.** For the same training FLOPs, who gets higher accuracy?
2. **Wall-clock throughput at batch 1.** Tokens per second on a single GPU with one user.
3. **Wall-clock throughput at production batch sizes.** Tokens per second per GPU at batch 32, 128, 256.
4. **Latency-per-token (TTFT and TPOT).** Time to first token, time per output token.
5. **Generation quality under decoding constraints.** E.g., strict format compliance, exact string match, infilling.
6. **Diversity.** Multiple distinct samples for the same prompt.
7. **Tail behavior.** How often does each mode produce a catastrophic failure (degenerate repetition, off-topic drift, broken JSON)?
8. **Training stability and cost.** From-scratch vs continual pretraining; convergence behavior.

Each axis has a "winner" and the winners are different. The honest answer to "is diffusion better?" is "it depends on which axis you care about, and on the model scale".

---

## 2. Axis 1: Accuracy at fixed compute (training FLOPs)

The hardest comparison to do honestly: how much FLOPs does it take to reach a given accuracy?

Best-known data points:

- **LLaDA-8B** (Nie et al., 2024): trained from scratch on ~2T tokens with the MDLM loss. Achieves ~MMLU 65.5, GSM8K 79.9 (zero-shot).
- **LLaMA-2-7B**: trained from scratch on ~2T tokens with AR loss. MMLU 45.3, GSM8K 14.6 (zero-shot, base model, not instruct).
- **LLaMA-3-8B**: ~15T tokens with AR loss. MMLU 67, GSM8K 79 (zero-shot, instruct).
- **NLD-Diffusion-8B-Instruct**: continual pretrain from Ministral3-8B-Instruct for ~1T additional tokens with the joint AR + diffusion loss. AR mode MMLU 73.3, GSM8K 90.1 (Table 8 of paper).
- **Qwen3-8B**: ~36T tokens with AR loss. MMLU 71.0, GSM8K 87.0 (instruct).

Reading these carefully:

1. LLaDA-8B at 2T tokens beats LLaMA-2-7B at 2T tokens on MMLU and GSM8K *zero-shot, base model*. That's strong evidence the diffusion loss is at least *competitive* per-FLOP with the AR loss at this scale.

2. LLaDA-8B at 2T tokens does *not* beat LLaMA-3-8B at 15T tokens. Scale matters more than training objective, at least up to this point.

3. NLD-Diffusion-8B-Instruct (which uses *both* losses) matches Qwen3-8B-Instruct trained on $\sim$ 35× more tokens, but it has the benefit of warm-starting from a strong AR model. The marginal cost of the diffusion loss on top of AR is small (the paper's Section 5.4 ablation).

**Tentative conclusion:** the diffusion loss is competitive with the AR loss for compute up to ~10²² FLOPs (~8B params, ~2T tokens). It is not clearly better. The interesting story is in joint training (Series 2), which yields strict improvements over either.

> **Caveat about data**. Most diffusion-LM papers (LLaDA included) report training on smaller corpora than current AR LLMs because their compute budget is smaller. We don't yet have a published "diffusion 70B trained on 15T tokens" data point. The fair comparison requires equal compute, equal data, equal tokenizer, equal context length, and it doesn't exist in 2026 outside of mid-size models. Plan to update this section when the data lands.

---

## 3. Axis 2: Throughput at batch 1 (the headline number)

This is where diffusion LMs shine.

NLD-Diffusion-8B numbers, from §6 of their tech report:

| Mode | TPF | Tok/s (DGX Spark, M=1, BF16) | Tok/s (B200, M=1, FP8) |
|---|---|---|---|
| AR | 1.0 | 38 | 97 |
| Diffusion (block 32, $\tau$=0.9) | 2.57 | 79 (2.1×) | 198 (2.0×) |
| Linear self-speculation | 6.06 | 138 (3.6×) | 296 (3.0×) |
| Quadratic self-speculation (FlexAttention) | 7.13 | 142 (3.7×) | 305 (3.1×) |

(Approximate, read off their figures, my numbers may be slightly off.)

Two takeaways:

1. **Diffusion-only mode at $\tau = 0.9$ gives ~2× speedup over AR at batch 1.** This is robust across hardware (Spark, H100, B200).
2. **Self-speculation gives another 1.5× on top.** This is the killer feature, it's not just a sampler tweak, it's a different inference algorithm that exploits the joint AR+diffusion training.

For the batch-1 single-user regime (on-device, agent, streaming), diffusion-mode NLD wins decisively.

---

## 4. Axis 3: Throughput at production batch sizes

This is where the picture inverts.

At batch 32+:

| Mode | Tok/s/req (B200, batch 32, FP8) |
|---|---|
| AR | 192 |
| Diffusion (block 32) | 138 |
| Linear self-speculation | 76 |

(The exact numbers depend on the prompt mix and the serving stack; ballpark from NLD §6.4.)

What's happening:

- At batch 32, AR is no longer memory-bound, the weights are amortized over 32 concurrent users. AR's compute is fully utilized, throughput is maximized.
- Diffusion's per-forward sequence length is longer than AR's (it processes a whole block at once), so each forward is *slower per request* in compute-bound regime. The TPF benefit gets eaten by per-forward compute.
- Self-speculation is even worse at high batch because the draft+verify cycle doubles per-cycle compute, which is no longer free.

**Diffusion is slower than AR for high-concurrency serving.** This is the trade-off the NLD paper makes very explicit: it positions diffusion as a *per-user* speedup, not a *system-throughput* speedup. Different use cases optimize different points on this curve.

> **NLD's claim, paraphrased:** "We do not aim to maximize aggregate throughput; we aim to give a single user faster generation, which we believe is the bottleneck for agent, on-device, and long-CoT use cases. For high-concurrency, just run AR mode of the same model."

The fact that NLD is *one model* that can switch to AR mode at inference time (lecture 1.4 §6 / NLD §3) means you don't lose anything by having both options.

---

## 5. Axis 4: Latency: TTFT and TPOT

**Time to first token (TTFT)** is the time from request arrival to the first output token. **Time per output token (TPOT)** is the average wall-clock per subsequent token.

- AR: TTFT $\approx$ prompt forward time (prefill). TPOT $\approx$ single-token decode time.
- Diffusion (block 32, threshold 0.9, $K_B \approx 12$): TTFT $\approx$ prompt forward + ~12 block-decode forwards (because you have to denoise the first block to commit any token). TPOT $\approx$ block-decode forward / TPF.

So diffusion has *worse* TTFT (you have to wait for the first block to finish denoising), but *better* TPOT (after the first block, each new committed token is cheap).

For chat / streaming use cases this can be annoying, the first token takes longer to appear. NLD's "small first block" trick (allow $B < 32$ for the first block) gets back most of the latency without hurting steady-state TPF.

For long generation (chain-of-thought, agent traces), the TPOT advantage dominates and total latency is much lower in diffusion mode.

---

## 6. Axis 5: Quality under decoding constraints

**Strict format compliance.** AR with constrained decoding (FSM, JSON-mode) trivially works because the model commits one token at a time. Diffusion needs more care: you have to ensure the bidirectional context within a block doesn't violate the constraint. Some samplers (FSM-guided diffusion) exist but are less mature. **Edge: AR.**

**Infilling.** Inserting a span in the middle of an existing text. AR needs a special infilling-trained head (FIM); diffusion does it natively by masking the target span and unmasking it conditioned on both left and right context. **Edge: Diffusion.**

**Exact string match.** When the answer is a known short string, AR can be very confident very fast (single token argmax). Diffusion can be slower because the threshold sampler may need multiple iterations even for a "trivial" answer. **Edge: AR for short answers, diffusion for long structured answers.**

**Beam search.** Beam search is a natural fit for AR; for diffusion it requires beaming over the joint denoising path which is much more expensive. NLD does not implement beam search in diffusion mode. **Edge: AR.**

---

## 7. Axis 6: Diversity

For tasks where you want $N$ distinct samples from the same prompt:

- AR with temperature gives diversity via per-token sampling. The samples can be highly correlated (the high-probability paths dominate).
- Diffusion with ancestral sampling gives diversity from the start of generation, different initial unmask orderings lead to different completions.

Empirically, diffusion samples are *more diverse* at matched accuracy. This matters for use cases like creative writing (multiple candidate generations), data augmentation (synthetic training data), and Monte Carlo decoding strategies.

**Edge: Diffusion.**

---

## 8. Axis 7: Tail behavior

How often does each mode produce a catastrophic failure?

- AR degenerates into repetition for low-temperature greedy sampling on certain prompts (the classic "I'm sorry, but I can't help with that. I'm sorry, but..." loop). Caused by attractors in the AR distribution.
- Diffusion degenerates into local incoherence, masked positions get committed with confident-but-wrong tokens that conflict with each other in a way AR would have avoided. Caused by independence assumptions in the within-block forward.

In published benchmarks both modes have a ~1% catastrophic-failure rate on long open-ended generation; the failure modes are *different* but the rates are comparable. **Edge: roughly tied.**

---

## 9. Axis 8: Training stability and cost

Diffusion training has higher gradient variance than AR (the $1/t$ weighting; lecture 1.3 §3.2). It requires:

- Slightly higher LR ($\sim 1.5\times$ AR's).
- Larger batch sizes (to dampen variance).
- The "global loss averaging" trick (Series 2) for very large-scale runs.

For 8B-scale runs the cost is modest, LLaDA was trained on the same hardware footprint as a comparable AR LM. At 70B+ scale the gradient variance is reported to be more problematic; there are no public 70B+ pure-diffusion LMs at the time of writing.

**Joint AR+diffusion training**, as in NLD, **dominates** pure diffusion on stability, the AR loss provides a low-variance backbone gradient. This is one of the strongest empirical arguments for the mixed regime. **Edge: AR (alone) > Joint (AR+diff) >> Diffusion (alone) for stability.**

---

## 10. Bringing it together

A summary table (subjective scoring, "+" = better, "−" = worse, "=" = tied):

| Axis | AR | Pure Diffusion | Mixed AR-Diffusion (NLD) |
|---|---|---|---|
| Accuracy at fixed compute | + | + | + |
| Throughput @ batch 1 | − | + | ++ (with self-spec) |
| Throughput @ batch 32+ | ++ | − | + (AR mode) |
| TTFT | + | − | =, +/− mode-dependent |
| TPOT | − | + | ++ (with self-spec) |
| Constrained decoding | ++ | = | ++ (use AR mode) |
| Infilling | − | ++ | ++ |
| Diversity | + | ++ | ++ |
| Tail failures | = | = | = |
| Training stability | ++ | = | + |

The NLD column dominates almost every row. The only "loss" is that it's a more complex training pipeline (Series 2) and a larger inference stack. But you get a *single set of weights* that can be deployed as AR, diffusion, or self-speculation depending on the per-request need.

This is the value proposition that motivates the entire NLD line of work. **You don't have to choose.**

---

## 11. What's next

Series 1 is now closed. You have:

- The motivation (lecture 1.1).
- The theory (1.2, 1.3, 1.4).
- The algorithms (1.5).
- The empirical landscape (this lecture).

The next time you read a diffusion-LM paper you should be able to map every section into one of these lectures and immediately identify what's standard vs novel.

**Series 2** picks up the threads we've left dangling:

- *Joint training:* how to optimize AR loss and diffusion loss simultaneously without sacrificing either (NLD's α-weighted joint objective).
- *Dual-stream input:* how to compute *both* losses in *one* forward pass (the $[\text{noisy} \,\Vert\, \text{clean}]$ trick).
- *Structured attention masks:* M_BD / M_OBC / M_BC, the decomposition that makes the dual stream work.
- *Self-speculation:* turning the AR head into a verifier for the diffusion drafter, getting 6+ TPF essentially for free.
- *Speed-of-light analysis at scale:* how close NLD gets to the theoretical ceiling.

The deep-dive on NLD's specific code, config, and benchmarks lives in **Series 3**, and the hands-on labs in **Series 4**.

---

## Suggested follow-up readings

- *Nemotron-Labs-Diffusion-8B model card* on HuggingFace, the most current benchmark numbers.
- *LLaDA* paper §6, the most thorough head-to-head AR vs diffusion at 8B scale (pre-NLD).
- *Dream-7B* paper §4, an alternative 7B-scale diffusion LM; useful to triangulate LLaDA's numbers.
- *Mercury (Anthropic, 2024)* and *Inception-V (DeepMind, 2025)*, closed-source diffusion LMs; only the announcements are public but they give some directional signals.
- *DeepSeek MTP / EAGLE-3*, AR-side competitors for self-speculation; we'll compare against them in Series 2.

## Exercises

1. **Reproduce the throughput table.** Use a small HuggingFace AR model (LLaMA-3-8B) and a small diffusion LM (LLaDA-8B from HF). Generate 256 tokens at batch 1 and batch 32, measure wall-clock. Verify the qualitative pattern of §3 and §4, diffusion wins at batch 1, AR wins at batch 32.

2. **TTFT measurement.** For the same models, measure time-to-first-token. Use a 1024-token prompt. Show that AR's TTFT is roughly the prefill time, and diffusion's TTFT is prefill + $K_B$ decode forwards.

3. **Diversity quantification.** Define "self-BLEU" between $N$ sampled completions (lower = more diverse). Generate $N = 8$ samples for the same prompt with each model, with matched temperature. Show that diffusion's self-BLEU is lower (i.e., samples are more diverse).

4. **Reading a benchmark table.** Find a recent diffusion-LM paper (any of LLaDA, Dream, SDAR, NLD). Pick one row of their main results table. Identify (a) what data was used to train, (b) what tokenizer, (c) what context length, (d) what evaluation protocol. Sanity-check that the comparison to an AR baseline controls for these. (You'll find that often it doesn't, and this is how papers can claim wins that don't hold under fair comparison.)

5. **The 70B question.** What would have to be true empirically for "pure diffusion is the right thing to do at 70B scale" to be the right answer? List two specific data points that would convince you. (This is open-ended; the point is to commit to a falsifiable view.)

---

End of Series 1. Series 2 begins with **why** you'd want to train AR and diffusion losses together, and what the joint training pipeline looks like in practice.
