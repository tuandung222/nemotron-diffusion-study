# 2.4 Comparisons: self-speculation vs MTP, EAGLE, Medusa, block-diffusion

> **Goal of this lecture.** Place mixed AR-diffusion (specifically NLD's self-speculation cycle) in its competitive landscape. By the end you should be able to (i) explain in 30 seconds the difference between self-speculation and each of MTP, EAGLE-1/2/3, Medusa, and pure block-diffusion, (ii) predict when each approach wins, and (iii) identify the failure modes of each.

Background assumed: Lecture 2.3 (self-speculation cycle). Familiarity with at least one of EAGLE / Medusa / MTP is helpful but we re-state the key bits.

---

## 1. The taxonomy of parallel decoding

Every approach to "more than one token per forward" answers two questions:

1. **Where do candidate tokens come from?** A small drafter, multiple heads, the same model in a different mode, or a non-AR generator.
2. **How are they verified?** A single forward of a verifier model, or no verifier at all (commit-and-pray).

Cross-product:

| Source of candidates | No verifier | Single-forward verifier |
|---|---|---|
| **Separate drafter** |, | Speculative decoding (Leviathan; EAGLE-1/2/3; Medusa-style with separate heads) |
| **Multiple LM heads** | Medusa (multi-token) | MTP (multi-token prediction) |
| **Bidirectional / diffusion** | Pure block-diffusion (LLaDA, MDLM, SDAR) | **Self-speculation (NLD)** |
| **Self with multi-mode** |, | **Self-speculation (NLD)** |

Five approaches, mapped to their structural choices. Let's walk through each.

---

## 2. MTP (Multi-Token Prediction)

**Idea.** During AR training, the model is asked to predict not just the next token but the next $k$ tokens simultaneously. The LM head is duplicated $k$ times (one per future position), and each head emits its own logit distribution. At inference, all $k$ heads are queried in one forward, and the $k$ predictions are committed left-to-right with a per-position acceptance check against the AR distribution of the next forward.

**Origin.** Originally proposed in DeepMind 2024 "Multi-token Prediction" papers; subsequently popularised by DeepSeek-V3 (which uses MTP heads to accelerate training as well as inference) and Meta's "Better & Faster Large Language Models via Multi-token Prediction" (2024).

**Strengths.**

- No drafter, no second model.
- The heads are trained jointly with the base model, so they inherit full capacity and align natively with the verifier.
- Lightweight overhead: $k$ extra heads, typically 1–4% of total parameters.

**Weaknesses.**

- Each head sees the same hidden state and so produces *correlated* mistakes: when head 1 is uncertain, heads 2 through $k$ are usually also uncertain at the same position. Acceptance falls off fast at $k > 3$.
- The training cost scales with $k$ (you need $k$ separate label tensors per batch).
- Quality on hard reasoning evals (MATH, code) is consistently 0.5–1 point below the equivalent AR-only model, the heads compete for capacity at the final-layer hidden state.

**TPF.** ≈ 2–3 at $k = 3$, with diminishing returns past that.

**Relation to NLD self-speculation.** MTP and self-speculation share the "one model, multi-mode" philosophy. The difference is *where* the parallel candidates come from:

- MTP: $k$ separate heads, each predicting one future position from the same hidden state.
- Self-speculation: a single head, but applied $B$ times under bidirectional attention to a block of `[MASK]` positions.

NLD explicitly characterises itself in §2 of the tech report as **"MTP without the extra heads"**, the diffusion mode gives you the same multi-token-per-forward capability as MTP, but the prediction comes from running the same head over multiple positions with a different attention pattern rather than running multiple heads at the same position.

This has two consequences:

1. **NLD's drafts are not correlated in the way MTP's are.** Each masked position in NLD's draft attends to all the others bidirectionally, so the draft is *globally consistent* across positions in the block. MTP's drafts are not, each head independently produces a top-1 for its position.

2. **NLD doesn't pay the AR-head-capacity-competition cost.** The diffusion drafts and AR predictions share the same LM head; there's no "training" of separate heads.

---

## 3. EAGLE-1, EAGLE-2, EAGLE-3

**Idea.** A specialised drafter model, typically 1B parameters on top of an 8B verifier, that takes the verifier's *intermediate hidden states* as inputs and emits draft tokens. The drafter is trained per-target: a different drafter for each verifier model.

**Variants.**

- **EAGLE-1.** Drafter sees the verifier's last hidden state.
- **EAGLE-2.** Drafter recurrently re-uses its own hidden state across draft positions.
- **EAGLE-3.** Drafter takes multi-layer hidden states from the verifier (not just last) as input, with a learnable gating over which layers to use. Plus tree-style branching at uncertain positions.

**Origin.** Li, Wang, Yao, et al. (2024); the EAGLE-3 paper is [arXiv:2503.01840](https://arxiv.org/abs/2503.01840).

**Strengths.**

- High per-position acceptance ($\alpha \approx 0.75$ on EAGLE-3).
- Drafter is much smaller than the verifier (1B vs 8B), so the marginal HBM cost is small relative to a same-size drafter.
- TPF on EAGLE-3 reaches ≈ 4 on average open-domain workloads.

**Weaknesses.**

- **Drafter is a second weight load.** At batch 1, the drafter forward is ~25% of a verifier forward in wall-clock, non-trivial.
- **Drafter must be trained per target.** A new base model means a new EAGLE drafter. This is a meaningful operational cost.
- **Drafter is capacity-limited.** Even at 1B params, the drafter's distribution is less accurate than the 8B verifier's, capping $\alpha$.
- **Tree branching is finicky.** EAGLE-3's tree-attention machinery adds significant code complexity for a small TPF gain.

**TPF.** ≈ 3 (EAGLE-1), ≈ 3.5 (EAGLE-2), ≈ 4 (EAGLE-3).

**Relation to NLD self-speculation.** EAGLE-3 and NLD self-speculation are the two strongest approaches as of 2026. Direct comparison from NLD §6:

| Metric | EAGLE-3 (8B + 1B drafter) | NLD self-spec (8B) |
|---|---|---|
| Total params | 9B | 8B + optional 36M LoRA |
| Acceptance rate | 0.75 | 0.92 |
| TPF (b=1, B200) | 4.0 | 6.0 (linear), 8.0 (quadratic) |
| HBM cost per cycle | 1.25× verifier | 1.0× verifier |
| Drafter training | Specialised (1B-1B per target) | Free (joint pretraining) |
| Maturity | Production (Anthropic, others) | Newer (NVIDIA) |

The headline: NLD self-speculation does what EAGLE-3 does, with higher acceptance, no separate drafter, and no per-target drafter training. The cost is that NLD requires the joint AR+diffusion pretraining recipe; EAGLE-3 can be bolted on to any existing AR model.

---

## 4. Medusa

**Idea.** Like MTP, Medusa adds $k$ "medusa heads" to a pretrained AR model. Unlike MTP, the heads are added *after* base pretraining as a fine-tuning step, so the base model is unchanged. The heads are small MLPs that predict candidate tokens at offsets $+1, +2, ..., +k$ from each position. Verification is a single forward of the base AR model.

**Origin.** Cai, Li, Geng, et al. (2024). [arXiv:2401.10774](https://arxiv.org/abs/2401.10774).

**Strengths.**

- Cheap to retrofit: no base-model retraining, just train the heads on a few billion tokens.
- No separate drafter model; the heads are tiny (~1% of base).
- Easy to deploy.

**Weaknesses.**

- Heads are trained *frozen against* the base, so they cannot use the gradient signal to adjust the base. The heads inherit the base's capacity at the *cost* of correlated mistakes across heads.
- TPF saturates at ≈ 2–2.5 even with sophisticated head architectures.
- Quality on hard evals drops 0.5–1 point because the head training contaminates with non-AR objectives.

**TPF.** ≈ 2–2.5.

**Relation to NLD.** Medusa is the "easy retrofit" option for accelerating an existing AR model without touching the base. It's strictly weaker than EAGLE-3 (lower acceptance) and far weaker than NLD self-speculation (the heads can't compete with NLD's bidirectional intra-block draft). The Medusa positioning is "low effort, modest gain"; NLD self-speculation requires substantial training effort but delivers 2× the TPF.

---

## 5. Pure block-diffusion (LLaDA, MDLM, SDAR, block-diffusion)

**Idea.** No AR mode at all. The model is purely a diffusion LM. Inference is iterative denoising over the output sequence, optionally factored into blocks (Lecture 1.4).

**Origin.** LLaDA (Nie et al. 2024), MDLM (Sahoo et al. 2024), Block Diffusion (Arriola et al. 2024), SDAR (Li et al. 2025).

**Strengths.**

- Conceptually clean: one model, one mode.
- Native infilling and editing (can mask any positions and recover them).
- Bidirectional context within blocks gives strong code/math quality.
- No drafter, no separate verifier.

**Weaknesses.**

- **No verifier.** You commit tokens based on the diffusion confidence threshold, with no AR cross-check. Quality vs threshold is a delicate trade-off.
- **TPF is capped by threshold.** Default threshold $\tau = 0.9$ gives TPF ≈ 2.5–3.
- **No AR mode.** You can't fall back to AR for workloads where diffusion is weaker (structured-format generation, strict reasoning chains).

**TPF.** ≈ 2.5–3.

**Relation to NLD.** Pure block-diffusion is the *predecessor* of mixed AR-diffusion. NLD adds the AR verify step (and the self-speculation cycle) on top of block-diffusion, raising TPF from 2.5–3 to 6–8 *and* getting a free AR mode. Pure block-diffusion is now strictly dominated by mixed AR-diffusion for any workload where you might want both modes.

The one exception: if you absolutely need *only* the diffusion mode (e.g., a dedicated infilling service), pure block-diffusion is simpler. But this is a narrow niche.

---

## 6. The comparison table

Putting everything in one place:

| Approach | TPF (b=1) | Extra params | Per-target training | KV-cache reuse | Mode flexibility |
|---|---|---|---|---|---|
| AR baseline | 1.0 | 0 | n/a | yes | AR only |
| MTP | 2.5 | $k$ heads (~3%) | yes (jointly) | yes | AR + multi-token |
| Medusa | 2.3 | $k$ heads (~1%) | yes (post-hoc) | yes | AR + multi-token |
| EAGLE-1 | 3.0 | 1B drafter | yes (per target) | yes | AR + spec |
| EAGLE-2 | 3.5 | 1B drafter | yes (per target) | yes | AR + spec |
| EAGLE-3 | 4.0 | 1B drafter | yes (per target) | yes | AR + spec |
| Block-diffusion | 2.8 | 0 | yes (from scratch) | yes | diffusion only |
| **NLD self-spec linear** | **6.0** | 0 (+ optional 36M LoRA) | yes (joint pretrain) | yes | **AR + diff + self-spec** |
| **NLD self-spec quadratic** | **8.0** | 0 (+ optional 36M LoRA) | yes (joint pretrain) | yes | **AR + diff + self-spec** |

NLD wins on TPF, ties or wins on every other axis. The cost is the joint pretraining recipe, which is more involved than retrofitting Medusa or EAGLE to an existing AR model.

---

## 7. When NLD-style self-speculation does **not** win

Three regimes where simpler approaches are competitive or better:

### 7.1 High-batch cloud inference (b ≥ 64)

At high batch, all approaches become compute-bound rather than memory-bound. Self-speculation's "single HBM load" advantage vanishes because the HBM load amortises across the batch. EAGLE-3's drafter HBM cost also amortises.

In this regime, **all parallel-decoding approaches converge to roughly 1.2–1.4× the AR baseline**, because the marginal benefit of extra acceptance is dominated by the marginal cost of extra compute per cycle. NLD's tech report confirms this: at b=256 on GB200, NLD self-speculation reports 1.4× speedup over AR, identical to EAGLE-3 and slightly better than MTP.

### 7.2 Strict-format outputs

If the output is a strict JSON or XML schema, the AR distribution at each position is essentially a one-hot (e.g., "the next character must be `}`"). Self-speculation's drafts virtually always match, so $\alpha \to 1$ and TPF saturates at $\frac{1}{2(1 - \alpha)} \to \infty$.

But the same is true for **any** parallel approach: MTP, EAGLE, even simple speculative decoding. The acceptance rate is so high that the cheapest approach wins. NLD's per-cycle wall-clock is slightly higher than a vanilla AR forward, so for strict-format outputs, plain AR or a simple greedy MTP is often faster end-to-end.

### 7.3 Workloads dominated by prefill

If 90% of the request's compute is in the prompt prefill (e.g., a 16K-token context with a 64-token completion), the decoding speedup matters less than the prefill speed. All approaches achieve roughly the same prefill speed because prefill is the same compute regardless of decoding strategy. NLD's self-speculation advantage shrinks proportionally.

NLD's tech report Table 8 includes this scenario: at a 16K-token prefix / 256-token completion, NLD's wall-clock speedup vs AR baseline is only 1.6× (compared to 3.3× at short prompt / long completion). EAGLE-3 reports 1.4× in the same regime.

---

## 8. Recommendation matrix

For practitioners deciding which approach to deploy:

| Workload | Recommended approach |
|---|---|
| On-device chat, b=1, latency-critical | **NLD self-speculation (linear)** |
| Long-form math/code reasoning, b=1 | **NLD self-speculation (quadratic)** |
| High-batch cloud inference, b ≥ 64 | NLD AR mode or EAGLE-3 (same wall-clock) |
| Quick retrofit of existing AR LM | Medusa (no training) or EAGLE-1 (light training) |
| Strict-format output (JSON, code completion) | Vanilla AR or Medusa |
| Long prompt, short completion | Any (prefill-dominated) |
| Infilling-only | Block-diffusion |
| Want all three modes | **NLD self-speculation** |

The recurring pattern: NLD is the right choice when you (a) need batch-1 throughput, and (b) can afford the joint pretraining recipe. For retrofits or high-batch workloads, simpler approaches are competitive.

---

## 9. The "what does NLD actually add over block-diffusion" question

A natural question from someone who's read Series 1 carefully: if block-diffusion already gives bidirectional intra-block draft, what new technical idea does NLD bring?

Answer: **the AR-verify step inside the same model**, made possible by the dual-stream training. This single design choice unlocks:

1. **Quality regularisation.** The AR verify acts as a high-confidence sanity check on the diffusion draft. Tokens that would have been wrong under AR get filtered before commit, raising final quality by 1–2 points on math evals.

2. **Threshold-free decoding.** Pure block-diffusion needs a confidence threshold $\tau$ that trades off TPF against quality. Self-speculation can run with $\tau$ effectively close to 0 (commit on AR match alone), so you get high TPF without losing quality.

3. **Mode flexibility.** Pure block-diffusion can't fall back to AR. Self-speculation can: for strict-format outputs, just set $B = 1$ and the cycle reduces to AR-only.

These three benefits flow from one architectural change: **train the model jointly with both losses + dual-stream input + structured attention mask**. Everything else is engineering.

---

## 10. Exercises

1. For each of MTP, Medusa, EAGLE-1, and block-diffusion, identify which of NLD's three benefits (quality regularization, threshold-free decoding, mode flexibility) the approach achieves. Note any that achieve a strict superset of what NLD achieves on some axis.

2. The comparison table in §6 lists TPF as a single number per approach. In reality TPF depends on the workload. Sketch a 2D plot with workload type on the x-axis (e.g., math reasoning, open-domain dialog, code, strict-format) and TPF on the y-axis, with one line per approach. Where does NLD's lead shrink?

3. Suppose you're a startup with $50K compute budget and an existing 8B AR base. You want to ship a 2× decoder speedup in 4 weeks. Which approach do you pick and why? Justify your trade-offs.

4. Devise a workload where Medusa beats NLD self-speculation. Hint: think about prefill cost.

5. Read EAGLE-3 §3 (drafter architecture). Identify two design choices that NLD self-speculation *cannot* replicate even in principle. Justify in 2–3 sentences each.

Solutions to (1), (3), (5) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 11. Further reading

- **EAGLE-3** (Li et al., 2024). [arXiv:2503.01840](https://arxiv.org/abs/2503.01840). The current SOTA two-model spec, and NLD's main competitor for batch-1 inference.
- **Medusa** (Cai et al., 2024). [arXiv:2401.10774](https://arxiv.org/abs/2401.10774). The cheap-retrofit baseline. Read §2 + §3.
- **DeepSeek-V3 Tech Report** (DeepSeek-AI, 2024). The MTP section (§3.4) is the cleanest production write-up of multi-token prediction.
- **Block Diffusion** (Arriola et al., 2024). [arXiv:2503.09573](https://arxiv.org/abs/2503.09573). The "pure block diffusion" baseline.
- **Nemotron-Labs-Diffusion Tech Report** (Fu et al., 2026). §6 contains the head-to-head comparison numbers used in this lecture.

Next, in Lecture 2.5: what's the *theoretical ceiling* on TPF? How close to that ceiling does NLD self-speculation actually get?
