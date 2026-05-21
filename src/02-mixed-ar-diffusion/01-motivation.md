# 2.1 — Motivation: why unify AR and diffusion in one model?

> **Goal of this lecture.** Build the case — economic, architectural, and empirical — for training **one** transformer that can serve **both** autoregressive and diffusion modes at inference. By the end you should be able to explain to a sceptic why this is *not* just "two models in a trench coat" and why doing it badly is worse than doing either mode alone.

Background assumed: Series 1 in full, especially Lecture 1.4 (block diffusion) and Lecture 1.6 (AR vs diffusion head-to-head). You should know what TPF means, what `compute_block_mask` produces, and roughly how speculative decoding works.

---

## 1. The single-model premise

Across the AR-diffusion literature there is a recurring observation that two-model setups (drafter + verifier, e.g. EAGLE-3) leave throughput on the table at batch 1 because **the drafter is a second weight load through HBM**. On a memory-bound device — anything from a laptop to a DGX Spark — the drafter forward costs almost as much wall-clock as a tiny target forward, and the drafter's quality bottlenecks acceptance length.

If we could collapse the drafter and the verifier into one network — same parameters, two attention masks — then:

1. The HBM load happens **once**, not twice.
2. The drafter has the **full capacity** of the target model, so it can draft long sequences with high acceptance.
3. The deployment story collapses to a **single checkpoint**, served in different attention modes depending on workload.

This is the *single-model premise* of mixed AR-diffusion training. Whether it actually works depends on whether a single transformer can carry both losses without one cannibalising the other.

> **Subtlety.** "Single model" does not mean "single mode at runtime". NLD shipped with three runtime modes — AR, diffusion, self-speculation — and a small (~36 M-parameter) LoRA adapter that is loaded on top of the base when serving the self-speculation mode. The single-model premise is about the *base* network: there is one set of base weights, jointly trained.

---

## 2. Three reasons a unified model is interesting

### 2.1 Capacity sharing

The naive concern is that two losses compete for the same parameters. The empirical answer from NLD's Table 2 is: they don't. Joint training of AR + diffusion on the *same* tokens gives:

| Eval | AR-only (control) | Joint AR + diffusion | Diff vs. control |
|---|---|---|---|
| MMLU (5-shot) | 65.4 | 65.1 | −0.3 |
| GSM8K (8-shot, CoT) | 73.6 | 73.5 | −0.1 |
| HumanEval (pass@1) | 56.7 | 57.3 | +0.6 |
| MATH-500 (greedy) | 49.2 | 50.1 | +0.9 |
| Avg over 6 reasoning evals | 60.8 | 60.9 | +0.1 |

The accuracy of the AR mode of the jointly-trained model is, to within noise, identical to the accuracy of the AR-only control trained on the same tokens. Two-tailed paired t-test on the six evals gives $p = 0.78$.

Two mechanistic reasons this is not surprising:

- **Different supervision, same forward.** AR and diffusion losses both supervise the same hidden states; the difference is the *target* the head is asked to predict. For tokens that are not masked, the AR cross-entropy already supervises them, and adding a masked-position loss is roughly orthogonal in gradient direction.

- **The masking acts as data augmentation, not a competing objective.** From the AR head's perspective, randomly masking 15% of tokens during training and asking the model to recover them looks suspiciously like dropout-with-context — a regularizer, not a competing task. Indeed several authors (LLaDA, MDLM) report *better* AR accuracy on small-data regimes when a masking loss is added.

There is a regime where the losses **do** conflict: very high mask ratios (> 70%) over many consecutive steps cause the AR loss to degrade. NLD's training schedule (see Lecture 3.3) carefully manages the masking ratio distribution to avoid this.

### 2.2 Single weight load at inference

This is the on-device argument and is the one that matters commercially.

Let $W$ be the size of the base model in bytes (≈ 16 GB for an 8B-parameter BF16 model). At batch 1 on a B200-class GPU with 8 TB/s HBM, a forward pass costs ≈ $W / B = 2$ ms of pure memory streaming.

**Two-model speculative decoding** (drafter + verifier, both stored in HBM):

$$
T_{\text{2-model}} \;=\; T_{\text{drafter}} \cdot d + T_{\text{verifier}},
$$

where $d$ is the draft length. For a 1B-parameter drafter and an 8B verifier on the same GPU, $T_{\text{drafter}} \approx 0.25$ ms and $T_{\text{verifier}} \approx 2$ ms. The 2-model cycle is ~4 ms for ~3 accepted tokens — roughly 750 tok/s.

**Self-speculation in a single 8B model** (diffusion drafts, AR verifies, same weights):

$$
T_{\text{self-spec}} \;=\; T_{\text{block-diff}} + T_{\text{AR-verify}},
$$

where both terms reuse the *same* HBM load (the KV cache and base weights are already in SRAM after the first forward). Empirically NLD measures $T_{\text{block-diff}} \approx 2.8$ ms (diffusion is slightly slower per forward than AR because of the longer block) and $T_{\text{AR-verify}} \approx 2.2$ ms. The cycle is ~5 ms for ~6 accepted tokens — roughly 1200 tok/s.

The two-model approach is *not* slower because of compute. It is slower because the drafter exists *and is loaded from HBM*. Once you stop having a drafter, you stop paying the HBM tax. The unified model thus wins on the hardware where the AR baseline was hurting most: low batch, low concurrency, memory-bound.

> **Pragmatic note.** On a GB200 NVL72 cluster serving batch 256, the same arithmetic flips: the drafter HBM tax amortises across the batch, and the two-model approach can match self-speculation. The mixed AR-diffusion advantage is biggest where the HBM load is the bottleneck.

### 2.3 Mode-switch flexibility

If you have **one** checkpoint that can run in three attention modes (AR / block diffusion / self-speculation), you can pick the mode per request based on workload:

| Workload | Best mode | Why |
|---|---|---|
| High-batch cloud inference (b ≥ 64) | AR | Already compute-bound; diffusion's bidirectional FLOPs don't help. |
| Single-user chat (b = 1, latency-critical) | Self-speculation | Maximum TPF, single weight load. |
| Single-user long-form code generation (b = 1, long output) | Block diffusion | Bidirectional intra-block gives better code coherence. |
| Streaming with mid-response edits | Block diffusion | Diffusion naturally supports infilling. |
| Strict-format outputs (JSON schema) | AR | Diffusion modes have a known weakness on long structured spans. |

A two-model deployment cannot do this without either swapping models (slow) or running both simultaneously (RAM-wasteful). A unified model just sets a flag.

> **Operational note.** NLD's `set_attention_mode(mode)` is a one-line runtime call on the attention block — no recompilation. You can switch modes mid-conversation.

---

## 3. What goes wrong if you don't unify

Three failure modes that you'll see in the literature, each motivating a piece of the NLD design we'll dissect:

### 3.1 Training one mode then bolting on the other

A natural-sounding strategy: pretrain AR (standard recipe), then **fine-tune** with a diffusion objective. This is what many "AR+diffusion at the end" papers do.

The result is consistently bad: the AR loss degrades catastrophically during diffusion fine-tuning because the model's KV-cache-friendly representations get disrupted by the bidirectional intra-block attention. To prevent this, the fine-tuning runs are kept short, but then the diffusion mode never reaches the quality of a from-scratch diffusion model.

This is the AR-diffusion analogue of the *catastrophic forgetting* problem in continual learning.

NLD's answer: **train both losses jointly from the start of stage 2** (after a brief AR-only warmup that stabilises the base), with $\alpha \in [0.3, 0.5]$ on the diffusion loss. We dissect this curriculum in Lecture 3.3.

### 3.2 Sharing parameters but training them with conflicting masking schedules

If your AR loss sees clean tokens at positions $0..t-1$ and your diffusion loss sees masked tokens at positions $0..L-1$ *in the same forward pass* without a structured attention mask separating them, then either:

- The AR loss "sees" masked tokens upstream (impossible — AR's whole point is left-to-right causal conditioning on clean tokens), or
- The diffusion loss is run on a separate forward (double FLOPs).

The solution is the **dual-stream input** + **structured attention mask** that we'll derive from first principles in Lecture 2.2. The mask is the technical core of mixed AR-diffusion training; without it the unification thesis collapses.

### 3.3 Drafting with diffusion but verifying with a different (smaller) AR

Several recent papers ("speculative diffusion decoding", "diffusion-spec") use a *separate* small AR model to verify diffusion drafts. This loses the single-weight-load advantage and reintroduces the drafter-verifier alignment problem.

NLD's self-speculation insists on verify-with-the-same-base, using only a tiny LoRA (≈ 36 M params on top of an 8 B base, < 0.5% extra) to align the diffusion drafter to the AR verifier's distribution. This is the topic of Lecture 2.3 and the LoRA mechanics are in Lecture 3.4.

---

## 4. The intellectual lineage

Mixed AR-diffusion did not appear ex nihilo. The proximate ancestors:

| Year | Work | Contribution |
|---|---|---|
| 2024 | Block Diffusion (Arriola et al., ICLR'25) | First clean factorization of "AR between blocks, diffusion within blocks". KV-cache reusable. |
| 2025 | SDAR (Li et al.) | Confirms block diffusion at 7B scale, introduces threshold-based commit. |
| 2025 | Dream-7B | Joint AR + diffusion training, demonstrates the loss-compatibility result above. |
| 2026 | Nemotron-Labs-Diffusion (Fu et al.) | Adds self-speculation, tri-mode runtime, LoRA drafter alignment, VLM extension. |

Note that *most* of the algorithmic ideas pre-date NLD; what NLD contributes is (i) the engineering effort to make all three runtime modes work in the same model, (ii) the LoRA drafter alignment that unlocks 5–6× TPF, and (iii) the VLM extension with asymmetric dual-stream.

For the reader's purposes, Series 2 is mostly about the *invariants* across these papers — the joint loss, the dual-stream input, the structured attention mask, the self-speculation cycle — because once you understand the invariants, the NLD-specific implementation details in Series 3 read as engineering choices rather than novel research.

---

## 5. What we are about to build (preview of Series 2)

The next four lectures construct the unified model in pieces:

- **2.2 — Joint loss and dual-stream layout.** Derive `L = L_AR + α · L_diff` from scratch, show why the dual-stream input is the right data layout, construct the 2L × 2L structured attention mask (M_BD, M_OBC, M_BC blocks). This is the technical core.
- **2.3 — Self-speculation.** Diffusion drafts + AR verify in one model, KV-cache sharing between modes, linear vs quadratic self-speculation, why a LoRA helps.
- **2.4 — Comparisons.** Mixed AR-diffusion vs MTP, EAGLE-1/2/3, Medusa, pure block-diffusion. When does the unified model win, and when do simpler approaches win?
- **2.5 — Speed-of-Light analysis.** The theoretical ceiling on tokens/sec for any decoder-only LM, and how close NLD gets on different hardware.

By the end of Series 2 you should be able to read the NLD model code (Series 3) without surprises: you'll already know what every attention mask construction is supposed to compute.

---

## 6. Common objections and short answers

Anticipating sceptical questions from someone with a strong AR background:

**Q: "If the diffusion loss is a regularizer, can't I just use dropout?"**
No. Dropout removes activations at random; the diffusion loss removes *tokens* and asks the model to predict them from the *remaining* tokens. This is closer to BERT-style MLM than to dropout, and it gives the model lookahead context that dropout cannot.

**Q: "Why train at $\alpha = 0.5$ instead of $\alpha = 1$ (equal weight)?"**
The diffusion loss has higher variance per token (because of the random mask ratio). Halving its weight is roughly equivalent to halving its effective sample size, which equalises gradient signal-to-noise between the two losses. At $\alpha = 1$, NLD reports the diffusion mode improves slightly but the AR mode degrades by ~1.5 points on MMLU. The 0.5 setting is the empirical sweet spot.

**Q: "Couldn't you just train two separate LoRAs on a shared AR base, one per mode?"**
You could, but you give up the single-weight-load advantage of self-speculation and the diffusion drafter would never see the AR-verifier's hidden state during training. NLD-style joint training produces drafters that natively align with the verifier; LoRA-on-top is strictly weaker.

**Q: "What about Mixture-of-Experts? Couldn't you have AR experts and diffusion experts?"**
You could, but every routed forward still pays the full MoE HBM cost for the routed experts plus the router. The HBM tax doesn't go away; you just hide it behind a load-balancing layer. The unified-dense formulation is simpler and benchmarks better in NLD's ablations.

**Q: "If the AR mode is identical to AR-only, why bother training the joint model at all?"**
The headline is the diffusion + self-speculation modes, not the AR mode. The point of the joint training is that you get the AR mode *for free* (no quality loss) on top of the diffusion mode. If you only ever needed AR, you would not train this model.

---

## 7. Exercises

1. Write down explicitly the chain-rule factorization of $p(y \mid x)$ for (a) pure AR, (b) pure block-diffusion with block size B, (c) mixed AR-diffusion with self-speculation. Identify which conditional independence assumptions each makes.

2. For the joint loss $L = L_{\text{AR}} + \alpha \cdot L_{\text{diff}}$, derive the gradient $\nabla_\theta L$ and identify under what conditions the two gradient terms are orthogonal. (Hint: think about the position of the masked tokens vs the AR target tokens.)

3. Suppose you have an AR-only model and you want to add a diffusion mode without re-training from scratch. List three failure modes that the literature has reported for "AR-then-diffusion" fine-tuning and propose one mitigation for each.

4. Given the table in §2.3 of this lecture, design a serving policy that picks the right mode per request based on (i) batch size, (ii) expected output length, (iii) whether the user is streaming. Express your policy as a decision tree.

Solutions are sketched in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 8. Further reading

- **Block Diffusion** (Arriola et al., 2024). [arXiv:2503.09573](https://arxiv.org/abs/2503.09573). Read §3 (the block factorization) and §4 (the cross-block KV cache).
- **SDAR** (2025). [arXiv:2510.06303](https://arxiv.org/abs/2510.06303). §2 has a clean derivation of joint training at 7B.
- **Dream-7B** (HKU, 2025). [arXiv:2508.15487](https://arxiv.org/abs/2508.15487). Reports the loss-compatibility result and ablates $\alpha$.
- **Nemotron-Labs-Diffusion Tech Report** (Fu et al., 2026). [PDF](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_Diffusion_Tech_Report_v1.pdf). §2 is the relevant motivation section; Table 2 is the loss-compatibility evidence.
- **EAGLE-3** (Li et al., 2024). [arXiv:2503.01840](https://arxiv.org/abs/2503.01840). The current SOTA two-model speculative decoder; necessary background for the comparison in Lecture 2.4.

Next up: Lecture 2.2, where we build the actual training machinery.
