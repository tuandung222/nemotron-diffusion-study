# 2.3 — Self-speculation: drafting with diffusion, verifying with AR

> **Goal of this lecture.** Take the unified AR + diffusion model from Lecture 2.2 and use it at inference in *both* modes within a single token-generation cycle: the diffusion mode emits a long draft in one forward, the AR mode verifies it in a second forward, and the longest matching prefix is committed. The cycle is what NLD calls **self-speculation** and is the source of the 5–6× TPF numbers in the tech report. By the end you should be able to describe the cycle step by step, write the acceptance rule, and explain why a single 8B model can outperform an 8B+1B drafter setup despite using the same compute.

Background assumed: Lectures 1.5 (sampling), 2.1 (motivation), 2.2 (dual-stream + structured attention mask). A working knowledge of Leviathan/Chen-style speculative decoding is useful but not strictly necessary — we re-derive the relevant bits.

---

## 1. Why self-speculation exists

Recall the two known ways to extract more than one token per weight load (Lecture 1.1):

1. **Speculative decoding** (Leviathan/Chen): a small *drafter* model produces $k$ candidate tokens; the *target* model verifies them in one forward. TPF $= 1 + \mathbb{E}[\text{accepted}]$, capped by the drafter's quality.
2. **Non-AR / diffusion**: the model itself produces multiple tokens in one forward via bidirectional intra-block attention.

Both improve over vanilla AR, and both have a ceiling:

- Speculative is limited by **drafter quality**. A 1B drafter on top of an 8B target reaches TPF ≈ 3 on most workloads; EAGLE-3 pushes it to ≈ 4 with a specialised drafter trained per-target.
- Diffusion is limited by **commit threshold and noise**. Confidence-threshold sampling at $\tau = 0.9$ gives TPF ≈ 2.5–3 on a well-trained model; aggressive thresholds risk quality.

Self-speculation **combines** both into one model:

> Use the model's *diffusion mode* as the drafter, then immediately use the model's *AR mode* as the verifier, sharing the same weights and the same KV cache.

The result is an algorithm with TPF $\geq \max(\text{TPF}_{\text{spec}}, \text{TPF}_{\text{diff}})$ that requires no extra parameters, no separate drafter, no extra HBM load. NLD measures TPF $\approx 6$ at the linear variant and TPF $\approx 8$ at the quadratic variant.

This lecture explains both variants.

---

## 2. The linear self-speculation cycle

We describe **linear self-speculation** first (the simpler variant) and treat quadratic self-speculation as a refinement in §5.

### 2.1 State at cycle start

At the beginning of cycle $c$, the model has:

- A *committed prefix* $y_{0:t}$ of length $t$ — these are tokens the user has either prompted with or that have been committed in earlier cycles.
- A *KV cache* for the committed prefix, stored in HBM. This KV cache was built either by the AR verify pass of the previous cycle, or by chunked prefill of the prompt.
- A *block length* $B$ (typically 32) and a *threshold* $\tau$ (typically 0.9).

The goal of cycle $c$ is to **commit at least one more token**, with the hope of committing up to $B$.

### 2.2 Step 1 — draft with diffusion

Append $B$ `[MASK]` tokens to the input:

$$
\tilde{y}_{0:t+B} \;=\; y_{0:t} \;\Vert\; [\texttt{[MASK]}]^B.
$$

Run a *single forward pass* in **diffusion mode**, with:

- The committed prefix $y_{0:t}$ contributes clean KV-cache entries (already in HBM).
- The $B$ masked positions form a new block; their attention mask is block-diffusion: each masked position attends to (i) the committed prefix and (ii) all other positions in the same block, bidirectionally.

This is exactly the inference call from Lecture 1.5 with one block to denoise. The forward produces logits $\ell^{\text{draft}}_{t+i}$ for $i = 1, \dots, B$, and we extract the **greedy draft**:

$$
\hat{y}^{\text{draft}}_{t+i} \;=\; \arg\max_v \ell^{\text{draft}}_{t+i}[v], \qquad i = 1, \dots, B.
$$

We do *not* sample with the threshold — we always emit all $B$ draft tokens. The threshold check happens in the next step.

### 2.3 Step 2 — verify with AR

Now run a *second forward pass* in **AR mode** on the same input, but with $\tilde{y}$ replaced by the draft:

$$
y^{\text{verify}}_{0:t+B} \;=\; y_{0:t} \;\Vert\; \hat{y}^{\text{draft}}_{t+1:t+B}.
$$

The AR forward attends causally over the full $t + B$ length and produces logits $\ell^{\text{ar}}_{t+i}$ for $i = 0, \dots, B - 1$ (the standard shift-by-one convention).

Critically, the KV cache from the committed prefix can be **reused as-is** in this forward — the AR mode's clean-prefix conditioning is identical to the diffusion mode's clean-prefix conditioning (this is enforced by the dual-stream training: see Lecture 2.2's $M_{\text{OBC}}$).

The new positions $t+1 \dots t+B$ generate their own KV cache entries during the verify forward, which we keep regardless of acceptance — we'll reuse them next cycle.

### 2.4 Step 3 — accept

Walk through the draft positions left to right. Position $t+i$ is **accepted** iff:

$$
\hat{y}^{\text{draft}}_{t+i} \;=\; \arg\max_v \ell^{\text{ar}}_{t+i - 1}[v], \quad \text{and the AR top-1 probability} \geq \tau.
$$

Stop at the first rejection: commit the longest matching prefix $y_{t+1:t+m}$ where $m$ is the largest index for which all positions $t+1, \dots, t+m$ are accepted.

Update state: $t \leftarrow t + m$. If $m < B$, also commit the AR-suggested correction at position $t + m + 1$ (this is the "+1 token" of speculative decoding — see Leviathan §3 for the derivation). The committed prefix grows by $m + 1$ tokens this cycle.

### 2.5 The whole cycle in pseudo-code

```python
def linear_self_speculation_step(model, y_committed, kv_cache, B=32, tau=0.9):
    """One cycle of linear self-speculation.

    Returns:
        y_new: newly committed tokens (length 1 to B+1).
        kv_cache_new: updated KV cache (extended by len(y_new)).
    """
    t = y_committed.shape[0]

    # --- Step 1: diffusion draft -----------------------------------------------
    z_draft = torch.cat([y_committed, MASK.repeat(B)], dim=0)        # [t+B]
    model.set_attention_mode("block_diff", block_length=B)
    out_draft = model(z_draft, kv_cache=kv_cache)
    draft_logits = out_draft.logits[t:t + B]                          # [B, V]
    y_draft = draft_logits.argmax(-1)                                 # [B]

    # --- Step 2: AR verify ------------------------------------------------------
    z_verify = torch.cat([y_committed, y_draft], dim=0)               # [t+B]
    model.set_attention_mode("autoregressive")
    out_verify = model(z_verify, kv_cache=kv_cache)
    ar_logits = out_verify.logits[t - 1:t + B - 1]                    # [B, V]
    ar_top1_token = ar_logits.argmax(-1)                              # [B]
    ar_top1_prob = ar_logits.softmax(-1).gather(-1, ar_top1_token[:, None]).squeeze(-1)

    # --- Step 3: accept the longest matching, high-confidence prefix ----------
    match_mask = (y_draft == ar_top1_token) & (ar_top1_prob >= tau)   # [B]
    if match_mask.all():
        m = B
    else:
        m = match_mask.cumprod(0).sum().item()                        # first 0 = stop

    # Commit accepted + the AR's correction at position m+1.
    if m < B:
        correction = ar_top1_token[m].unsqueeze(0)                    # [1]
        y_new = torch.cat([y_draft[:m], correction], dim=0)           # [m+1]
    else:
        y_new = y_draft                                                # [B]

    # The verify KV cache is reusable; the diffusion KV cache is discarded.
    return y_new, out_verify.kv_cache
```

> **Implementation note.** The `out_verify.kv_cache` covers the *full* `t + B` positions even when only `m` tokens are accepted. NLD's actual code keeps only the first `t + m + 1` entries — the tail is fresh-but-stale. The discard is a one-line slice; the rest of the cache is byte-for-byte reusable.

### 2.6 Why this works

Three claims, each verifiable by the reader:

1. **The AR verify is correct.** The AR forward sees exactly the same clean-prefix conditioning the model was trained for. Its top-1 token at each position is the AR mode's belief about what the next token "should" be. If the diffusion draft happens to match this belief, the AR forward would have produced the same token autoregressively.

2. **The acceptance rule preserves AR distribution.** By accepting only when the draft matches the AR top-1 AND the AR confidence is high, the committed tokens are statistically identical to the tokens an AR-only pass would have produced. (Formal proof in Leviathan & Kalman §3.2; the threshold-$\tau$ variant is a strict subset of the original speculative acceptance rule.)

3. **KV-cache reuse is structurally correct.** The dual-stream training (Lecture 2.2) constrained the diffusion mode's clean-prefix attention pattern to match the AR mode's prefix attention pattern. So the KV cache the AR forward produces for positions $1, \dots, t$ is bitwise identical to the KV cache the diffusion forward would have produced for the same positions in clean mode. The cache is shareable.

The third claim is the technical bedrock of self-speculation. If the dual-stream training were sloppy — for instance, if the model accidentally learned different KV representations for "clean prefix in diffusion mode" vs "clean prefix in AR mode" — then self-speculation would silently mis-acccept tokens and the output quality would degrade. **Validating that the cached KV is reusable across modes is the single most important debugging check** when reimplementing this from scratch (see Lab 4.5).

---

## 3. The arithmetic of TPF

### 3.1 Per-cycle accounting

Each cycle costs **two forwards**:

- Draft forward: input length $t + B$, attention is block-diffusion (cost scales like $(t + B) \cdot B$ with KV cache; the $B^2$ intra-block term is the only extra).
- Verify forward: input length $t + B$, attention is causal (cost scales like $(t + B) \cdot 1$ with KV cache).

Both forwards reuse the prefix KV cache from previous cycles, so the per-cycle compute scales linearly with $B$, not with $t$. At batch 1, the wall-clock per cycle is approximately:

$$
T_{\text{cycle}} \;\approx\; T_{\text{HBM}} \cdot (1 + \epsilon_B) \;+\; T_{\text{HBM}} \;=\; T_{\text{HBM}} \cdot (2 + \epsilon_B),
$$

where $T_{\text{HBM}} \approx W / \text{HBM bandwidth}$ is the memory-bound per-forward cost (≈ 2 ms for an 8B BF16 model on a B200) and $\epsilon_B$ accounts for the slight extra attention compute in the draft forward (≈ 5–10% for $B = 32$).

So roughly $T_{\text{cycle}} \approx 4–5$ ms per cycle on B200 at batch 1.

### 3.2 Expected committed tokens per cycle

Let $\alpha$ be the per-position acceptance probability (the probability that the diffusion draft matches the AR top-1 and exceeds threshold $\tau$). Modelling the draft as i.i.d. Bernoulli($\alpha$), the expected number of accepted tokens is:

$$
\mathbb{E}[m] \;=\; \sum_{k=0}^{B-1} \alpha^k \cdot (1 - \alpha) \cdot k \;+\; \alpha^B \cdot B \;=\; \alpha \cdot \frac{1 - \alpha^B}{1 - \alpha}.
$$

Plus the AR's +1 correction at position $m + 1$, giving:

$$
\mathbb{E}[\text{tokens committed}] \;=\; \mathbb{E}[m] + 1 \;=\; \frac{1 - \alpha^B}{1 - \alpha}.
$$

At $\alpha = 0.6$, $B = 32$: $\mathbb{E} \approx 2.5$. At $\alpha = 0.8$: $\mathbb{E} \approx 5$. At $\alpha = 0.9$: $\mathbb{E} \approx 9$.

### 3.3 TPF as a function of $\alpha$ and $B$

$$
\text{TPF} \;=\; \frac{\mathbb{E}[\text{tokens committed per cycle}]}{\text{forwards per cycle}} \;=\; \frac{1}{2} \cdot \frac{1 - \alpha^B}{1 - \alpha}.
$$

Tabulated for $B = 32$:

| $\alpha$ | $\mathbb{E}[m] + 1$ | TPF |
|---|---|---|
| 0.4 | 1.67 | 0.83 |
| 0.5 | 2.0 | 1.0 |
| 0.6 | 2.5 | 1.25 |
| 0.7 | 3.3 | 1.65 |
| 0.8 | 5.0 | 2.5 |
| 0.85 | 6.7 | 3.35 |
| 0.9 | 9.5 | 4.75 |
| 0.95 | 18 | 9 |

NLD's reported TPF of ≈ 6 corresponds to $\alpha \approx 0.92$, which is consistent with the model card's claim that ~92% of draft positions exceed threshold on typical workloads.

> **Reality check.** TPF measured in practice often falls 10–20% below this i.i.d. model because acceptance events are *correlated* — long runs of high-confidence drafts cluster, and so do runs of low-confidence ones. The above formula is a useful first-order tool but not the ground truth.

### 3.4 Why $B$ has a sweet spot

Notice that TPF saturates at $\frac{1}{2(1 - \alpha)}$ as $B \to \infty$. At $\alpha = 0.9$ the saturation is 5, at $\alpha = 0.95$ it's 10. So increasing $B$ has diminishing returns once $B \gg 1 / (1 - \alpha)$.

There's also a cost penalty: $\epsilon_B \to 1$ as $B$ grows because attention becomes a meaningful fraction of the forward. Past $B = 64$, the per-cycle wall-clock starts inflating and TPF in wall-clock terms regresses.

NLD picks $B = 32$ as the wall-clock-optimal default on 8B-class models. We'll come back to this in Lecture 2.5 (speed-of-light) and Lecture 3.7 (benchmark analysis).

---

## 4. Why a LoRA helps

A subtle issue: the diffusion mode is trained on the joint loss $L_{\text{AR}} + \alpha \cdot L_{\text{diff}}$ at $\alpha = 0.5$, which means the *diffusion drafter is half-trained* relative to the AR verifier (in terms of effective gradient weight). The result is that the diffusion drafter's top-1 predictions are systematically a few percentage points lower-confidence than the AR verifier's at corresponding positions.

This shows up as a 5–10 point TPF gap: theoretical TPF (using AR-verifier's distribution as the "true" distribution) is ≈ 7, but measured TPF is ≈ 5.

NLD's fix: train a **drafter alignment LoRA** on top of the joint-trained base, with a hybrid loss that explicitly minimises the distance between the diffusion drafter's distribution and the AR verifier's distribution at the same positions.

We give the headline:

- LoRA rank: 128.
- LoRA scaling factor $\alpha$: 512 (yes, that's $\alpha / r = 4$).
- Target: only the `o_proj` of attention (other projections frozen).
- Total parameter overhead: ~36 M (< 0.5% of 8B base).
- Training: roughly 1B tokens (a tiny fraction of the base pretrain).
- Loss: LK-hybrid loss (logit-KL + hidden-state regression). Closed form in NLD §3.5.

The result is reported TPF jumping from ≈ 5 to ≈ 6 on math reasoning and ≈ 5.5 to ≈ 6.5 on code generation. The LoRA is **optional at inference** — you load it only when you want maximum self-speculation TPF. AR-only or diffusion-only modes don't need it.

We dissect the LoRA training pipeline in Lecture 3.4.

---

## 5. Quadratic self-speculation

The linear variant accepts a *single* draft per cycle. What if you could draft multiple candidates and accept the best one? That is **quadratic self-speculation**.

The intuition: at low-confidence positions, the diffusion draft might be slightly wrong. Instead of committing the AR correction (which is what linear does), spend an extra forward to generate *another* draft conditioned on the partial commit so far, and check if it has higher acceptance.

The cost is one extra forward per "redraft" call, so the per-cycle compute grows quadratically with the redraft depth (hence the name). The benefit is that $\alpha$ effectively rises, increasing TPF.

Schematically:

```
Cycle k of quadratic self-speculation:
  1. Diffusion draft (B tokens).                       (1 forward)
  2. AR verify.                                         (1 forward)
  3. Compute longest accepted prefix m.
  4. If m < B and high-confidence redraft is profitable:
     a. Diffusion draft positions [t+m+1, t+B] again.   (1 forward)
     b. AR verify the new tail.                          (1 forward)
  5. Else: commit y[t+1:t+m+1] and continue.
```

The cost depends on how often step 4 is profitable, which depends on the workload. NLD reports quadratic self-speculation pays off on:

- Long-form math reasoning, where intermediate steps are high-confidence but final-answer positions need a second look. TPF: ~8.
- Code generation, where syntax-constrained positions accept easily but identifier choices need a redraft. TPF: ~7.

And does **not** pay off on:

- Open-domain dialog, where the draft is already saturating $\alpha \approx 0.95$ at $B = 16$; redrafting cannot improve much. TPF: ~6.5 (same as linear).
- Instruction-following with strict format, where the AR distribution is so sharp that the draft rarely disagrees; no redraft needed.

The compute decision (whether to redraft at a given cycle) is itself a learned gate in NLD's implementation. We unpack the gate and the FlexAttention-based mask machinery in Lecture 3.5.

---

## 6. Comparison to two-model speculative decoding

Quick contrast (full table in Lecture 2.4):

| Property | Two-model spec (EAGLE-3 + 8B) | Self-speculation (NLD 8B) |
|---|---|---|
| Total params | ~9B | 8B (+ optional 36M LoRA) |
| HBM load per cycle | 2× (drafter + verifier) | 1× (single model, single KV cache) |
| Acceptance rate $\alpha$ | ~0.75 (drafter limited) | ~0.92 (full-capacity drafter) |
| TPF at batch 1 | ~4 | ~6 (linear), ~8 (quadratic) |
| Drafter training | Specialised per target | Free (joint training) |
| Mode flexibility | Drafter only | Three modes (AR, diff, self-spec) |
| Deployment complexity | Two checkpoints, KV management | One checkpoint, mode flag |

The headline takeaway: at batch 1, self-speculation wins on every axis. At high batch, the two-model advantages erode (HBM amortises) and the two approaches converge.

---

## 7. Common bugs in self-speculation implementations

If you reimplement this from scratch (Lab 4.5), expect to hit:

1. **KV cache mismatch.** The most common failure: the diffusion mode and AR mode produce *different* KV cache entries for the same clean prefix, so cache reuse silently corrupts the verify forward. Symptom: TPF is fine but eval scores drop ≈ 2–3 points. Detection: assert that `kv_cache_diff[:t] == kv_cache_ar[:t]` after both forwards. Fix: ensure the structured attention mask is built consistently.

2. **Off-by-one in AR logits.** The AR forward at input position $i$ predicts token $i+1$. When extracting logits for verification, slice `out_verify.logits[t-1:t+B-1]`, not `out_verify.logits[t:t+B]`. Detection: TPF drops to ~1 immediately, very obvious.

3. **Threshold applied to the wrong distribution.** The threshold $\tau$ should apply to the AR verifier's confidence, not the diffusion drafter's. If you accidentally apply it to the drafter's distribution, you get a self-consistency check that's weaker than the AR check, and quality degrades.

4. **Forgetting to discard partially-accepted tail.** When `m < B`, positions $t+m+1, \dots, t+B$ in the verify KV cache are *fresh but stale* — they were generated under an input that won't match the next cycle's prefix. You must slice the cache to length $t + m + 1$ before continuing. NLD's code explicitly does this; if you forget, the next cycle's verify forward sees an inconsistent cache and produces garbage.

All four are caught by a single end-to-end correctness test: run self-speculation against AR-only on a fixed prompt and verify the outputs are identical at $\tau = 1.0$ (where self-speculation degenerates to AR-only because nothing meets the threshold).

---

## 8. Exercises

1. Derive the expected number of accepted draft tokens $\mathbb{E}[m]$ from §3.2. Verify the limit $\mathbb{E}[m + 1] \to 1 / (1 - \alpha)$ as $B \to \infty$, and identify the value of $B$ at which $\mathbb{E}[m + 1]$ reaches 95% of this limit.

2. Suppose you have a workload where the per-position acceptance probability $\alpha_i$ varies across positions (e.g., higher at function-call boundaries, lower at identifier positions). Modify the TPF formula in §3.3 to handle this. Does the formula still saturate?

3. Write the structured attention mask for a self-speculation cycle in matrix form. Hint: it should look like two stacked block-causal masks of size $(t + B) \times (t + B)$ — one for the diffusion draft forward, one for the AR verify forward — and they should agree on the clean-prefix region.

4. Implement the linear self-speculation loop in PyTorch using only the model's forward function. Verify (i) bitwise reproducibility across runs with fixed seeds, (ii) that the output matches AR-only at $\tau = 1.0$.

5. Why does the quadratic self-speculation gate need to be learned rather than rule-based? Construct a workload where a simple rule ("redraft if $m < B / 2$") underperforms the learned gate.

Solutions to (1), (3), (5) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 9. Further reading

- **Leviathan, Kalman, Matias** (2022). *Fast Inference from Transformers via Speculative Decoding*. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192). The canonical speculative-decoding paper. Read §3 in full.
- **Chen, Borgeaud, Irving, et al.** (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318). DeepMind's parallel treatment; §3.2 has the cleanest version of the acceptance rule.
- **EAGLE-3** (Li et al., 2024). [arXiv:2503.01840](https://arxiv.org/abs/2503.01840). The current SOTA two-model spec. The drafter architecture is genuinely clever but requires per-target training; compare against NLD's "drafter = self" approach.
- **Block Diffusion** (Arriola et al., 2024). [arXiv:2503.09573](https://arxiv.org/abs/2503.09573). Their §5.3 is the first place the idea of "diffusion drafts + AR verify" appears in a unified model.
- **Nemotron-Labs-Diffusion Tech Report** (Fu et al., 2026). [PDF](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_Diffusion_Tech_Report_v1.pdf). §4 (linear self-speculation), §5 (quadratic self-speculation), Tables 4–6 (benchmarks).

Next: Lecture 2.4, where we put self-speculation in context against MTP, EAGLE, Medusa, and pure block-diffusion. After that: 2.5 — how close to the theoretical speed-of-light does NLD get on actual hardware.
