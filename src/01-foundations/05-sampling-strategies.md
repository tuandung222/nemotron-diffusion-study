# 1.5 Sampling strategies for diffusion LMs

> **Goal of this lecture.** The training loss is solved (it's MDLM, lecture 1.3). The block factorization is solved (lecture 1.4). What's left to determine real-world TPF is the **sampler**, the rule that decides which masked positions to commit at each denoising step. This lecture surveys the main families of samplers, derives their TPF/quality trade-offs, and lands on the **confidence-threshold sampler** that NLD ships by default. It closes with the **speed-of-light** analysis that gives an upper bound on what any sampler can achieve.

By the end you should be able to:

1. Implement the three canonical samplers (top-$k$, threshold, ancestral) in ~20 lines of PyTorch each.
2. Choose between them given a quality/speed target.
3. Compute the speed-of-light TPF for a given task and use it as a benchmark for how close your sampler is to optimal.

---

## 1. What the sampler actually decides

Recall the inference loop from lecture 1.4 (block-diffusion mode, $K_B$ steps per block of size $B$):

```python
for k_iter in range(K_per_block):
    logits = model(xt)                          # (B, V) for the current block
    commit_set = sampler(xt, logits, ...)       # which positions to unmask
    xt[commit_set] = argmax_or_sample(logits[commit_set])
    if all_committed(xt[block]): break
```

The model produces logits over the vocabulary at every masked position. The sampler picks (a) **which positions to commit** and (b) **what to put there** (argmax, or sampled from the categorical).

The sampler's choice space:

1. **Per-step commit count:** how many positions to unmask per forward pass. Constant ("top-$k$ unmask"), data-dependent ("commit all positions with confidence above $\tau$"), or learned ("predicted by an auxiliary head").
2. **Selection criterion:** which positions to commit. Top-confidence is the obvious choice; alternatives include random ("ancestral"), highest-margin, or learned.
3. **Token choice:** at committed positions, argmax (greedy) or sample (with optional temperature).
4. **Remasking:** can a previously committed token be re-masked in a later step if the model becomes uncertain about it? (Most samplers say no; some say yes.)

The choices interact: a sampler that commits many positions per step at low confidence will be fast but may make local mistakes; a sampler that commits one position per step at peak confidence will be slow but the per-token quality approaches the diffusion model's marginal-best.

---

## 2. The three canonical samplers

### 2.1 Top-$k$ unmask (fixed parallelism)

The simplest sampler: at every iteration, commit the top $k$ positions by confidence. Pseudocode:

```python
def topk_unmask_sampler(xt, logits, k):
    # logits has shape (L, V). xt has shape (L,) with MASK at unfilled positions.
    masked = (xt == MASK)
    probs = softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values     # (L,)
    confidence[~masked] = -inf                # only consider masked positions
    top_indices = confidence.topk(k).indices
    return set(top_indices.tolist())
```

Properties:

- **Deterministic TPF**: exactly $k$ per step. So $K_B = B / k$ for a block of size $B$.
- **Quality varies with $k$**: small $k$ → high quality (each step commits the most-confident position), large $k$ → faster but with more risk of committing borderline positions.
- **Common defaults**: $k = 1$ recovers single-position-per-step (essentially AR-like). $k = B$ recovers single-forward-per-block (very fast, lower quality).

This sampler is what early MDM papers used (CMLM, Mask-Predict). NLD does not use this by default; it uses (2.2).

### 2.2 Confidence-threshold sampler (NLD default)

Commit every position whose top-1 confidence exceeds a threshold $\tau$:

```python
def threshold_sampler(xt, logits, threshold=0.9):
    masked = (xt == MASK)
    probs = softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
    commit = masked & (confidence >= threshold)
    if commit.sum() == 0:
        # if nothing meets the threshold, fall back to committing the single most-confident
        confidence[~masked] = -inf
        commit[confidence.argmax()] = True
    return commit.nonzero(as_tuple=True)[0].tolist()
```

Properties:

- **Data-dependent TPF**: number of commits per step depends on how confident the model is. On easy positions (formatted output, repetitive structure) many positions clear the threshold; on hard positions (creative reasoning, new entities) only a few do.
- **Quality near AR**: commits only happen at high confidence, so the per-step quality is at the model's most-confident output, which is exactly what you'd get from AR's argmax at that position.
- **Threshold $\tau$**: the single most important hyperparameter. $\tau = 0.5$ commits aggressively (high TPF, some risk); $\tau = 0.9$ commits conservatively (lower TPF, near-AR quality); $\tau = 0.95$ is borderline.

> **NLD default**: $\tau = 0.9$. Their measured TPF in diffusion-only mode is 2.57, meaning on average 2.57 positions per forward exceed 0.9 confidence. The remaining $\sim$ 30 positions in a $B = 32$ block take more steps.

The fallback in the code ("if nothing meets the threshold, commit the most-confident") is essential, without it the loop hangs on rare hard-prediction blocks.

This is the sampler we'll assume by default for the rest of the book.

### 2.3 Ancestral sampling

Instead of using argmax, sample from the categorical at the committed positions, with optional temperature $T$:

```python
def ancestral_sampler(xt, logits, T=1.0, k=1):
    masked = (xt == MASK)
    probs = softmax(logits / T, dim=-1)
    confidence = probs.max(dim=-1).values
    confidence[~masked] = -inf
    top_indices = confidence.topk(k).indices
    # sample (not argmax) at the chosen positions
    out_tokens = torch.multinomial(probs[top_indices], num_samples=1).squeeze(-1)
    return top_indices.tolist(), out_tokens.tolist()
```

Use when you want diversity (creative writing, chat with high temperature). The $T = 0$ limit recovers argmax (= top-$k$ unmask).

**Important:** ancestral sampling does *not* by itself increase TPF. It changes *which* token gets committed at a position, not *how many* positions are committed per step. To get speed, combine it with top-$k$ or threshold.

---

## 3. Remasking: when commitments can be undone

All the samplers above are *monotone*: once a position is committed, it stays committed. This is the simplest case and matches the absorbing-state forward process (where the forward direction can only mask, never unmask).

But the *reverse* process doesn't have to be monotone, the variational posterior $q(x_{t-1} \mid x_t, x_0)$ at intermediate $t$ has nonzero probability of putting `[MASK]` back at a position that was previously visible (revisit lecture 1.2 §4). In practice, allowing the sampler to **remask** a low-confidence commit at a later step can improve quality:

```python
def threshold_sampler_with_remasking(xt, logits, threshold=0.9, remask_threshold=0.3):
    masked = (xt == MASK)
    probs = softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
    # commit new positions that clear the high threshold
    new_commits = masked & (confidence >= threshold)
    # remask positions that have *dropped* below the low threshold
    # (note: confidence here is recomputed for already-committed positions)
    drops = (~masked) & (confidence < remask_threshold)
    return new_commits, drops  # (positions to commit, positions to remask)
```

Trade-off:

- **Pro**: when the model later sees more context and realizes an earlier commit was wrong, it can fix it.
- **Con**: more iterations on average, lower TPF.

Most production diffusion LMs do **not** remask (NLD does not in its default samplers). The accuracy gain is typically <1 point on benchmarks while the TPF cost is 20–30%.

Remasking matters more when you're sampling at high diversity (ancestral with $T \gg 1$) because the early samples are noisier and more likely to be wrong.

---

## 4. Learned samplers

The samplers above are hand-designed. A natural extension: train a small auxiliary network to predict, given the current state $(xt, \text{logits})$, *which positions to commit*. This is the **Generator-Sampler** decomposition pursued by Lou et al. (SEDD) and several follow-ups.

In principle a learned sampler can be optimal, it can implicitly learn task-specific signals (e.g., "for math, commit the final answer position last", "for code, commit syntax tokens first").

In practice the gains are modest (1–2 TPF points over the confidence-threshold sampler) and the engineering cost is significant. NLD does not use a learned sampler at inference; **its analog of a learned sampler is the AR head**, which is involved in self-speculation (Series 2) to verify diffusion drafts. That's a fundamentally different mechanism and arguably better than a learned sampler in the same architecture.

We'll mention learned samplers in Series 2 where they're a natural alternative design.

---

## 5. The speed-of-light (SOL) analysis

This is the conceptual centerpiece of the lecture. The question: **what's the theoretical upper bound on TPF for a diffusion LM, regardless of sampler?**

### 5.1 Defining SOL

For a given target sequence $\mathbf{y}^*$ and a perfectly-trained model $p_\theta$, the speed-of-light TPF is the *largest* expected number of positions per forward pass that can be committed *while still recovering $\mathbf{y}^*$*. Formally:

$$
\text{SOL-TPF}(\mathbf{y}^*) \;=\; \mathbb{E}\!\left[\frac{B}{K^*_B}\right],
$$

where $K^*_B$ is the minimum number of forward passes such that there exists a sampler that commits the correct tokens of $\mathbf{y}^*$ in $K^*_B$ steps.

Equivalently: $K^*_B$ is the number of steps needed when the sampler is **clairvoyant**, it always knows whether a position's current top-1 prediction equals $y^*_i$, and commits accordingly.

### 5.2 Computing SOL empirically

Given a held-out test set:

```python
def compute_sol_tpf(model, examples, B=32, K_max=32):
    total_commits = 0
    total_forwards = 0
    for prompt, y_star in examples:
        xt = init_block_state(prompt, len(y_star))
        for k_iter in range(K_max):
            logits = model(xt)
            # "clairvoyant" commit: any position whose argmax matches y_star
            argmax_t = logits.argmax(dim=-1)
            commit = (xt == MASK) & (argmax_t == y_star)
            if commit.sum() == 0:
                # if NO position can be correctly committed greedily, give up, would need beam
                break
            xt[commit] = y_star[commit]
            total_commits += commit.sum()
            total_forwards += 1
    return total_commits / total_forwards
```

The SOL is the highest possible TPF achievable by a *correct* greedy sampler. Any real sampler will get a smaller TPF (because it doesn't know $y^*$).

### 5.3 What does SOL tell us?

From the NLD paper (Figure 7-ish), SOL for the 8B model on standard reasoning benchmarks is in the range **5–8 TPF**. The threshold-0.9 sampler achieves 2.5–3 TPF. Self-speculation gets to 5–6.

Interpretation:

1. **Hand-designed samplers leave ~50% on the table.** A clairvoyant sampler would commit 2× as many tokens per forward. Closing this gap is the active frontier.
2. **Self-speculation closes most of the gap.** By using the AR head as a verifier (Series 2), NLD effectively turns the diffusion draft into a "near-clairvoyant" sampler, it commits the tokens that the AR head would have committed, which is a strong proxy for "correct".
3. **The SOL itself is upper-bounded by the model's capacity.** A more accurate model has higher SOL. There is no universal SOL; it's a per-model, per-distribution quantity.

The SOL framework is also how you decide whether to add a sampler tweak: if your current sampler is 90% of SOL, further engineering on it is wasted; if it's 50% of SOL, there's room. The threshold-0.9 default is at roughly 50–60% of SOL for NLD-8B, which is why self-speculation is worth the engineering cost.

> **Methodological warning.** SOL as defined requires access to $\mathbf{y}^*$. It is *not* an upper bound for open-ended generation; it's an upper bound for *replicating a specific target*. For open-ended generation, the relevant question is "TPF at fixed quality" which is what the head-to-head benchmarks in lecture 1.6 measure.

---

## 6. Sampler choice in production

A summary of the practical recipe for production diffusion-LM serving:

1. **Use threshold sampler by default.** $\tau$ in [0.85, 0.95]. NLD ships 0.9.
2. **Add a fallback rule** to commit the most-confident position when nothing clears $\tau$, otherwise the loop can stall.
3. **No remasking by default.** Adds complexity, minor quality win.
4. **For diverse generation, use ancestral with temperature** at the commit step, not at the threshold step. (Threshold uses argmax confidence; sampling uses ancestral.)
5. **For maximum TPF, switch to self-speculation (Series 2)**, this is a fundamentally better mechanism than tuning a sampler.

The NLD paper has an ablation (their Fig. 6) sweeping $\tau \in [0.5, 0.99]$. At $\tau = 0.5$, TPF is ~5 but accuracy drops 5+ points on reasoning. At $\tau = 0.95$, TPF is ~2.2 but accuracy is full. $\tau = 0.9$ is the empirical sweet spot.

---

## 7. Where this fits in the bigger picture

We've now seen:

- **Loss** (lecture 1.3): MDLM with $1/t$ reweighting.
- **Architecture** (lecture 1.4): block-diffusion attention mask.
- **Inference algorithm** (this lecture): confidence-threshold sampler, ~2.5–3 TPF in production.

To go beyond 3 TPF *without* sacrificing quality you need **either** a smarter sampler (learned) or a smarter *verifier* (the self-speculation route NLD takes). We'll see in Series 2 that self-speculation, by reusing the same backbone in AR mode as a verifier, achieves 5–6 TPF with negligible quality loss, essentially closing the SOL gap.

Before that, lecture 1.6 puts AR and diffusion side by side: where each wins, where each loses, and how to read the comparison tables.

---

## Further reading

- Ghazvininejad, M., Levy, O., Liu, Y., Zettlemoyer, L. *Mask-Predict: Parallel Decoding of Conditional Masked Language Models*. EMNLP 2019. The original confidence-threshold/iterative-refinement paper for translation.
- Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W. *MaskGIT: Masked Generative Image Transformer*. CVPR 2022. Same idea applied to image VQ tokens; introduces the standard cosine remasking schedule.
- Zheng, K., Chen, Y., Mao, H., Liu, M.-Y., Zhu, J., Zhang, Q. *Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inactive Iterations*. ICLR 2025. Important, shows that MDM samplers can stall in "inactive iterations" where no commits happen; motivates the threshold fallback.
- Nemotron-Labs-Diffusion technical report, §4. Their ablation of $\tau$ and the SOL discussion.
- Geng, H., Stenetorp, P. *On the Theoretical Limits of Masked Diffusion Sampling*. 2025. The cleanest derivation of SOL bounds.

## Exercises

1. **Implement all three samplers.** Threshold, top-$k$, ancestral. Test them on a toy diffusion LM (you can use the Lab 2 notebook in Series 4 once it's written, or a small custom model). Verify that with the same number of forward passes you get different TPFs and (qualitatively) different outputs.

2. **TPF distribution under threshold sampling.** For a fixed prompt and a fixed model, run threshold sampling with $\tau = 0.9$ over 1000 samples (varying seed). Plot the histogram of (a) per-block $K_B$, (b) total NFE. Show empirically that the TPF distribution is *not* tight, there's significant variance, which has implications for latency SLOs.

3. **Computing SOL.** For a Lab notebook's tiny diffusion LM, compute SOL-TPF on a held-out subset of the training data using the procedure in §5.2. How does SOL compare to the threshold sampler's measured TPF? What's the "sampler optimality gap"?

4. **Threshold sweep.** Reproduce the NLD Fig. 6 ablation in miniature: sweep $\tau \in \{0.5, 0.7, 0.8, 0.85, 0.9, 0.95\}$ on your small model and plot quality (e.g., perplexity or BLEU) vs measured TPF. Identify the elbow of the curve.

5. **Remasking ablation.** Implement the threshold-with-remasking sampler. Measure whether it improves a hard reasoning task on your tiny model. (Hint: it probably won't on a small model; the gap shows up at larger scale.)

Sketched solutions in [Appendix B](../appendix/reading-list.md#exercise-solutions).
