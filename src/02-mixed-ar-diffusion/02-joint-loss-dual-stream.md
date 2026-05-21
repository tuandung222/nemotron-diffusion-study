# 2.2 — Joint loss, dual-stream input, and the structured attention mask

> **Goal of this lecture.** Derive the training machinery of mixed AR-diffusion from first principles. By the end you should be able to (i) write down the joint loss in closed form, (ii) explain the dual-stream input layout and why it is necessary, and (iii) construct the full 2L × 2L structured attention mask block by block. This is the *technical* core of Series 2. Take your time.

Background assumed: Lectures 1.3 (MDLM loss), 1.4 (block diffusion + `compute_block_mask`), and 2.1 (motivation for unification). You should be comfortable reading PyTorch-style index arithmetic and reasoning about attention masks as boolean matrices.

---

## 1. The joint loss

We want a single transformer $f_\theta$ whose forward pass produces *both* a next-token AR loss *and* a masked-position diffusion loss in one go. Concretely:

$$
\mathcal{L}(\theta) \;=\; \underbrace{\mathbb{E}_{x \sim \mathcal{D}}\left[-\sum_{t=0}^{L-1} \log p_\theta(x_{t+1} \mid x_{\leq t})\right]}_{\mathcal{L}_{\text{AR}}(\theta)} \;+\; \alpha \cdot \underbrace{\mathbb{E}_{x \sim \mathcal{D}}\,\mathbb{E}_{r \sim \text{Beta}(1,1)}\,\mathbb{E}_{M \sim \text{Bernoulli}(r)^L}\left[-\sum_{t \in M} \log p_\theta\!\left(x_t \mid x_{[L]\setminus M}\right)\right]}_{\mathcal{L}_{\text{diff}}(\theta)}.
$$

Two things to notice:

- $\mathcal{L}_{\text{AR}}$ is the standard next-token loss on **clean** tokens with **causal** conditioning.
- $\mathcal{L}_{\text{diff}}$ is the MDLM simplified loss (Lecture 1.3) on a **uniformly randomly masked** version of the same sequence with **bidirectional** conditioning on the unmasked positions.

The two losses share $\theta$ but condition on **different inputs**. That is the entire complication of mixed AR-diffusion training: you need one forward pass that simultaneously gives you clean-AR conditioning *and* masked-diffusion conditioning, **without** the AR head leaking labels into the diffusion head or vice versa.

### 1.1 Why $\alpha < 1$ in practice

Why not $\alpha = 1$? Equate the two losses' gradient magnitudes naively and you get a problem: the diffusion loss is supervised on a *random subset* of tokens (the masked ones) at a random mask ratio $r$, so its per-sample variance is much higher than the AR loss's variance.

Concretely, the gradient signal-to-noise for the diffusion loss is roughly:

$$
\text{SNR}_{\text{diff}} \;\approx\; \frac{\mathbb{E}[\|\nabla_\theta \log p_\theta(x_t \mid \cdot)\|^2]}{\text{Var}(\|\nabla_\theta \mathcal{L}_{\text{diff}}\|)}
$$

and the variance is inflated by the random masking, by a factor that grows with the per-batch mask-ratio variance. Halving $\alpha$ halves the contribution of this variance term to the joint update, leaving the AR signal-to-noise dominant. NLD's empirically-best $\alpha$ is **0.5** for the dense model and **0.3** for the VLM variant (where vision tokens add a further variance source).

> **Practical note.** Don't read too much into the exact value of $\alpha$. It is the kind of hyperparameter that's tuned per-recipe and per-mask-schedule; what matters is that it is *finite and bounded away from 1*.

### 1.2 The masking schedule

The Bernoulli mask ratio $r \sim \text{Beta}(1,1)$ above is the simplest choice (uniform on $[0, 1]$). Several papers tweak this:

- **MDLM/LLaDA:** uniform $r \sim U(0, 1)$. Maximum variance.
- **D3PM:** cosine schedule, $r = \sin^2(\pi t / 2T)$ for diffusion step $t \in [0, T]$. Concentrates mass at moderate mask ratios.
- **NLD:** uniform $r \sim U(0.05, 0.95)$ with a hard floor and ceiling. The floor avoids the "no masked tokens to predict" degenerate case; the ceiling avoids the "almost everything is masked" high-variance regime.

The schedule choice matters at the 0.5-point-on-MMLU level. We treat it as an engineering knob in this lecture; see Lecture 3.3 for NLD's specific recipe.

---

## 2. The dual-stream input layout

We need *one* forward pass to produce both losses. The trick is to feed the model **two copies of the same sequence** concatenated along the sequence dimension, with different per-copy conditioning.

### 2.1 The construction

Given a sequence $x \in \{0, ..., V-1\}^L$ and a binary mask $m \in \{0, 1\}^L$ (with $m_t = 1$ meaning *masked*), define:

$$
\tilde{x} \;=\; \text{noisy\_view}(x, m) \;=\; \begin{cases} \texttt{[MASK]} & \text{if } m_t = 1 \\ x_t & \text{otherwise} \end{cases}, \qquad t = 0, ..., L-1.
$$

The **dual-stream input** is

$$
z \;=\; \underbrace{\tilde{x}}_{\text{noisy view, length } L} \;\Vert\; \underbrace{x}_{\text{clean view, length } L} \;\in\; \{0, ..., V\}^{2L}.
$$

Position indices in $z$ run from $0$ to $2L - 1$. Positions $[0, L)$ are the "noisy half" (where the diffusion loss is computed); positions $[L, 2L)$ are the "clean half" (where the AR loss is computed).

> **Why this works.** Both halves see the *same underlying tokens*, but with different per-position conditioning available via the attention mask (next section). The forward pass computes hidden states $h \in \mathbb{R}^{2L \times d}$, from which:
>
> - The diffusion logits are taken from $h_{[0:L)}$ at the masked positions.
> - The AR logits are taken from $h_{[L:2L)}$ at every position (with the standard shift-by-one).

The hidden states at the two halves are *not* the same even at unmasked positions, because each half sees a different attention pattern. We need that asymmetry — without it the two losses degenerate to the same loss.

### 2.2 Why not just two forwards?

A reasonable question: why concatenate? Why not just run two separate forwards, one with the noisy view and one with the clean view, and add the losses?

Three reasons:

1. **Activation re-use.** Within a single forward, the model's intermediate hidden states for the *clean prefix* of each block are computed once and consumed by both the diffusion attention (M_OBC, see §3) and the AR attention (M_BC). Two separate forwards would compute these activations twice.

2. **Gradient correlation.** When both losses are computed in the same forward, the gradient updates correlate at the level of shared activations. Empirically this gives slightly tighter optimisation than two-forward variants, which produce uncorrelated gradient noise.

3. **KV-cache layout.** At inference, the self-speculation mode requires that the clean-prefix KV cache from the AR-verify forward can be reused as the conditioning context for the next diffusion-draft forward. The dual-stream training layout makes this layout natural: the clean half's KV is already there.

The cost is a 2× sequence length per forward at training time, which translates to ≈ 1.7× extra wall-clock (attention is not strictly quadratic in $L$ thanks to FlashAttention; for typical $L = 4096$ the constant is closer to 1.5×). NLD considers this acceptable.

### 2.3 The vision-asymmetry sneak preview

In NLD-VLM, vision tokens *do not* live in the noisy stream. They only appear in the clean half, because vision tokens are never masked (they come from a pretrained encoder, not from the LM's vocabulary). The dual-stream becomes **asymmetric**: noisy half is text-only of length $L_t$; clean half is text + vision of length $L_t + L_v$. The structured attention mask gets correspondingly asymmetric. We defer this to Lecture 3.6.

---

## 3. The structured attention mask

This is the most important diagram in the entire Series. The 2L × 2L attention mask is composed of four quadrants, each of which is a specific block-structured submask. We'll define them one at a time and then assemble.

### 3.1 Notation

Index $z$ by $i, j \in [0, 2L)$. We say $j$ is in the **noisy half** if $j < L$, and in the **clean half** if $j \geq L$. The block size is $B$ (e.g. 32). Block index $b(t) = \lfloor t / B \rfloor$ for $t \in [0, L)$.

For each pair $(i, j)$, the attention mask $M_{ij} \in \{0, 1\}$ controls whether query $i$ may attend to key/value $j$. We use the convention $M_{ij} = 1$ means "attend allowed", $M_{ij} = 0$ means "block".

### 3.2 The four quadrants

The 2L × 2L matrix decomposes into:

$$
M \;=\; \begin{pmatrix} M_{\text{noisy} \to \text{noisy}} & M_{\text{noisy} \to \text{clean}} \\ M_{\text{clean} \to \text{noisy}} & M_{\text{clean} \to \text{clean}} \end{pmatrix} \;\in\; \{0, 1\}^{2L \times 2L}.
$$

We'll determine each quadrant by asking: "what kind of conditioning does this query need?"

### 3.3 The clean → clean quadrant ($M_{\text{BC}}$, block-causal AR mask)

The clean half is responsible for the AR loss. Each clean query $i \in [L, 2L)$ should attend to clean keys at strictly earlier positions in the *underlying* sequence, which means clean keys $j \in [L, 2L)$ with $j - L \leq i - L - 1$, i.e. $j \leq i - 1$.

This is just the standard **strict lower-triangular causal mask** on the clean half:

$$
M_{\text{BC}}[i, j] \;=\; \mathbb{1}[j < i], \qquad i, j \in [L, 2L).
$$

The AR loss is computed exactly as it would be in a vanilla decoder LM — every clean query attends only to earlier clean keys.

### 3.4 The noisy → clean quadrant ($M_{\text{OBC}}$, out-of-block clean conditioning)

Now the interesting part. The noisy half is responsible for the diffusion loss. Each noisy query $i \in [0, L)$ should attend to:

- Clean keys at positions *before its current block* (so the diffusion task is conditioned on the previously-committed history), but **not** clean keys at positions within its current block or later (else it would leak the answer it's supposed to predict).

In block-coordinates: noisy query at noisy-position $i$ (block $b(i)$) attends to clean key at noisy-position $j - L$ (block $b(j - L)$) iff $b(j - L) < b(i)$.

$$
M_{\text{OBC}}[i, j] \;=\; \mathbb{1}\!\left[\,b(j - L) < b(i)\,\right], \qquad i \in [0, L),\; j \in [L, 2L).
$$

This is a **block-strictly-lower-triangular** mask between the two halves.

> **Why "out-of-block clean"?** The name comes from NLD's code: the noisy query attends to clean keys *outside its current block*. Within its current block, it has to make do with the noisy view (next quadrant), not the clean answer.

### 3.5 The noisy → noisy quadrant ($M_{\text{BD}}$, block-diffusion intra-block mask)

Within a single block, the noisy half does *bidirectional* attention — every masked position attends to every other position in the same block (clean or masked), but only within its block:

$$
M_{\text{BD}}[i, j] \;=\; \mathbb{1}\!\left[\,b(i) = b(j)\,\right], \qquad i, j \in [0, L).
$$

This is the **block-diagonal** mask from Lecture 1.4 (`compute_block_mask`). It is *not* lower-triangular; within a block, attention is fully bidirectional.

> **Subtlety.** A masked noisy position $i$ has access to (a) other masked positions in its block, (b) unmasked positions in its block (which are clean tokens from the input), and (c) clean keys from earlier blocks (via M_OBC). The model thus has bidirectional intra-block context **plus** AR-style cross-block context. This is exactly the block-diffusion factorization from Lecture 1.4, lifted into the dual-stream layout.

### 3.6 The clean → noisy quadrant (must be zero)

The clean half should *never* attend to the noisy half. Doing so would leak masked information into the AR head's input, breaking the AR loss.

$$
M_{\text{clean} \to \text{noisy}}[i, j] \;=\; 0, \qquad i \in [L, 2L),\; j \in [0, L).
$$

This is the simplest quadrant and is also the one most commonly bugged in implementations. If you accidentally leak any nonzero entries here, the AR loss starts dropping suspiciously fast on training data (the model is cheating by reading the unmasked clean version through the noisy backdoor) and the model fails to generalize. **Always assert this quadrant is all zeros.**

### 3.7 The full mask, in one matrix

Putting it all together:

$$
M \;=\; \begin{pmatrix} M_{\text{BD}} & M_{\text{OBC}} \\ \mathbf{0}_{L \times L} & M_{\text{BC}} \end{pmatrix}.
$$

In pseudo-PyTorch:

```python
def compute_dual_stream_mask(L: int, B: int) -> torch.Tensor:
    """Build the 2L × 2L mixed AR-diffusion attention mask.

    Args:
        L: per-stream sequence length.
        B: block size (default 32 in NLD).

    Returns:
        mask: float tensor of shape (2L, 2L); 0 where allowed, -inf where blocked
              (additive softmax mask convention).
    """
    block_id = torch.arange(L) // B            # [L], values in [0, L/B)

    # 1. M_BD: block-diagonal (bidirectional intra-block).
    M_BD = (block_id[:, None] == block_id[None, :])  # [L, L]

    # 2. M_OBC: noisy → clean, block-strictly-lower-triangular.
    M_OBC = (block_id[None, :] < block_id[:, None])  # [L, L]

    # 3. M_BC: clean → clean, strict causal lower-triangular.
    M_BC = torch.tril(torch.ones(L, L, dtype=torch.bool), diagonal=-1)

    # 4. clean → noisy: all zero.
    zeros = torch.zeros(L, L, dtype=torch.bool)

    # Assemble.
    top = torch.cat([M_BD, M_OBC], dim=1)      # [L, 2L]
    bot = torch.cat([zeros, M_BC], dim=1)      # [L, 2L]
    M = torch.cat([top, bot], dim=0)           # [2L, 2L]

    # Convert to additive softmax mask.
    return torch.where(M, torch.zeros(1), torch.tensor(float("-inf")))
```

This 25-line function is the **entire** structural difference between an AR transformer and a mixed AR-diffusion transformer at training time. Everything else (the embedding table, the FFN, the LM head) is unchanged.

> **Spend a minute** matching the four quadrants to the four `cat` calls in this code, until you can re-derive the function from the diagram and the diagram from the function.

### 3.8 A worked example

Let $L = 8$, $B = 2$. The 16 × 16 mask has block_id = `[0, 0, 1, 1, 2, 2, 3, 3]`. Reading row by row (1 = attend, 0 = block):

```
Position legend:
  rows 0-7  = noisy half (n0..n7)
  rows 8-15 = clean half (c0..c7)
  cols 0-7  = noisy keys
  cols 8-15 = clean keys

         n0 n1 n2 n3 n4 n5 n6 n7   c0 c1 c2 c3 c4 c5 c6 c7
   n0  [  1  1  0  0  0  0  0  0    0  0  0  0  0  0  0  0 ]   (block 0; no clean prefix)
   n1  [  1  1  0  0  0  0  0  0    0  0  0  0  0  0  0  0 ]
   n2  [  0  0  1  1  0  0  0  0    1  1  0  0  0  0  0  0 ]   (block 1; sees clean block 0)
   n3  [  0  0  1  1  0  0  0  0    1  1  0  0  0  0  0  0 ]
   n4  [  0  0  0  0  1  1  0  0    1  1  1  1  0  0  0  0 ]   (block 2; sees clean blocks 0,1)
   n5  [  0  0  0  0  1  1  0  0    1  1  1  1  0  0  0  0 ]
   n6  [  0  0  0  0  0  0  1  1    1  1  1  1  1  1  0  0 ]   (block 3; sees clean blocks 0,1,2)
   n7  [  0  0  0  0  0  0  1  1    1  1  1  1  1  1  0  0 ]

   c0  [  0  0  0  0  0  0  0  0    0  0  0  0  0  0  0  0 ]   (AR; no prior tokens)
   c1  [  0  0  0  0  0  0  0  0    1  0  0  0  0  0  0  0 ]
   c2  [  0  0  0  0  0  0  0  0    1  1  0  0  0  0  0  0 ]
   c3  [  0  0  0  0  0  0  0  0    1  1  1  0  0  0  0  0 ]
   c4  [  0  0  0  0  0  0  0  0    1  1  1  1  0  0  0  0 ]
   c5  [  0  0  0  0  0  0  0  0    1  1  1  1  1  0  0  0 ]
   c6  [  0  0  0  0  0  0  0  0    1  1  1  1  1  1  0  0 ]
   c7  [  0  0  0  0  0  0  0  0    1  1  1  1  1  1  1  0 ]
```

Sanity checks:

- The bottom-left 8×8 quadrant is all zero (clean never attends to noisy). ✓
- The bottom-right 8×8 quadrant is strict lower-triangular (vanilla AR causal). ✓
- The top-left 8×8 quadrant is block-diagonal with 2×2 blocks (intra-block bidirectional). ✓
- The top-right 8×8 quadrant is block-strict-lower-triangular (noisy attends to earlier clean blocks). ✓

If you can read this 16×16 pattern fluently, you can read the NLD code in Series 3 without confusion.

---

## 4. Computing the losses

Given the dual-stream forward pass produces hidden states $h \in \mathbb{R}^{2L \times d}$ and logits $\ell = W_{\text{LM}} h \in \mathbb{R}^{2L \times V}$:

### 4.1 The AR loss

Take logits from the clean half, positions $[L, 2L)$, shifted by one (standard next-token):

$$
\mathcal{L}_{\text{AR}}(\theta) \;=\; -\frac{1}{L} \sum_{t=0}^{L-1} \log \mathrm{softmax}(\ell_{L + t})_{x_{t+1}}.
$$

This is identical to vanilla AR loss; the dual-stream layout is invisible to this term.

### 4.2 The diffusion loss

Take logits from the noisy half, positions $[0, L)$, **but only at masked positions** ($m_t = 1$). Following the MDLM simplified loss from Lecture 1.3, weight each masked position by $1 / r$ (inverse mask ratio) to debias:

$$
\mathcal{L}_{\text{diff}}(\theta) \;=\; -\frac{1}{|M|} \sum_{t \in M} \frac{1}{r} \cdot \log \mathrm{softmax}(\ell_t)_{x_t}.
$$

The $1/r$ factor comes from the standard MDLM derivation; without it the loss is biased toward low-mask-ratio regimes.

### 4.3 Pseudo-PyTorch

```python
def joint_loss(model, x: torch.LongTensor, B: int, alpha: float = 0.5):
    """
    x: [L] clean token ids.
    B: block size (e.g. 32).
    alpha: diffusion loss weight.
    """
    L = x.shape[0]

    # 1. Sample mask ratio and mask.
    r = torch.empty(1).uniform_(0.05, 0.95).item()
    m = (torch.rand(L) < r)                          # [L] boolean, True = masked

    # 2. Build noisy view (corrupt x at masked positions with [MASK] token).
    MASK = model.config.mask_token_id
    x_noisy = torch.where(m, torch.full_like(x, MASK), x)

    # 3. Build dual-stream input and structured attention mask.
    z = torch.cat([x_noisy, x], dim=0)               # [2L]
    M_attn = compute_dual_stream_mask(L, B)          # [2L, 2L]

    # 4. Forward pass.
    h = model(z, attention_mask=M_attn)              # [2L, d]
    logits = model.lm_head(h)                        # [2L, V]

    # 5. AR loss from clean half (positions [L, 2L)).
    clean_logits = logits[L:]                        # [L, V]
    ar_targets = x.clone()
    ar_loss = F.cross_entropy(clean_logits[:-1], ar_targets[1:])

    # 6. Diffusion loss from noisy half (positions [0, L)).
    noisy_logits = logits[:L]                        # [L, V]
    diff_loss_per_pos = F.cross_entropy(
        noisy_logits[m], x[m], reduction="mean"
    ) / r                                            # MDLM 1/r reweighting

    return ar_loss + alpha * diff_loss_per_pos
```

This 30-line function is the complete mixed AR-diffusion training step. We dissect the production NLD version (which differs only in batching and parallelism wrappers) in Lecture 3.3.

---

## 5. Sanity checks that should pass

Before you ever look at training curves, verify these:

1. **The clean → noisy quadrant is all zero.** If it isn't, your AR loss is cheating.
2. **The AR loss alone (set α = 0) equals the AR loss of a vanilla decoder LM on the same data.** If it doesn't, the dual-stream layout is interfering with the AR computation.
3. **The diffusion loss alone (set α = ∞) is monotone non-increasing across training.** If it isn't, the masking or the attention mask is wrong.
4. **At α = 0.5, the per-token cost is ≈ 1.6–1.8× the AR-only baseline.** If it's higher, you're paying for unnecessary FLOPs; if it's lower, you're missing some attention coverage.
5. **Removing M_OBC drops diffusion accuracy by 5–10 points on math evals.** This confirms that the cross-block clean conditioning is what makes block diffusion work at long context.

Run all five before claiming "the mixed model works".

---

## 6. Exercises

1. Re-derive the four quadrants of M for a sequence where the *first* B tokens are a prompt (clean, never masked). Hint: the prompt acts as block −1, accessible to every block's noisy half but never the target of a loss.

2. Suppose you replace M_OBC with the full lower-triangular mask (i.e. allow noisy queries to attend to *all* clean keys at strictly earlier positions, not just earlier blocks). Why does this leak labels at block boundaries? Show with a 4-token example.

3. The dual-stream input has length 2L. Modify `compute_dual_stream_mask` to handle the asymmetric VLM case where the clean half has length $L + V$ (with $V$ vision tokens at the front) and the noisy half has length $L$. The vision tokens should be attendable from *both* halves.

4. Estimate the FLOP overhead of dual-stream training vs AR-only training as a function of $L$ and the FlashAttention block size. At what $L$ does the overhead exceed 2×?

5. The MDLM loss uses a $1/r$ reweighting at masked positions. Derive this from first principles by computing the expected loss under the random masking distribution and showing that the reweighted estimator is unbiased.

Solutions to (1), (3), (5) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 7. Further reading

- **Block Diffusion** (Arriola et al., 2024). [arXiv:2503.09573](https://arxiv.org/abs/2503.09573). §3 derives the block factorization; their Algorithm 1 is the cleanest published statement of the dual-stream training loop.
- **MDLM: Simple and Effective Masked Diffusion Language Models** (Sahoo et al., 2024). [arXiv:2406.07524](https://arxiv.org/abs/2406.07524). §3.2 derives the $1/r$ reweighting.
- **SDAR** (2025). [arXiv:2510.06303](https://arxiv.org/abs/2510.06303). §3 has a particularly clear presentation of the dual-stream mask, with a different but equivalent notation.
- **FlashAttention-3** (Dao et al., 2024). [arXiv:2407.08608](https://arxiv.org/abs/2407.08608). §4 explains how to efficiently realize structured masks of this shape on H100/B200.
- **Nemotron-Labs-Diffusion Tech Report** (Fu et al., 2026). Figure 3 is the canonical visual reference for the 2L × 2L mask.

Next: Lecture 2.3, where we exploit this training machinery to do self-speculation at inference.
