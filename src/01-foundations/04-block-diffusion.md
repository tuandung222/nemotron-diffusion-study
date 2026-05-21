# 1.4 - Block diffusion: from full-sequence to block-wise denoising

> **Goal of this lecture.** Understand why "diffuse the whole sequence" - the LLaDA/MDLM default - is the wrong default for long-context generation, and how **block diffusion** (Arriola et al., 2024; SDAR, 2025) restores KV-cache reuse, enables left-to-right streaming, and is the immediate predecessor of NLD's training/inference setup.

By the end you should be able to explain three things to a colleague:

1. **The KV-cache problem with full-sequence diffusion** - why it asymptotically wastes the same compute that an AR LM saves.
2. **The block-diffusion factorization** - how splitting the sequence into causal *blocks* of bidirectional content gives you the best of both modes.
3. **The block-mask machinery** - concretely, what `compute_block_mask` (NLD's function) constructs, and why it's the natural intersection of AR's causal mask and MDM's full bidirectional mask.

This lecture is *the* prerequisite for understanding Series 2 (the dual-stream attention machinery) and Series 3 (NLD code internals). Take your time.

---

## 1. The problem: full-sequence diffusion is wasteful

In the LLaDA/MDLM inference loop, we initialize the *entire* completion as `[MASK]` and refine it in $K$ steps:

```
xt = concat(prompt, [MASK] * L_out)
for step in range(K):
    logits = model(xt)
    xt = commit_some_positions(xt, logits)
```

Two costs:

1. **Every forward pass attends over the full sequence** of length $L_p + L_{\text{out}}$. Quadratic in $L_{\text{out}}$.
2. **No KV-cache reuse.** Between iterations, *some positions get committed* and become "clean", but the *values* of the clean positions are not the same as they were the iteration before (they changed from `[MASK]` to a real token). So the K/V projections of those positions change, and we cannot reuse them.

In the AR baseline, each new token only causes $O(L)$ new work (one new query attending over $L$ cached keys), so generating an entire sequence is $O(L^2)$ total. With full-sequence diffusion at $K$ iterations, the cost is $K \cdot O(L^2)$ - strictly *worse* than AR per token.

The only thing saving full-sequence diffusion is that $K \ll L$ in practice. With LLaDA at TPF $\approx 4$ and a 1024-token completion, $K \approx 256$; AR would do 1024 forward passes, but each one is cheaper because of the KV cache. The arithmetic doesn't obviously favor either approach until you measure both wall clock and quality.

> **Practical sign that this is a problem:** if you read papers reporting diffusion-LM speeds, the largest claimed speedups are at short completion lengths (e.g. ≤ 256 tokens). At long context (4k+) the numbers degrade. That's the KV-cache problem biting.

---

## 2. The fix: factor the sequence into blocks

Suppose we split the completion into contiguous **blocks** of size $B$ (e.g., $B = 32$):

$$
\mathbf{y} \;=\; \underbrace{(y_1, \dots, y_B)}_{\text{block } 1} \;\Vert\; \underbrace{(y_{B+1}, \dots, y_{2B})}_{\text{block } 2} \;\Vert\; \dots \;\Vert\; \underbrace{(y_{(M-1)B+1}, \dots, y_{MB})}_{\text{block } M}.
$$

The **block diffusion factorization** says:

$$
p_\theta(\mathbf{y} \mid \mathbf{x}) \;=\; \prod_{m=1}^M p_\theta\!\left(\mathbf{y}_{(m)} \;\Big|\; \mathbf{x},\, \mathbf{y}_{(1)}, \dots, \mathbf{y}_{(m-1)}\right),
$$

where each block $\mathbf{y}_{(m)}$ is generated **via a small diffusion process conditioned on the prompt and all previous, already-committed blocks**.

In words:

- **Between blocks**, the model is autoregressive (each block conditions only on past blocks, never future).
- **Within a block**, the model uses full bidirectional / diffusion-style denoising to generate that block's tokens jointly.

This is the **block-diffusion factorization**, due to Arriola, Sahoo, Gokaslan, Yu, et al. ("Block Diffusion: Interpolating between Autoregressive and Diffusion Language Models", ICLR 2025).

### 2.1 The arithmetic now works

Per block:

- Forward pass attends over (prompt + committed prefix + current block) - length grows by $B$ between blocks, not by 1.
- KV cache of the prompt and all *previously committed* blocks **is reusable** because those positions are stable across the block's diffusion steps.
- Number of forward passes within a block: $K_B \ll B$ (say $K_B = 8$ to generate a 32-token block).

Total cost for generating $L_{\text{out}}$ tokens in $M = L_{\text{out}} / B$ blocks:

$$
\text{cost} \;\approx\; \sum_{m=1}^M K_B \cdot (L_p + (m-1) B + B) \;=\; K_B \cdot \left( M L_p + \frac{M(M+1)}{2} B \right).
$$

Compare to AR: $L_{\text{out}} \cdot L_p + \frac{L_{\text{out}}(L_{\text{out}}+1)}{2}$ (the second term is the quadratic over the completion). Plug in $B = 32, K_B = 8, L_{\text{out}} = 1024$ and the block-diffusion cost is *less than half* the AR cost - and you've committed many tokens per forward pass.

### 2.2 The TPF arithmetic

TPF for block diffusion is *per block*, not per forward:

$$
\text{TPF}_{\text{block}} \;=\; \frac{B}{K_B}.
$$

With $B = 32$ and $K_B = 8$ ⇒ TPF = 4. With confidence-based sampling that commits multiple positions per step, NLD measures TPF in the 2.5–3 range for diffusion-only inference, and up to 6 for self-speculation.

The key observation: **block diffusion's TPF does not depend on $L_{\text{out}}$**. You can generate 4k or 16k tokens and the TPF stays the same. That's the property AR has too. Full-sequence diffusion, by contrast, has TPF that depends on the total length (because $K$ usually scales with $L$).

---

## 3. The block-diffusion attention mask

The conceptual claim is "AR between blocks, bidirectional within a block". How is that realized as an attention mask?

Let positions be indexed $1, 2, \dots, L$, partitioned into blocks of size $B$. Define `block_id(i) = ⌈i/B⌉`. The block-diffusion attention mask $M_{\text{BD}}[i, j]$ is:

$$
M_{\text{BD}}[i, j] \;=\;
\begin{cases}
1 & \text{if } \texttt{block\_id}(j) < \texttt{block\_id}(i), \\
1 & \text{if } \texttt{block\_id}(j) = \texttt{block\_id}(i), \\
0 & \text{otherwise.}
\end{cases}
$$

Visually, for $L = 12$ and $B = 3$:

```
       j: 1 2 3 4 5 6 7 8 9 10 11 12
i= 1     [ 1 1 1 0 0 0 0 0 0 0  0  0 ]   ┐
i= 2     [ 1 1 1 0 0 0 0 0 0 0  0  0 ]   ├ block 1: bidir within, nothing future
i= 3     [ 1 1 1 0 0 0 0 0 0 0  0  0 ]   ┘
i= 4     [ 1 1 1 1 1 1 0 0 0 0  0  0 ]   ┐
i= 5     [ 1 1 1 1 1 1 0 0 0 0  0  0 ]   ├ block 2: bidir within, can see block 1
i= 6     [ 1 1 1 1 1 1 0 0 0 0  0  0 ]   ┘
i= 7     [ 1 1 1 1 1 1 1 1 1 0  0  0 ]   ┐ block 3: bidir within, can see blocks 1-2
...
```

This is precisely a **block-lower-triangular** mask: bidirectional inside each diagonal block, fully visible to all earlier blocks, invisible to all later blocks. It's the natural interpolant between AR (each row sees only $\le i$) and full bidirectional (each row sees all).

### 3.1 NLD's `compute_block_mask` function

NVIDIA's reference implementation [exposes this directly](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B/blob/main/modeling_nemotron_labs_diffusion_vlm.py). The function `compute_block_mask(L, block_size)` produces exactly the matrix above. Here's a faithful Python implementation:

```python
def compute_block_mask(L: int, block_size: int) -> torch.Tensor:
    """
    Block-diffusion attention mask.
    Returns (L, L) bool tensor: M[i, j] = True iff position j is attendable from i.
    """
    # Bucket each position into its block.
    block_id = torch.arange(L) // block_size  # shape (L,)
    # j_block < i_block  -> attend  (j is in an earlier block, fully visible)
    # j_block == i_block -> attend  (j is in the same block, bidirectional)
    # j_block >  i_block -> mask    (j is in a later block, hidden)
    return block_id.unsqueeze(1) >= block_id.unsqueeze(0)  # (L, L)
```

You can compose this with the standard `attention_mask` (the one that ignores padding) by an elementwise `&`. The result is what every NLD forward pass uses during diffusion training.

### 3.2 What if block_size = 1?

If $B = 1$, every "block" is a single position, the matrix collapses to **standard lower-triangular** = AR causal mask. Block diffusion at $B = 1$ is AR.

If $B = L$ (one block spanning the whole sequence), the matrix is **all ones above and at the diagonal** = full bidirectional. Block diffusion at $B = L$ is LLaDA / MDLM.

**Block diffusion smoothly interpolates between AR and MDM.** This is the title of Arriola's paper, and it's the cleanest formal statement of what block diffusion is.

### 3.3 What block size should you use?

In practice $B$ is a small integer chosen to balance:

- **TPF**: larger $B$ → more positions can be denoised in parallel within the block → higher TPF (good).
- **Quality**: larger $B$ → more positions have to be denoised jointly with weaker conditioning (each one sees only the rest of the block, not the right-context outside) → quality may degrade for very large blocks.
- **Memory**: larger $B$ → wider attention within the block, but the KV cache for past blocks is the same.
- **Convergence at training time**: very small $B$ collapses toward AR (slow convergence in tokens-per-forward, fast in quality); very large $B$ collapses to MDM and inherits its variance issues.

The Arriola paper (Fig. 5) does an ablation: $B = 4$ to $B = 32$ are all reasonable; $B = 32$ gives the best TPF/quality trade-off. **NLD uses $B = 32$ as its default**, matching the Arriola finding. NLD config: `"block_size": 32`.

---

## 4. Training with block diffusion

You can adapt the MDLM training loop to block diffusion in two ways:

### Option A - Per-block independent noise

For each example:

1. Pick a block index $m$ to denoise (uniformly).
2. Mask Bernoulli($t$) of positions *within block $m$*, leaving all other blocks untouched (i.e., their original $x_0$ values).
3. Forward through the model with the block-diffusion attention mask.
4. Compute MDLM-style loss on the masked positions of block $m$.

This precisely matches the factorization $p(\mathbf{y}) = \prod_m p(\mathbf{y}_{(m)} \mid \mathbf{y}_{(<m)})$ - each training example only conditions one block on the rest.

### Option B - All-blocks-simultaneously noise

For each example:

1. Pick a noise level $t \sim \mathcal{U}[0, 1]$.
2. Mask Bernoulli($t$) of positions *across all blocks*, independently per position.
3. Forward through the model with the block-diffusion attention mask (which ensures each masked position only attends within its block and to earlier blocks).
4. Compute MDLM-style loss on *all* masked positions.

Option B is more efficient (uses every position as training signal) but Option A is more faithful to the factorization. **NLD uses Option B**, with the same $1/t$ reweighting as MDLM, and applies it across all blocks simultaneously in one forward.

### 4.1 DP-rank varying masking

There's a subtle wrinkle that NLD adds: **across data-parallel ranks, the masking ratio $t$ is varied per rank**. The intuition: at any given training step, different GPUs in the data-parallel group are training the model on different points in the diffusion noise distribution. This decorrelates gradients across ranks (which is the whole point of data parallelism with averaged gradients) more effectively than if every rank used the same $t$.

In code:

```python
# At each training step, each DP rank draws its own t:
t = sample_t_for_rank(rank=dist.get_rank(), step=global_step)
```

This is a small empirical trick - Series 2 has the detailed treatment when we get to global loss averaging. For now just note it exists.

---

## 5. Inference with block diffusion

The inference loop is now:

```python
def block_diffusion_generate(model, prompt, L_out, B, K_per_block, sampler, threshold):
    L_p = len(prompt)
    M = (L_out + B - 1) // B           # number of blocks
    xt = list(prompt) + [MASK] * L_out
    kv_cache = init_cache()

    for m in range(M):
        block_start = L_p + m * B
        block_end = min(block_start + B, L_p + L_out)

        # We only re-process positions from block_start onward in subsequent steps;
        # everything before block_start is in the KV cache.
        for k_iter in range(K_per_block):
            logits = model(xt, kv_cache=kv_cache, new_positions=range(block_start, len(xt)))
            commit_set = sampler(xt, logits, threshold=threshold)
            for i in commit_set:
                xt[i] = argmax(logits[i])
            # If all positions in current block are committed, break early
            if all(xt[i] != MASK for i in range(block_start, block_end)):
                break

        # Block m is done - flush its KVs to the cache and move on
        kv_cache.commit_block(block_start, block_end)

    return xt[L_p:]
```

Key points:

1. **KV cache is reused across blocks** - only the *current* block's positions get reprocessed in each iteration.
2. **Early stopping per block** - if the sampler commits all positions in a block before $K_{\text{per block}}$ steps, the loop exits.
3. **Sampler is parameterizable** - confidence threshold, ancestral sampling, learned sampler, all plug in here. Lecture 1.5.

This pattern is the basis of every modern block-diffusion LM serving stack (NLD, SDAR, Dream-7B). vLLM's diffusion-LM support implements approximately this loop with paged attention for the KV cache.

---

## 6. Block diffusion as a stepping stone to NLD

NLD is, at a high level: **block diffusion + AR coupling + self-speculation**.

That is:

- Block diffusion contributes: the block-causal attention mask, the per-block diffusion loss, the KV-cache-friendly inference pattern.
- AR coupling (Series 2) contributes: the dual-stream input layout, the additional AR loss, the ability to switch modes at inference.
- Self-speculation (Series 2) contributes: the drafter/verifier mechanism on top of the shared backbone.

Every one of those layers is purely additive - you can ablate it and recover the previous-tier model.

**Conceptual NLD = AR + α · block-diffusion-LLaDA.** Same backbone, two losses (joint), three inference modes (mask choice).

If you understand this lecture you've already understood the diffusion component of NLD. The rest is plumbing.

---

## 7. What we now have

- **The block-diffusion factorization**: AR between blocks, diffusion within a block.
- **The block-diffusion attention mask** `compute_block_mask`, which is block-lower-triangular.
- **The training loop adaptation**: same as MDLM, just with the block mask.
- **The inference loop adaptation**: KV-cache reuse across blocks, $K_B$ steps per block.
- **The intuition for $B$**: $B = 1$ → AR; $B = L$ → MDM/LLaDA; $B = 32$ → the empirical sweet spot.

What remains in Series 1: **samplers** (lecture 1.5) and the **head-to-head with AR** (lecture 1.6). After that, Series 2 picks up with how to add the AR side and self-speculation on top of block diffusion.

---

## Further reading (primary sources)

- Arriola, M., Sahoo, S., Gokaslan, A., Yu, Z., Marroquin, E., Rush, A., Schiff, Y., Chiu, J. T., Kuleshov, V. *Block Diffusion: Interpolating between Autoregressive and Diffusion Language Models*. ICLR 2025. **The block-diffusion paper.** §3 has the factorization, §4 the attention mask, §5 the ablations.
- SDAR (paper + code), 2025. Production-grade block-diffusion training; the recipe that NLD's data-parallel masking is borrowed from.
- Han, S., Pasunuru, R., Iyer, S., et al. *Dream-7B: A Diffusion-Native Language Model*. 2025. Alternative 7B-scale block-diffusion implementation; useful comparison point for NLD.
- Yang, Q., Mei, K., Gao, Y., et al. *SDAR: A Self-supervised Diffusion Autoregressive Language Model*. 2025. Specifically explains the block size 32 choice and the diffusion-to-AR ratio sweep.

## Exercises

1. **Implement `compute_block_mask`.** Write the function from scratch (no peeking at the NLD source). Verify by printing the $12 \times 12$ matrix for $L = 12, B = 3$ and comparing to the visualization in §3.

2. **TPF as a function of block size.** Suppose the per-block early-stopping mean is $\bar K_B = a + b \cdot B$ for constants $a > 0, b \in (0, 1)$. Derive TPF as a function of $B$, find the $B$ that maximizes TPF, and interpret. (Concrete values to try: $a = 4, b = 0.2$.)

3. **Why is block diffusion better than running MDM on chunks?** Compare block diffusion (one model, block-causal mask) to a naive baseline: run MDM separately on each 32-token chunk, conditioning only on a fixed-length prefix. Identify two specific quality reasons block diffusion is better.

4. **Implement Option B (all-blocks-simultaneously) training**. Write the training step in PyTorch for a 6-layer GPT-style model with $L = 256, B = 32$. The function should: sample $t$, mask Bernoulli($t$) positions, build the block-causal attention mask, forward, compute MDLM loss. About 40 lines.

5. **KV cache invariance check.** Argue analytically that, in block-diffusion inference, the K/V of a position $p$ in block $m$ is invariant once block $m$ is fully committed. *(Hint: think about which inputs the K/V projection at position $p$ depends on, and which of those change between iterations within block $m$ vs. between blocks.)*

Sketched solutions in [Appendix B](../appendix/reading-list.md#exercise-solutions).
