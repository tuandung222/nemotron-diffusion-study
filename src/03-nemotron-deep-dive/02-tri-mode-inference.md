# 3.2 Tri-mode inference: `set_attention_mode` deep dive

> **Goal of this lecture.** Read the actual source code of `set_attention_mode`, `compute_block_mask`, `block_diff_mask`, and `sbd_block_diff_mask` from `modeling_nemotron_labs_diffusion_vlm.py`, and connect each line to the abstract masks from Lecture 2.2. By the end you should be able to (i) construct the FlexAttention mask for any of the four modes from first principles, (ii) explain why each branch exists, and (iii) modify the code to add a fifth mode if needed.

Background assumed: Lecture 2.2 (structured attention mask theory), Lecture 3.1 (config + auto_map), comfort reading PyTorch + a smattering of `torch.nn.attention.flex_attention`.

Source file (all line numbers in this lecture refer to this file): `modeling_nemotron_labs_diffusion_vlm.py` in [https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B/tree/main](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B/tree/main).

---

## 1. The four attention modes

NLD's attention layer (`NemotronLabsDiffusionVLMFlexAttention`, line 90) supports four modes, selected via `self.mode`:

| Mode | Use case | Mask predicate |
|---|---|---|
| `autoregressive` | AR generation, AR-verify step of self-spec | `q >= kv` |
| `bidirectional` | Pure block-diffusion drafting | `True` (no mask) |
| `block_diff` | Dual-stream training, AR-mode bidirectional intra-block | `M_BD ∨ M_OBC ∨ M_BC` |
| `sbd_block_diff` | Quadratic self-speculation training | `M_BD ∨ M_OBC ∨ fully_causal_clean` |

The mode is set by `set_attention_mode(mode, block_size)` (line 149):

```python
def set_attention_mode(self, mode, block_size=None):
    self.mode = mode
    self.block_size = block_size
```

This is called *per layer* at the start of every forward pass. Note: there's no global "mode", every attention layer sets its own. In practice all 34 layers share the same mode, but the API allows per-layer overrides (used during the LoRA drafter alignment experiments, Lecture 3.4).

---

## 2. The `compute_block_mask` dispatcher

After setting the mode, the layer calls `compute_block_mask(mode, q_len, block_size)` to materialise the actual FlexAttention `BlockMask` object (line 153). Inside, four inner predicate functions are defined and one is selected:

```python
def bidirectional_mask(b, h, q, kv):
    return (q >= kv) | (q < kv)        # always True

def autoregressive_mask(b, h, q, kv):
    return (q >= kv)                   # standard causal

def block_diff_mask(block_size, b, h, q_idx, kv_idx, n):
    # ... 3-term composition; see §3 below

def sbd_block_diff_mask(block_size, b, h, q_idx, kv_idx, n):
    # ... 3-term composition with strict causal clean→clean; see §5 below
```

The signature `(b, h, q_idx, kv_idx)` is the FlexAttention convention: a boolean function over batch index, head index, query position, and key/value position. The function is JIT-compiled and broadcast over all pairs.

The dispatch on `mode` (line 235–255):

```python
if mode == 'bidirectional':
    attn_mask = bidirectional_mask
elif mode == 'autoregressive':
    attn_mask = autoregressive_mask
elif mode == 'block_diff':
    assert block_size is not None
    n = (q_len // 2) if q_len is not None else self.max_seq_length
    attn_mask = lambda b, h, q, kv: block_diff_mask(block_size, b, h, q, kv, n)
elif mode == 'sbd_block_diff':
    assert block_size is not None
    n = (q_len // 2) if q_len is not None else self.max_seq_length
    attn_mask = lambda b, h, q, kv: sbd_block_diff_mask(block_size, b, h, q, kv, n)
else:
    raise ValueError(f"Unknown attention mode: {mode}")
```

Two notable points:

- For the block-diff modes, the predicate receives `n = q_len // 2`. The variable name `n` is suggestive: $n$ is the length of *one view* (either noisy or clean). The total query length is $2n$ because the views are concatenated. The mask predicate uses $n$ to classify each position as belonging to the noisy view (`q < n`) or the clean view (`q >= n`).
- The `q_len = None` branch falls back to `max_seq_length`, which is the model's maximum context. This is the prefill case.

After predicate selection, the code constructs a `BlockMask`:

```python
block_mask = create_block_mask(
    attn_mask, B=None, H=None, Q_LEN=Q_LEN, KV_LEN=Q_LEN
)
return block_mask
```

`create_block_mask` from `torch.nn.attention.flex_attention` produces a sparse mask representation that's much cheaper to apply than a dense $Q_{\text{LEN}} \times Q_{\text{LEN}}$ boolean array. For $L = 4096$ and $B_{\text{block}} = 32$, the mask sparsity is ~95% (most pairs are masked); the BlockMask exploits this directly in the kernel.

---

## 3. The `block_diff_mask` predicate: line by line

This is the heart of NLD's dual-stream training. The full function (line 161):

```python
def block_diff_mask(block_size, b, h, q_idx, kv_idx, n):
    # Indicate whether token belongs to xt or x0
    x0_flag_q  = (q_idx  >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices (within each view)
    block_q  = torch.where(x0_flag_q  == 1,
                            (q_idx  - n) // block_size,
                            q_idx  // block_size)
    block_kv = torch.where(x0_flag_kv == 1,
                            (kv_idx - n) // block_size,
                            kv_idx // block_size)

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
        (block_q > block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q  == 0)
    )

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal
```

Let's unpack each term.

### 3.1 Identifying view membership

```python
x0_flag_q  = (q_idx  >= n)
x0_flag_kv = (kv_idx >= n)
```

The convention: positions `0..n-1` are the *noisy* view ($x_t$), positions `n..2n-1` are the *clean* view ($x_0$). The `x0_flag` is 1 if the position is in the clean view.

The variable naming reflects the original DLM literature: $x_t$ is the noisy/masked version, $x_0$ is the clean original.

### 3.2 Block indexing

```python
block_q = torch.where(x0_flag_q == 1,
                       (q_idx - n) // block_size,
                       q_idx // block_size)
```

Within each view, positions are partitioned into blocks of size $B = 32$. The block index is computed *relative to the start of the view*. So:

- Position 0 (noisy) → block 0
- Position 31 (noisy) → block 0
- Position 32 (noisy) → block 1
- Position $n$ (clean) → block 0  (because we subtract $n$ first)
- Position $n + 31$ (clean) → block 0
- Position $n + 32$ (clean) → block 1

The `n` subtraction is the key trick: it makes block 0 of the clean view *correspond to the same content positions* as block 0 of the noisy view. So we can compare across views.

### 3.3 M_BD: block diagonal

```python
block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
```

Allowed if:
- Same block index (same content range), AND
- Both query and key in the same view.

So this term enables:
- **Noisy → noisy intra-block**: each masked position can attend to all other masked/observed positions in the same block. This is the bidirectional denoising attention.
- **Clean → clean intra-block**: this is part of the AR loss; clean positions can attend to all other clean positions in the same block. But, this would *break* causality! In particular it would let clean position $i$ see clean position $i+1$, leaking the label.

Wait. The third term `block_causal` will handle the clean-only case more restrictively. Let's see how the terms compose. The final mask is the **OR** of all three:

```
allowed = M_BD ∨ M_OBC ∨ M_BC
```

So a clean-to-clean intra-block pair is `True` under M_BD. Under M_BC it's also gated to `(block_q >= block_kv)`, but that's true for intra-block.

But this seems to violate causality on the clean view. Let me re-read.

Actually no: **the `block_diff_mask` is the *training-time* mask for the `bidirectional` block-diffusion paradigm**. In this paradigm, even the clean view uses *bidirectional intra-block* attention. This works because:

- The diffusion loss is computed *only* at noisy positions where the original token has been replaced by `[MASK]`.
- The AR loss is computed only at clean positions, *but* using a strict-causal mask reused later (the AR-mode forward).

In the joint training step, the model uses `block_diff` for the dual-stream forward, but the AR loss is computed using a *separate* forward in `autoregressive` mode. So the M_BD term being permissive on clean-clean is fine, clean-view attention here is for the diffusion-loss computation only.

This is a subtle code organisation: there are *two* forwards per joint-training step:
1. `block_diff` forward on `[noisy ‖ clean]`, producing the diffusion-loss logits at noisy positions and the clean-view representations.
2. `autoregressive` forward on `[clean]` only, producing the AR-loss logits.

Both forwards share weights; the AR forward is a regular causal pass over `clean`.

The trick of using `block_diff` to compute the diffusion loss while still allowing clean→clean intra-block is fine because the diffusion loss doesn't read clean→clean, it only reads noisy→{noisy, clean} predictions.

### 3.4 M_OBC: offset block-causal

```python
offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q  == 0)
)
```

Allowed if:
- Noisy query, clean key, AND
- The query's block index is strictly greater than the key's block index.

So a noisy position in block $i$ can attend to clean positions in blocks $0, 1, ..., i-1$ but **not** block $i$. The "offset" is the strict `>`. This is the **conditional context**: when denoising noisy block $i$, the model conditions on the *clean* version of all preceding blocks (which is its left-context, analogous to KV cache in AR).

The reason for `>` not `>=`: block $i$'s clean view is the *label* for denoising block $i$. If we allowed the noisy view to look at the clean view at the same block index, we'd leak the answer.

### 3.5 M_BC: block-causal

```python
block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
```

Allowed if:
- Both clean (`x0_flag` = 1 for both), AND
- Query's block index ≥ key's block index.

So clean-view positions can attend to clean-view positions in the same or earlier block. This is **block-causal**: causality at block granularity, not token granularity. Within a block, the M_BD term provides bidirectional attention (as we noted above). Across blocks, it's strictly causal.

Why block-causal and not token-causal? Two reasons:

1. **It makes the AR-loss forward cheap.** During inference, the KV cache is computed in block-causal layout. Each new block prefills against a cached past, analogous to AR but at block granularity.

2. **It enables self-speculation.** During self-speculation, the AR-verify forward reuses the clean-prefix KV from the most recent committed prefix. The KV is laid out block by block.

### 3.6 Putting it together

The three terms compose as follows:

| q (block i) | kv (block j) | M_BD | M_OBC | M_BC | Allowed? |
|---|---|---|---|---|---|
| noisy | noisy (j = i) | ✓ |  |  | ✓ |
| noisy | noisy (j ≠ i) |  |  |  | ✗ |
| noisy | clean (j < i) |  | ✓ |  | ✓ (cond context) |
| noisy | clean (j = i) |  |  |  | ✗ (label leak guard) |
| noisy | clean (j > i) |  |  |  | ✗ |
| clean | noisy (any j) |  |  |  | ✗ |
| clean | clean (j ≤ i) | (✓ if j = i) |  | ✓ | ✓ (block-causal) |
| clean | clean (j > i) |  |  |  | ✗ |

This matrix is exactly the structured 2L × 2L mask from Lecture 2.2 Figure 3.

> **Mental model.** The noisy view is the "current block being denoised". The clean view is the "left-context". The mask enforces (a) intra-block bidirectional denoise, (b) left-context conditioning via offset-block-causal, (c) strict block-causal evolution of the clean view across blocks.

---

## 4. Why FlexAttention?

A dense materialisation of this mask is impractical:

- For sequence length $L = 4096$ and dual-stream length $2L = 8192$, the mask has $8192^2 = 67$M entries, 8 MB at boolean, but the *write* and *read* costs dominate.
- The mask is highly structured (block-sparse): only ~5% of entries are `True`.
- A new mask must be built per forward pass (when $n$ or $B$ varies).

PyTorch's **FlexAttention** API solves this by accepting a *predicate function* and compiling it into a fused attention kernel. The kernel:

1. Tiles Q × K into blocks of, say, 128 × 128.
2. For each tile, evaluates the predicate at the four corners and decides if the tile is *full* (all True), *empty* (all False), or *partial* (needs per-element check).
3. Empty tiles are skipped entirely. Full tiles use standard FlashAttention. Partial tiles use a per-element predicate evaluation.

For NLD's `block_diff` mask, ~95% of tiles are empty, so the kernel achieves > 10× speedup vs a dense mask.

The trade-off: FlexAttention is **PyTorch 2.5+** only (or torch.compile + manual lowering). For older PyTorch, NLD falls back to SDPA with a dense mask, usable but slower.

---

## 5. The `sbd_block_diff_mask` for quadratic self-speculation

Lecture 3.5 will cover quadratic self-speculation in detail. For now, just note the variant predicate (line 206):

```python
def sbd_block_diff_mask(block_size, b, h, q_idx, kv_idx, n):
    # ... same x0_flag and block_q/block_kv as block_diff_mask ...

    # **1. Block Diagonal Mask (M_BD), only on the noisy view **
    block_diagonal = (block_q == block_kv) & (x0_flag_kv == 0) & (x0_flag_q == 0)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
        (block_q > block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q  == 0)
    )

    # **3. Fully Causal Mask, replaces block_causal on the clean view **
    fully_causal = (q_idx >= kv_idx) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | fully_causal
```

Two differences from the standard `block_diff_mask`:

1. **M_BD is restricted to noisy → noisy only** (the `& (x0_flag_kv == 0)` clause). The clean view no longer enjoys intra-block bidirectional attention.
2. **M_BC is replaced with `fully_causal`**, using token-level rather than block-level causality on the clean view.

In aggregate: the clean view in `sbd_block_diff` mode uses **strict token-causal attention** (standard AR mask), while the noisy view still uses bidirectional intra-block + offset-block-causal to clean.

This stricter clean-side mask is required for **quadratic self-speculation** because the clean side serves not only as conditional context but also as the "verified candidate" for the next round of drafting. A bidirectional clean view would leak future tokens; the fully causal mask preserves AR-side integrity.

We'll see in Lecture 3.5 how this interacts with the candidate expansion: the noisy view is expanded from $B$ to $B(B+1)/2$ positions, and the clean view's strict causality ensures that each candidate position's verification depends only on tokens that have actually been committed.

---

## 6. End-to-end: what happens in one forward pass

To make the mask construction concrete, here's the lifecycle of one dual-stream forward pass during training:

```
1. Trainer constructs dual-stream input
     z = concat([noisy_ids, clean_ids])  # shape (B, 2L)

2. Trainer calls model.forward(input_ids=z, ...)

3. Inside model.forward:
   a. Embed tokens: hidden_states = embedding(z)       # (B, 2L, d)
   b. For each of 34 layers:
        i.  layer.set_attention_mode('block_diff', block_size=32)
        ii. block_mask = layer.compute_block_mask('block_diff',
                                                   q_len=2L, block_size=32)
        iii. hidden_states = layer(hidden_states, ..., attn_mask=block_mask)

4. lm_head(hidden_states)  →  logits of shape (B, 2L, V)

5. Loss:
   - Diffusion loss: cross-entropy at positions 0..L-1 where noisy_ids == MASK
   - AR loss: separate forward in 'autoregressive' mode on clean_ids alone
```

This is **two forwards per training step**, once in `block_diff` mode on the 2L sequence (for diffusion loss + clean-view representations), once in `autoregressive` mode on the L clean sequence (for AR loss). The forward cost is ~3× a standard AR forward of length L, because the block-diff forward sees 2L tokens and the AR forward sees L.

> **Cost accounting.** A standard AR pretraining step on length L is $O(L^2 d + L d^2)$. NLD's joint training step is roughly $O((2L)^2 d + 2L d^2) + O(L^2 d + L d^2) = O(5 L^2 d + 3 L d^2)$, about 3× the FLOPs. The tech report's reported training cost of "~$3\times$ standard pretraining" matches.

At inference, the model uses one of the four modes per forward, and the cost is the same as a standard AR forward (mode = `autoregressive`) or 2L-sequence forward (mode = `block_diff` for full self-spec cycle). Lecture 2.3 covered the inference-side accounting; here we're confirming it matches the code.

---

## 7. Modifying the mask: a hypothetical fifth mode

Suppose you wanted to add a fifth mode that allowed **block-bidirectional on both views with strict block boundaries**: every block (in both views) is bidirectional intra-block, but no cross-block attention.

You'd add to `compute_block_mask`:

```python
def block_isolated_mask(block_size, b, h, q_idx, kv_idx, n):
    x0_flag_q  = (q_idx  >= n)
    x0_flag_kv = (kv_idx >= n)

    block_q  = torch.where(x0_flag_q  == 1, (q_idx  - n) // block_size, q_idx  // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)

    # Same block, same view
    return (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

# In the dispatch:
elif mode == 'block_isolated':
    assert block_size is not None
    n = (q_len // 2) if q_len is not None else self.max_seq_length
    attn_mask = lambda b, h, q, kv: block_isolated_mask(block_size, b, h, q, kv, n)
```

Use case: a degenerate "block-only" denoising baseline, with no cross-block conditioning. Useful for ablation studies. Tech-report Table 3 contains exactly this ablation, it shows ~3 points worse on math evals vs the full `block_diff`.

---

## 8. Exercises

1. Construct the truth table for `sbd_block_diff_mask` analogous to §3.6. Confirm the differences from `block_diff_mask`.

2. The `block_diff_mask` allows clean → clean intra-block via M_BD. Argue informally why this does *not* leak the AR label, given that the AR loss is computed in a *separate* `autoregressive`-mode forward.

3. Compute the sparsity of the `block_diff_mask` for $L = 4096$, $B = 32$ (so $n = 4096$, $2L = 8192$). How many "True" entries are there, as a fraction of all $8192^2$ pairs?

4. Why does FlexAttention use `B = None, H = None` in `create_block_mask`? What does that signal?

5. Modify the `block_diff_mask` to allow **only the diagonal of M_BD** (i.e., each noisy position attends only to itself, plus the clean offset-block-causal terms). What new mode would this implement? When would it be useful?

Solutions to (3), (5) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 9. Further reading

- **FlexAttention paper / PyTorch docs.** [pytorch.org/docs/main/nn.attention.flex_attention](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html), the API and kernel design.
- **`modeling_nemotron_labs_diffusion_vlm.py`**, lines 90–270, the full implementation we walked.
- **NLD Tech Report, §3.3**, diagrams of M_BD, M_OBC, M_BC and the rationale.
- **Block Diffusion (Arriola et al., 2024)**, §3, the predecessor construction of the block-causal mask.

Next, Lecture 3.3: how the joint training pipeline schedules masking ratios across DP ranks, why "global loss averaging" matters, and the 2-stage curriculum that produces the released checkpoint.
