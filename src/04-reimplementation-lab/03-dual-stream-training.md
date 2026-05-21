# Lab 4.3 Dual-stream training

> **Notebook:** [`notebooks/03-dual-stream-training.ipynb`](https://github.com/tuandung222/nemotron-diffusion-study/blob/main/notebooks/03-dual-stream-training.ipynb)
>
> **Runtime:** CPU < 3 min · GPU < 30 s · **Params:** 0.83M

This is the most important lab in the series, it introduces the **dual-stream input layout** and the **structured attention mask** $M_{BD} \cup M_{OBC} \cup M_{BC}$ that make joint AR + diffusion training work in a single forward.

## What you build

### 1. The 2L × 2L structured mask

```python
def make_dual_stream_mask(L, block_size, device):
    total = 2 * L
    q = torch.arange(total, device=device).view(-1, 1)
    kv = torch.arange(total, device=device).view(1, -1)
    x0_flag_q = (q >= L).long()
    x0_flag_kv = (kv >= L).long()
    block_q = torch.where(x0_flag_q == 1, (q - L) // block_size, q // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv - L) // block_size, kv // block_size)
    M_BD = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    M_OBC = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
    M_BC  = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
    return M_BD | M_OBC | M_BC
```

The output is a `(2L, 2L)` boolean tensor. Visualised with `L=8, block_size=4`:

```
[[1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]   ← noisy block 0, can see itself only
 [1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0]   ← noisy block 1, can see itself + clean block 0 (M_OBC)
 [0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0]
 [0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0]
 [0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0]   ← clean block 0, sees itself only (M_BD restricted)
 [0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]   ← clean block 1, sees clean blocks ≤ 1 (M_BC)
 [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]
 [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]]
```

This is identical to NLD's `block_diff_mask` (Lecture 3.2 §3). Read the section now if rows 8-15 look surprising, the key insight is that **clean rows are strictly causal**, not bidirectional.

### 2. The joint loss

```python
def joint_training_step(model, x, mask_token_id, block_size_diff, alpha=0.5):
    # AR forward: causal mask, length L
    causal = make_causal_mask(L, device=x.device)
    ar_logits = model(x, causal)
    L_ar = F.cross_entropy(ar_logits[:, :-1], x[:, 1:])

    # Diffusion forward: block_diff mask, length 2L
    dual = torch.cat([noisy, x], dim=-1)
    bd_mask = make_dual_stream_mask(L, block_size_diff)
    bd_logits = model(dual, bd_mask)
    L_diff = F.cross_entropy(bd_logits[:, :L][mask_indices], x[mask_indices])

    return L_ar, L_diff, L_ar + alpha * L_diff
```

Two forwards, two losses, one model. Exactly the joint objective from Lecture 2.2.

## What you measure

- Both losses drop together, AR ≈ 0.07, diffusion ≈ 1.6 after 150 steps.
- AR generation works: "The quick brown fox jumps over the lazy dog. Pack my box with five fi…"
- Block-diffusion generation produces text with **TPF > 1**.

## Sanity assertions

The notebook includes seven explicit assertions on the mask structure:

- Diagonal is True (self-attention always allowed).
- Noisy → noisy *across* blocks is False.
- Noisy → clean *within own block* is False (label-leak guard).
- Noisy → clean *previous block* is True (M_OBC).
- Clean → noisy is False (always).
- Clean → future clean is False.
- Clean → past clean is True.

All seven must pass, if they don't, the mask is wrong and downstream labs will fail.

## What the next lab changes

- We expose `generate()` with a `mode` parameter that dispatches to AR / block-diffusion / self-speculation.

## Prerequisites

- Labs 4.1, 4.2.
- Lectures 2.2 (joint loss + dual stream), 3.2 (structured masks in NLD).
