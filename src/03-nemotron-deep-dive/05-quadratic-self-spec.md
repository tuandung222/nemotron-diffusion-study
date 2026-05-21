# 3.5 — Quadratic self-speculation + FlexAttention

> **Goal of this lecture.** Walk through `sbd_inference_diffusion_quadratic` (NLD's "quadratic" self-speculation mode) at the code level. Quadratic self-spec expands the draft from $B = 32$ candidate positions to $B(B+1)/2 = 528$ candidate positions per cycle, using a custom FlexAttention mask, and bumps TPF from ~6 (linear) to ~8 (quadratic). By the end you should be able to: (i) explain why "quadratic" is the right name, (ii) reconstruct the position-id and attention-mask layout from the code, and (iii) reason about when quadratic vs linear self-spec is the right choice.

Background assumed: Lecture 2.3 (linear self-spec), Lecture 3.2 (mask modes, especially `sbd_block_diff_mask`), Lecture 3.4 (linear self-spec code). Familiarity with FlexAttention.

References: `sbd_inference_diffusion_quadratic` at line 1140 of `modeling_nemotron_labs_diffusion_vlm.py`, and the `sbd_block_diff_mask` predicate (line 206).

---

## 1. The intuition: try more drafts in parallel

Linear self-spec drafts $B$ positions per cycle and commits the longest accepted prefix. If the average accepted prefix is $\mathbb{E}[m] = 6$, you "waste" $B - m = 26$ positions per cycle — they were drafted but rejected because the verifier disagreed at some earlier position.

What if instead of drafting just *one* candidate per position, we drafted *multiple* candidates and let the verifier pick the best? That's the quadratic self-spec idea: at each of the $B$ positions, instead of drafting one token conditioned on its left neighbours' drafts, draft a *block* of candidates.

Concretely, quadratic self-spec drafts:
- Position 1: $[m_1]$ ($1$ candidate)
- Position 2: $[d_1, m_2]$ ($d_1$ from position 1's draft, $m_2$ is masked candidate)
- Position 3: $[d_1, d_2, m_3]$
- ...
- Position $B$: $[d_1, ..., d_{B-1}, m_B]$

Total candidates: $1 + 2 + 3 + ... + B = B(B+1)/2$. For $B = 32$, that's 528 candidates per cycle.

The verifier then sees all $B(B+1)/2$ candidates in *one* forward pass (with a custom mask), and picks the longest path that has high acceptance.

---

## 2. Why "quadratic"?

Linear self-spec drafts $O(B)$ candidates per cycle. Quadratic drafts $O(B^2)$. Hence the name.

But quadratic cost in WHAT? Two metrics:

- **FLOPs per cycle.** Quadratic forward sees $B^2/2$ sequence positions instead of $B$. So the per-cycle FLOPs are ~$B/2$ times larger than linear self-spec. For $B = 32$, that's ~16× more FLOPs per cycle.
- **HBM bytes per cycle.** Still ~1× (the cache is read once); the extra positions write to the activation buffer but not back to HBM in the forward direction.

So quadratic self-spec is **compute-heavy but memory-light**. It only makes sense when you have spare compute, i.e., when you're memory-bound. At batch 1 on B200, you're heavily memory-bound — so the extra compute is "free" (the GPU was idle anyway). At batch 256, the extra compute is not free — and quadratic self-spec is actually slower than linear.

The tech report confirms: quadratic self-spec wins on **b=1 latency** at the cost of being neutral or slightly worse at high batch.

---

## 3. The `sbd_inference_diffusion_quadratic` entrypoint

Function signature (line 1140):

```python
@torch.no_grad()
def sbd_inference_diffusion_quadratic(
    self,
    clean_input_ids: Optional[torch.Tensor],
    draft_input_ids: torch.Tensor,
    block_length: int,
    draft_only: bool = False,
    past_key_values: Optional[Cache] = None,
    use_cache: bool = False,
    pixel_values: Optional[torch.FloatTensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
):
```

Two modes:

- `draft_only=True`: just produce drafts from the clean prefix. Used for the first cycle (no prior draft).
- `draft_only=False`: the full quadratic-expansion forward.

### 3.1 The `draft_only` branch

The simpler branch (line 1157):

```python
if draft_only:
    assert clean_input_ids is not None

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    enc_config.self_spec_inference_mode = "default"
    input_ids = torch.cat([clean_input_ids, draft_input_ids], dim=-1)

    # vision-aware forward (similar to text-only path)
    outputs = self.encoder(input_ids=input_ids,
                           position_ids=None,
                           past_key_values=past_key_values,
                           use_cache=use_cache,
                           is_training=False)

    hidden_states = outputs.last_hidden_state
    logits = self.diffusion_head(hidden_states)

    past_key_values = getattr(outputs, "past_key_values", None)
    if use_cache and past_key_values is not None:
        _crop_dynamic_cache(past_key_values, clean_input_ids.shape[1])

    return logits, past_key_values
```

This is the "warm-up" forward: append a block of masks to the clean prefix and run a single diffusion forward. Returns logits at the draft positions.

After this forward, the cache holds the clean prefix's KV (we crop to `clean_input_ids.shape[1]`). The draft tokens themselves are *not* in the cache.

### 3.2 The full quadratic-expansion branch

The interesting branch (line 1186+):

```python
else:
    enc_config.self_spec_inference_mode = "quadratic"

    draft_len = block_length * (block_length + 1)        # 2 × B(B+1)/2 = B(B+1)
    draft_input_ids = torch.cat(
        [
            draft_input_ids.view(-1, block_length, 1),
            torch.full(
                (draft_input_ids.shape[0], block_length, block_length),
                fill_value=self.config.mask_token_id,
                device=draft_input_ids.device,
            ),
        ],
        dim=-1,
    ).view(-1, draft_len)
```

This is the candidate expansion. Trace it:

- Input: `draft_input_ids` of shape `(B_batch, block_length)` — the linear-style drafts from a previous cycle.
- Reshape to `(B_batch, block_length, 1)`.
- Concatenate `(B_batch, block_length, block_length)` of mask tokens → `(B_batch, block_length, block_length+1)`.
- Flatten to `(B_batch, block_length × (block_length+1))`.

So per position $i = 0, 1, ..., B-1$, we now have $B+1$ tokens: the linear draft token at position $i$, followed by $B$ mask tokens. The total length per batch is $B(B+1) = 32 \cdot 33 = 1056$.

Wait — the tech report says $B(B+1)/2 = 528$ candidates. Where's the factor of 2?

The factor of 2 comes from the fact that the *total* sequence length is $B(B+1)$, but each position only attends to $B(B+1)/2$ positions on average (the lower triangle of the candidate matrix). So compute-wise it's quadratic in $B$, but the sequence representation is $B(B+1)$ tokens.

### 3.3 Position IDs: the heart of the trick

Lines 1209–1218 build the position-id tensor:

```python
per_block_position_ids = torch.arange(
    clean_len, clean_len + block_length + 1, device=draft_input_ids.device
)[None,].repeat(block_length, 1)
per_block_position_ids += torch.arange(block_length, device=draft_input_ids.device).view(-1, 1)
```

Step 1: `torch.arange(clean_len, clean_len + B + 1)` produces a `(B+1,)` row of consecutive position IDs starting from `clean_len`.

Step 2: `.repeat(B, 1)` makes this a `(B, B+1)` matrix where every row is `[clean_len, clean_len+1, ..., clean_len+B]`.

Step 3: Add `torch.arange(B)` along axis 0 (as a column vector). So row $i$ gets shifted by $i$:

```
Row 0: [clean_len    , clean_len + 1, clean_len + 2, ..., clean_len + B    ]
Row 1: [clean_len + 1, clean_len + 2, clean_len + 3, ..., clean_len + B + 1]
Row 2: [clean_len + 2, clean_len + 3, clean_len + 4, ..., clean_len + B + 2]
...
Row B-1: [clean_len + B-1, clean_len + B, ..., clean_len + 2B - 1]
```

Each row $i$ is a candidate sequence: "if we had committed $i$ tokens so far, here's the next $B+1$ positions' position-ids".

The key insight: **different candidates share position IDs**. For instance, row 0's position $j$ and row 1's position $j-1$ both have position-id `clean_len + j`. They represent the same "absolute" position in the eventual output, just under different candidate prefixes.

This allows the model to **reuse rotary embedding angles** across candidates that share absolute positions. Mechanically, when computing attention from a candidate at position $j$ in row $i$, all keys at the same absolute position (across different rows) share the same rope angle.

### 3.4 The custom attention mask

The mask for quadratic self-spec is `sbd_block_diff_mask` (Lecture 3.2 §5). Recap:

- Within the noisy view: bidirectional intra-block (M_BD restricted to noisy positions only).
- Noisy → clean: offset-block-causal (M_OBC).
- Clean → clean: **strict token-causal** (replacing M_BC).

The strict token-causal clean→clean is what enables **multi-candidate verification in one forward**. Each candidate position attends to its own preceding-position candidates, not the candidates of other rows.

For the 1056-token quadratic sequence at $B = 32$:
- Length: $32 \cdot 33 = 1056$ noisy positions per batch (the candidates).
- Plus `clean_len` positions of clean prefix (unchanged).
- Total mask: $(clean\_len + 1056)^2$ pairs, with ~99.5% sparsity (most pairs masked).

FlexAttention handles this efficiently. The kernel processes only the ~5K non-zero tiles in the mask.

---

## 4. The acceptance / commit step

After the quadratic forward, the model has logits for all $B(B+1)/2 = 528$ candidate positions. The accept rule is similar to linear self-spec but generalised:

1. For each candidate row $i = 0, ..., B-1$, find the longest accepted prefix using the same threshold rule (AR-match + confidence ≥ threshold).
2. **Among all rows, pick the row with the longest accepted prefix.**
3. Commit that row's accepted prefix.

The "pick the longest" step is what makes quadratic-spec quadratic-better-than-linear: instead of being limited by the *single* draft sequence's accepted length, we get the *max* over $B$ alternative draft sequences.

### 4.1 Concrete TPF gain

Suppose linear self-spec has acceptance rate $\alpha = 0.92$ per position, giving:

$$
\mathbb{E}[m]_\text{linear} = \frac{1 - 0.92^{32}}{1 - 0.92} \approx 5.7
$$

For quadratic self-spec with $B$ independent draft sequences (an idealization), the max over $B$ trials of geometric-with-parameter $(1-\alpha)$ is:

$$
\mathbb{E}[\max_{i \leq B} m_i] \approx \log_{1/\alpha}(B) + 1/(1-\alpha) - \gamma
$$

For $B = 32, \alpha = 0.92$: $\log_{1/0.92}(32) \approx 41.7$, so the dominant term is large but capped at $B$. In practice the effective max is ~10, matching NLD's reported quadratic TPF.

### 4.2 Why not pure $\alpha$ improvement?

A natural alternative: improve $\alpha$ (via better drafter alignment) instead of expanding to quadratic. The tech report explored this with the LoRA recipe (Lecture 3.4) and got $\alpha = 0.92 \to 0.95$, which gives:

- Linear TPF at $\alpha = 0.95$: $\mathbb{E}[m] \approx 8.5$.
- Quadratic TPF at $\alpha = 0.92$: $\mathbb{E}[m] \approx 10$.

So quadratic is still slightly better, but the LoRA route is simpler. NLD's recommended stack: LoRA + linear self-spec for production b=1; quadratic self-spec for "demo / max-throughput" scenarios where you're willing to pay 1.5× the per-cycle FLOPs for an extra 1.5× TPF.

---

## 5. Wall-clock vs FLOPs

A note on the trade-off: quadratic self-spec is FLOP-heavy. For B = 32, the quadratic forward processes 1056 noisy positions vs 32 for linear self-spec — about 33× more positions.

But because the GPU is memory-bound at b=1, the wall-clock cost is not 33× more. The forward time is dominated by HBM access (the model weight load), which is independent of the number of positions processed (as long as everything fits in SRAM).

Empirically (tech report Table 8 on B200, b=1):

| Mode | FLOPs per cycle | Wall-clock per cycle | TPF |
|---|---|---|---|
| AR | 1.0× | 5 ms | 1.0 |
| Linear self-spec | 1.1× | 5.5 ms | 6.0 |
| Quadratic self-spec | 8× (in FLOPs) | 7.5 ms | 8.0 |

Quadratic costs 50% more wall-clock per cycle but commits 1.3× more tokens, so the per-token wall-clock is **lower**.

This is the key trick: at b=1 on memory-bound hardware, extra FLOPs cost negligible wall-clock until you saturate the SMs. Quadratic exploits this slack to get more TPF.

---

## 6. When NOT to use quadratic self-spec

Quadratic loses its advantage in three scenarios:

### 6.1 High batch sizes

At b ≥ 16, the GPU starts becoming compute-bound. Now the extra FLOPs from quadratic actually translate to extra wall-clock. Linear self-spec wins on per-token throughput.

NLD's tech report shows quadratic and linear self-spec converge in TPF around b = 8, with linear leading at b = 16 and beyond.

### 6.2 Long sequences with small block_size

If the workload is "1000-token output", the cycle count is `1000 / m`. For linear with $m = 6$, that's 167 cycles. For quadratic with $m = 8$, that's 125 cycles.

But each quadratic cycle is 1.5× wall-clock vs linear. So total wall-clock:
- Linear: 167 × 5.5 ms = 919 ms.
- Quadratic: 125 × 7.5 ms = 937 ms.

Essentially the same. Quadratic's per-cycle TPF advantage is eaten by its per-cycle wall-clock disadvantage.

### 6.3 Very short outputs

If the output is < 32 tokens (less than one block), the diffusion-draft is partly wasted. The first cycle has limited acceptance potential because so few positions are committed-then-verified.

For short outputs, even AR baseline can be competitive with self-spec, regardless of mode.

---

## 7. The training side: how is quadratic supported?

Quadratic self-spec at inference is supported by a specific training configuration:

```yaml
use_sbd_objective: true        # use sbd_block_diff_mask instead of block_diff
block_length: 32               # match inference
self_spec_inference_mode: quadratic    # inference flag, not training
```

The `use_sbd_objective: true` in the config triggers the trainer to use `sbd_block_diff_mask` (strict-token-causal clean side, restricted-bidirectional noisy side) instead of the standard `block_diff_mask`. This affects training only.

The model trained with `sbd_block_diff_mask` is **slightly worse** at linear self-spec (because the clean view is now token-causal, which is less permissive) but **substantially better** at quadratic self-spec (because the mask correctly handles per-candidate causality during verification).

NLD's released checkpoints support **both** modes by serving two attention-mask configurations: `block_diff` (for linear) and `sbd_block_diff` (for quadratic). At inference, the user picks the mode via `self_spec_inference_mode = "default" | "quadratic"`.

---

## 8. Practical recommendation

For most workloads, the recommendation is **linear self-spec with LoRA + threshold 0.9**. Quadratic is worth the complexity when:

- Workload is single-stream (b=1) on B200 or similar high-bandwidth GPU.
- Output length is moderate (32–512 tokens).
- Latency is the primary metric (not throughput).

In these conditions, quadratic delivers ~1.3× linear's tokens-per-second. The code complexity (custom FlexAttention mask, position-id manipulation, candidate expansion) is real, but the implementation is contained in NLD's source.

For everything else, linear self-spec is the right default.

---

## 9. Visualizing the candidate matrix

Concretely, consider $B = 4$ (small block for illustration). The candidate matrix has shape $(B, B+1) = (4, 5)$:

```
Row 0: [d0,   m,    m,    m,    m  ]
Row 1: [d1,   d0,   m,    m,    m  ]
Row 2: [d2,   d1,   d0,   m,    m  ]
Row 3: [d3,   d2,   d1,   d0,   m  ]
```

Here `d_i` are the linear draft tokens at position $i$ (committed earlier), and `m` is the mask token.

Position-ids (with `clean_len = 100`):

```
Row 0: [100, 101, 102, 103, 104]
Row 1: [101, 102, 103, 104, 105]
Row 2: [102, 103, 104, 105, 106]
Row 3: [103, 104, 105, 106, 107]
```

Each row has 5 positions; the position-ids overlap row-to-row.

After the forward, the logits at the mask positions in row $i$ predict what would come next *if we had committed* `d_0, d_1, ..., d_{i-1}` as the prefix. The acceptance step then picks the row with the longest prefix that:
- Matches AR argmax (verified)
- Has draft confidence ≥ threshold

For example, if Row 2 has accepted positions `[d2 ✓, d1 ✓, d0 ✓, m ✗]` (the d-prefix accepts; the m fails), we commit 3 tokens. If Row 3 has `[d3 ✓, d2 ✓, d1 ✓, d0 ✓, m ✓]`, we commit 5 tokens (all of d3..d0, plus one mask).

In the best case, the longest accepted row is row $B-1$ with $B$ tokens committed. In the worst case, the longest accepted row is row 0 with 1 token committed.

Linear self-spec has $\mathbb{E}[m] \approx 6$. Quadratic raises this to $\approx 8$, because we get the *max* over $B = 32$ candidate prefixes.

---

## 10. Exercises

1. Compute the FLOP overhead of quadratic vs linear self-spec for $B = 32$. State as a ratio.

2. Derive the exact $\mathbb{E}[m]$ for quadratic self-spec under the simplifying assumption of i.i.d. acceptance with probability $\alpha$ across candidates. Compare to the formula for linear self-spec.

3. The `sbd_block_diff_mask` differs from `block_diff_mask` in two ways (see Lecture 3.2 §5). Which of these two differences is required for quadratic self-spec to work, and which is redundant?

4. Suppose we wanted to expand to "cubic" self-spec: $B(B+1)(B+2)/6$ candidates per cycle. Sketch the candidate matrix and the position-id layout. State the per-cycle FLOP and TPF implications.

5. The released NLD checkpoint supports linear (block_diff trained) and quadratic (sbd_block_diff trained). Suppose a user wants to fine-tune the checkpoint to maximize quadratic TPF only. What objective and mask should they use?

Solutions to (2), (3) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 11. Further reading

- **`sbd_inference_diffusion_quadratic`** at line 1140 of the modeling file. Read it after this lecture.
- **NLD Tech Report §6.3** — full quadratic self-spec description, including the ablation that shows when it pays off.
- **FlexAttention reference** — for the mask construction. The quadratic self-spec FlexAttention mask is one of the more complex examples in the wild.
- **Speculative decoding survey** (Xia et al., 2024). [arXiv:2401.07851](https://arxiv.org/abs/2401.07851). For the broader landscape of "multi-candidate" speculative decoding.

Next, Lecture 3.6: extending NLD to the VLM with a Pixtral vision encoder + asymmetric dual-stream.
