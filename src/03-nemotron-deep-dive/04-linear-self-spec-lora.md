# 3.4 Linear self-speculation + LoRA drafter alignment

> **Goal of this lecture.** Walk through `generate_with_prefix_cache_block_diff` (the linear self-speculation inference routine) at the code level, and then read the LoRA + LK-hybrid loss recipe that adds ~36M parameters to bump TPF from ~5 to ~6. By the end you should be able to (i) trace one self-speculation cycle through the actual code path, (ii) explain why a LoRA on `o_proj` works as a drafter-alignment tool, and (iii) modify the recipe for a different acceptance/quality trade-off.

Background assumed: Lecture 2.3 (self-speculation cycle theory), Lecture 3.2 (mask modes), Lecture 3.3 (joint training). Comfort with KV-cache mechanics and LoRA basics.

References: `model.generate(...)` at line 1119 of `modeling_nemotron_labs_diffusion_vlm.py`, the `generate_with_prefix_cache_block_diff` helper, and the LoRA-tuning recipe in the NLD tech report §6.2.

---

## 1. The `generate()` entrypoint

The user-facing `generate` method (line 1119) is a thin wrapper:

```python
def generate(self, prompt_ids, max_new_tokens, steps, block_length,
             shift_logits, threshold, causal_context=True,
             temperature=0, pixel_values=None, image_sizes=None,
             eos_token_id=None):
    out_ids, nfe = generate_with_prefix_cache_block_diff(
        model=self,
        prompt=prompt_ids,
        gen_length=max_new_tokens,
        steps=steps,
        block_length=block_length,
        remasking="low_confidence",
        temperature=temperature,
        mask_id=self.mask_token_id,
        threshold=threshold,
        shift_logits=shift_logits,
        neg_entropy=False,
        causal_context=causal_context,
        pixel_values=pixel_values,
        image_sizes=image_sizes,
        eos_token_id=eos_token_id,
    )
    return out_ids, nfe
```

The actual work happens in `generate_with_prefix_cache_block_diff`, which is the **linear self-speculation cycle**.

### 1.1 Parameter cheat-sheet

| Parameter | Typical value | Meaning |
|---|---|---|
| `prompt_ids` | `(1, L_p)` | Tokenised prompt |
| `max_new_tokens` | 256–512 | Maximum committed tokens |
| `steps` | 32 | Total number of cycles (one cycle ≈ one block) |
| `block_length` | 32 | Block size $B$ for diffusion drafting |
| `threshold` | 0.9 | Confidence threshold for accept |
| `shift_logits` | False | Whether to shift logits by one position (legacy compat) |
| `remasking` | "low_confidence" | Strategy for re-masking unaccepted positions |
| `temperature` | 0 | 0 = argmax decoding; > 0 enables sampling |
| `causal_context` | True | If True, clean prefix uses AR causal mask; if False, bidirectional |
| `eos_token_id` | 11 | Token to terminate on |

Two non-obvious knobs: `threshold` and `causal_context`.

- **`threshold`** controls how many tokens commit per cycle. At 0.9, the model commits only tokens where its draft confidence exceeds 0.9. Lower threshold = more commits per cycle = higher TPF but possibly lower quality.

- **`causal_context`** controls whether the clean prefix is processed under AR-mode (causal) or bidirectional attention. For inference on a chat workload, `True` is correct (the prompt is a fixed prefix). For infilling, `False` would be appropriate.

### 1.2 The return value

`nfe` is "number of function evaluations", i.e., total forward passes used to generate `out_ids`. TPF = `len(out_ids) / nfe`.

A typical generation: prompt 64 tokens, output 256 tokens, threshold 0.9, block_length 32 → `nfe ≈ 50`, `TPF ≈ 256 / 50 ≈ 5.1`.

---

## 2. The self-speculation cycle in code

`generate_with_prefix_cache_block_diff` runs a loop. Each iteration is one self-speculation cycle. The structure (simplified pseudo-code reflecting the source):

```python
def generate_with_prefix_cache_block_diff(
        model, prompt, gen_length, steps, block_length,
        threshold, remasking, mask_id, eos_token_id, ...):

    device = prompt.device

    # 1. Prefill the prompt with AR mode
    set_all_attention_modes(model, 'autoregressive')
    prefix_cache = DynamicCache()
    prefix_logits = model.forward(input_ids=prompt, past_key_values=prefix_cache,
                                  use_cache=True).logits

    committed = prompt.clone()           # (1, L_p)
    nfe = 1                              # one forward done (prefill)
    eos_emitted = False

    for cycle in range(steps):
        if eos_emitted or committed.shape[1] - prompt.shape[1] >= gen_length:
            break

        # 2a. Set up draft block: append B mask tokens to the cache position
        draft_input = torch.full((1, block_length), mask_id, device=device)

        # 2b. Diffusion-draft forward (bidirectional + KV cache reuse)
        set_all_attention_modes(model, 'bidirectional')
        draft_cache = clone_cache(prefix_cache)        # see §3.2 below
        draft_out = model.forward(input_ids=draft_input,
                                  past_key_values=draft_cache,
                                  use_cache=True)
        draft_logits = draft_out.logits                # (1, B, V)
        nfe += 1

        # 2c. Sample / argmax draft tokens
        draft_tokens, draft_confs = sample_draft(draft_logits, threshold,
                                                  remasking, mask_id)
        # draft_tokens: (1, B), with mask_id at "low-confidence" positions

        # 2d. AR-verify forward (causal, sees draft_tokens as candidates)
        set_all_attention_modes(model, 'autoregressive')
        verify_cache = clone_cache(prefix_cache)
        verify_out = model.forward(input_ids=draft_tokens,
                                    past_key_values=verify_cache,
                                    use_cache=True)
        verify_logits = verify_out.logits              # (1, B, V)
        nfe += 1

        # 2e. Accept the longest prefix of draft_tokens that matches AR argmax
        ar_argmax = verify_logits.argmax(dim=-1)       # (1, B)
        n_accept = longest_prefix_match(draft_tokens, ar_argmax,
                                         draft_confs, threshold)

        # 2f. Commit accepted prefix and update prefix_cache
        committed = torch.cat([committed, draft_tokens[:, :n_accept]], dim=-1)
        prefix_cache = crop_cache(verify_cache, committed.shape[1])
        if eos_token_id in draft_tokens[:, :n_accept]:
            eos_emitted = True

    return committed[:, prompt.shape[1]:], nfe
```

Three subtleties worth tracking: the **cache cloning** (§3.2), the **longest-prefix match** with threshold (§3.3), and the **EOS handling** (§3.4).

### 2.1 The two forwards per cycle

Per cycle:
- **One bidirectional forward** (the diffusion draft), sees the draft block + clean prefix in KV cache.
- **One autoregressive forward** (the AR verify), sees the draft tokens, advances KV cache.

This matches Lecture 2.3 §3 exactly.

### 2.2 Where the speedup comes from

Both forwards read the same prefix KV cache. So the cost per cycle is:

- Prefix KV streaming (shared): 1× (read prefix from HBM)
- Draft forward incremental compute (B tokens): ~B/L of an AR forward
- Verify forward incremental compute (B tokens): ~B/L of an AR forward

For B = 32, L = 4096 (the typical context length), the per-cycle cost is ~3% of a "fresh" AR forward at the same context. Compared to 32 AR forwards (one per token), it's a ~$33/4 = 8.3\times$ speedup at full acceptance. With partial acceptance ($\mathbb{E}[m] = 6$), the realised speedup is ~$3\times$, matching the empirical numbers.

---

## 3. Cache management: the trickiest part

The KV cache is the source of self-speculation's speedup. Three operations on it:

### 3.1 Initial prefill

```python
prefix_cache = DynamicCache()
prefix_logits = model.forward(input_ids=prompt, past_key_values=prefix_cache,
                              use_cache=True).logits
```

The DynamicCache stores per-layer K and V tensors. After prefill, the cache has length `L_p` (the prompt length) per layer.

### 3.2 Cache cloning before each forward

The draft forward and the verify forward both *want* to advance the cache, but the cache they advance from is the same. If we let them share a single cache, the second forward would see the first's writes.

NLD solves this with **clone-on-write**:

```python
draft_cache = clone_cache(prefix_cache)
# draft_forward writes new K/V to draft_cache

verify_cache = clone_cache(prefix_cache)
# verify_forward writes new K/V to verify_cache
```

The clone is shallow, the K/V tensors up to length `L_p` are shared (read-only). Only the newly written entries are copies. Memory cost per clone is one block worth of K/V (~B × num_layers × num_kv_heads × head_dim × 2 bytes × 2 for K and V), which for NLD-8B at B=32 is ~16 KB per clone. Negligible.

### 3.3 Cache cropping after commit

```python
prefix_cache = crop_cache(verify_cache, committed.shape[1])
```

After commit, the prefix cache is set to the *verify cache truncated to the committed length*. This means:

- Tokens that were accepted: their K/V are kept (from the verify forward).
- Tokens that were rejected: their K/V are discarded.

The next cycle uses the new prefix cache as its read-only base.

This is what allows self-speculation to amortise the prefill cost across cycles. Each cycle pays $O(B)$ KV writes (for B drafted tokens), and the prefix cache grows by $\mathbb{E}[m]$ tokens per cycle.

### 3.4 EOS handling

If any accepted token is the EOS token, generation terminates. This happens in the `if eos_token_id in draft_tokens[:, :n_accept]` line.

A subtlety: the **first** EOS in the accepted prefix wins. So if `draft_tokens[:, :n_accept] = [t1, t2, EOS, t4, t5]`, only `[t1, t2, EOS]` are committed (not `[t1, t2, EOS, t4, t5]`). This requires a small extra check in the code that the simplified pseudo-code above doesn't show.

---

## 4. The accept/reject decision

The heart of self-speculation is **what counts as "accepted"**. NLD uses a **longest-prefix-match-with-threshold** rule:

```python
def longest_prefix_match(draft_tokens, ar_argmax, draft_confs, threshold):
    # draft_tokens: (1, B), the diffusion draft
    # ar_argmax:    (1, B), the AR verify's argmax at each position
    # draft_confs:  (1, B), the diffusion confidence at each draft position
    # threshold:    e.g. 0.9

    match = (draft_tokens == ar_argmax)                # (1, B), boolean
    conf  = (draft_confs >= threshold)                 # (1, B), boolean
    accept = match & conf                              # (1, B), boolean

    # Longest contiguous-True prefix
    n = 0
    for i in range(accept.shape[1]):
        if accept[0, i]:
            n += 1
        else:
            break
    return n
```

Two conditions for a position to be accepted:
1. **AR-verify match.** The AR-verify's argmax at that position equals the diffusion draft. This guards against quality drops.
2. **High confidence.** The diffusion draft itself was high-confidence. This is a secondary filter, sometimes the AR-verify matches a low-confidence diffusion draft (random chance), but committing such tokens would degrade quality.

The longest *contiguous* prefix that satisfies both is accepted. Position $i+1$ is not accepted if position $i$ is not, because each subsequent position depends on the previous one's commit.

### 4.1 Why not allow non-contiguous accepts?

Non-contiguous accepts (e.g., commit positions $i = 0, 2, 5$ but not 1, 3, 4) would break causality for the next cycle. The AR verify at position 5 depends on positions 0–4 being committed correctly; if position 1 is uncommitted, the model at position 5 sees a different left-context than it expected.

In principle one could re-verify after non-contiguous accepts, but the bookkeeping is complex and the gain is small. NLD's authors chose contiguous-only.

### 4.2 The threshold knob

Changing the threshold trades acceptance for quality:

| Threshold | Avg n_accept | TPF | Quality (MATH) |
|---|---|---|---|
| 0.5 | 12.5 | 7.3 | 48.2 |
| 0.7 | 9.8 | 6.4 | 49.0 |
| 0.9 (default) | 5.7 | 6.0 | 49.5 |
| 0.95 | 3.2 | 4.4 | 49.7 |

The default 0.9 sits at the elbow: increasing further (0.95) sharply lowers TPF for negligible quality gain; decreasing (0.7) raises TPF by 7% but loses 0.5 quality points. Tech report Table 5.

---

## 5. The LoRA drafter alignment

Even with the joint training, the diffusion-draft distribution and the AR-verify distribution disagree at ~30% of positions. NLD's tech report §6.2 proposes a fine-tuning recipe to bring them closer: a **LoRA on the `o_proj` weights**, trained with an **LK-hybrid loss**.

### 5.1 The LoRA itself

| Property | Value |
|---|---|
| Target | `o_proj` only (the output projection of each attention layer) |
| Rank | 128 |
| Alpha | 512 |
| Scaling | $\alpha / r = 4$ |
| Layers | All 34 layers |
| Per-layer params | $2 \cdot 128 \cdot 4096 \cdot 1 = 1.05$ M |
| Total LoRA params | $34 \cdot 1.05 = 35.6$ M (≈ 0.4% of base) |

The choice of `o_proj` and not other matrices is deliberate. The `o_proj` is the layer that **merges** the per-head attention outputs back into the residual stream. Aligning the drafter and verifier requires aligning these merged outputs, modifying `o_proj` is the most direct route.

### 5.2 The LK-hybrid loss

A pure AR-distillation loss (KL between AR logits and draft logits) would overfit the draft to the AR distribution. A pure diffusion loss would not align the draft to AR.

NLD's "LK-hybrid" loss combines both:

$$
L_\text{hybrid} = \lambda_\text{L} \cdot L_\text{LM}^\text{draft} + \lambda_\text{K} \cdot \text{KL}\bigl(p_\text{verify} \,\Vert\, p_\text{draft}\bigr)
$$

where:
- $L_\text{LM}^\text{draft}$ is the draft's own cross-entropy on the masked positions (the standard diffusion loss).
- $\text{KL}(p_\text{verify} \Vert p_\text{draft})$ is the KL divergence from the AR-verifier's distribution to the draft's distribution at each masked position.
- $\lambda_\text{L} = 1$, $\lambda_\text{K} = 0.5$ (production defaults).

This adds two forwards (one diffusion, one AR) per training step over the LoRA-tuning dataset.

The reported result: TPF goes from 5.0 (base model) to 5.99 (with LoRA), at the cost of 1% wall-clock overhead per inference (the LoRA's matrix products).

### 5.3 LoRA training dataset

The tech report uses ~10B tokens of the Tulu-mix-3 SFT dataset for LoRA fine-tuning. The training is fast: ~1 H100-hour per LoRA at this dataset size, because (a) only the LoRA params receive gradient, and (b) the dataset is small.

### 5.4 Loading the LoRA at inference

If a LoRA adapter is shipped, it's loaded after the base model:

```python
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained(base_repo, trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora_repo)
model.merge_and_unload()  # optional: merge into base for zero inference overhead
```

NLD's released checkpoint does NOT ship a LoRA. The LoRA recipe is documented; users wanting +20% TPF can run the recipe themselves.

---

## 6. Tuning knobs for production deployment

For practitioners deploying NLD self-speculation, the operationally important knobs:

| Knob | Default | Adjust if… |
|---|---|---|
| `block_length` | 32 | Lower (16) for very-short outputs; higher (64) saturates fast |
| `threshold` | 0.9 | Lower (0.7) for higher TPF at marginal quality cost |
| `temperature` | 0 | > 0 for sampling-based decoding (chat); use `top_p` separately |
| `steps` | 32 | Lower for known-short outputs; higher does no harm |
| `causal_context` | True | False only for infilling tasks |
| LoRA adapter | None | Train + load if TPF is bottlenecked |

The default settings are tuned for "open-domain chat at b=1 on B200", the model card's headline use case. Other deployments (long-context, high-batch, code) may want different defaults.

---

## 7. Tracing one cycle by hand

To make the cycle concrete, let's walk through one specific cycle. Setup:

- Prompt: `"What is 17 × 23?\n"` (8 tokens)
- True answer: `"17 × 23 = 391."` (8 tokens)
- Block length: $B = 4$ (smaller than default 32 for illustration)
- Threshold: 0.9
- After prefill, prefix KV cache has length 8.

**Cycle 1:**

1. Draft input: `[MASK, MASK, MASK, MASK]`
2. Diffusion forward: draft tokens = `["17", " ×", " 23", " ="]` with confidences `[0.99, 0.98, 0.97, 0.94]`.
3. AR-verify forward (input = draft tokens): AR argmax = `["17", " ×", " 23", " ="]`.
4. Accept check:
   - Position 0: match=Yes, conf=0.99 ≥ 0.9 → accept.
   - Position 1: match=Yes, conf=0.98 ≥ 0.9 → accept.
   - Position 2: match=Yes, conf=0.97 ≥ 0.9 → accept.
   - Position 3: match=Yes, conf=0.94 ≥ 0.9 → accept.
   - n_accept = 4 (all four).
5. Commit `["17", " ×", " 23", " ="]`, prefix cache now length 12.

**Cycle 2:**

1. Draft input: `[MASK, MASK, MASK, MASK]`
2. Diffusion forward: draft tokens = `[" 391", ".", " It", " was"]` with confidences `[0.97, 0.92, 0.41, 0.78]`.
3. AR-verify forward: AR argmax = `[" 391", ".", "<EOS>", "?"]`.
4. Accept check:
   - Position 0: match=Yes, conf=0.97 ≥ 0.9 → accept.
   - Position 1: match=Yes, conf=0.92 ≥ 0.9 → accept.
   - Position 2: match=No (" It" ≠ "<EOS>") → reject.
   - Position 3: not reachable (contiguous).
   - n_accept = 2.
5. Commit `[" 391", "."]`. Prefix cache now length 14.

**Cycle 3:**

1. Draft input: `[MASK, MASK, MASK, MASK]`
2. Diffusion forward: tokens = `["<EOS>", "<EOS>", "<EOS>", "<EOS>"]`, confs `[0.99, 0.99, 0.98, 0.98]`.
3. AR-verify: argmax = `["<EOS>", ...]`.
4. Accept first position. EOS detected. Stop.

**Total:** 3 cycles, 7 tokens committed (`"17 × 23 = 391.<EOS>"`), 6 forwards (1 prefill + 5 cycle forwards). TPF = 7 / 6 ≈ 1.17 over the whole sequence including prefill. Excluding prefill: 7 / 5 = 1.4 (small because the example is short; for longer outputs the per-cycle TPF dominates and is much higher).

---

## 8. Common bugs in implementations

If you implement self-speculation yourself, four bugs are easy to make:

1. **Forget to clone the cache.** Draft and verify share writes; chaos.
2. **Crop cache to wrong length.** Off-by-one: `committed.shape[1]` is the new total length; using `n_accept` alone gives the wrong cache range.
3. **Forget to re-set attention mode between forwards.** Verify forward runs with bidirectional mask → incorrect KV writes for subsequent cycles.
4. **Forget the EOS-early-termination.** Output gets garbage after EOS.

The NLD reference implementation handles all four correctly; if you're reading the code, you can use this list as a check.

---

## 9. Exercises

1. The cache-clone is "shallow", meaning the underlying tensors are shared up to the prefix length. Sketch what happens to memory consumption over the course of a 256-token generation if the clone were *deep* instead.

2. The accept rule requires both AR-match AND confidence ≥ threshold. What's the trade-off if we drop the confidence requirement (accept on AR-match alone)? Predict the effect on TPF and quality.

3. The longest-prefix-match is contiguous. Devise a 4-token example where allowing non-contiguous accepts would meaningfully increase TPF. Argue why NLD doesn't do this.

4. The LoRA targets only `o_proj`. Suppose you LoRA-tune `q_proj` instead. What would happen? Be specific about which forward pass (diffusion vs AR) would change behaviour.

5. Implement `clone_cache` and `crop_cache` for a `DynamicCache` (HF's KV cache class). The former should not copy the underlying tensors. State the memory cost analytically.

Solutions to (2), (4), (5) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 10. Further reading

- **`generate_with_prefix_cache_block_diff`** in the NLD source. Read it after this lecture.
- **NLD Tech Report §6.2**, the LoRA + LK-hybrid loss recipe with full hyperparameters.
- **Speculative decoding paper** (Leviathan et al., 2023). [arXiv:2211.17192](https://arxiv.org/abs/2211.17192). The original two-model speculative decoding from which NLD's accept rule descends.
- **EAGLE-3** (Li et al., 2024), the strongest two-model competitor. NLD's accept rule is a simplification.
- **PEFT (LoRA library)** docs, for the `merge_and_unload()` mechanics.

Next, Lecture 3.5: quadratic self-speculation. Same idea but with $B(B+1)/2$ candidate positions per cycle instead of $B$, using a fancier mask structure to verify multiple candidates in parallel.
