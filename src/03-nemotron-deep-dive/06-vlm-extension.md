# 3.6 VLM extension: Pixtral encoder + asymmetric dual-stream

> **Goal of this lecture.** Read the VLM-specific code in `modeling_nemotron_labs_diffusion_vlm.py`, the Pixtral encoder, the 2-layer MLP projector with 2×2 spatial merge, the `_embed_with_vision` helper, and the "asymmetric dual-stream" that puts vision tokens *only* in the clean view. By the end you should be able to (i) trace an image from raw bytes to LM input tokens, (ii) explain why vision tokens skip the noisy view, and (iii) recover the "exact merge" initialization recipe that bootstraps from a vision-language base.

Background assumed: Series 1 (foundations) + Series 2 (mixed AR-diff) + Lectures 3.1–3.5 of Series 3 (NLD code-level). Familiarity with at least one VLM architecture (LLaVA, Pixtral, Qwen2-VL) for context.

References: HF page https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B, the vision-aware paths in `modeling_nemotron_labs_diffusion_vlm.py` (`_embed_with_vision`, `forward` with `pixel_values`, `multi_modal_projector` class), and the Pixtral 12B paper.

---

## 1. The high-level architecture

NLD-VLM-8B is structurally the **same 8B text decoder + a Pixtral-style vision tower bolted on the front**. Three pieces:

```
[Image bytes]
    ↓ (preprocessor → patches)
[Pixtral encoder, 24 layers]
    ↓ (last hidden, dim 1024)
[Multimodal projector, 2-layer MLP + 2×2 spatial merge]
    ↓ (dim 4096, one token per merged patch)
[NLD-8B decoder (text-only), UNCHANGED]
```

Total parameters:

| Component | Params |
|---|---|
| Pixtral encoder (24L) | ~0.5B |
| Multimodal projector | ~30M |
| NLD-8B decoder (shared) | 8B |
| **Total VLM** | **~8.5B** |

The vision overhead is < 7% of total parameters. The NLD-8B decoder remains tri-mode (AR / diffusion / self-spec) and the Pixtral encoder is a fixed module, no diffusion magic on the vision side.

---

## 2. The Pixtral encoder

NLD adopts Pixtral's vision encoder verbatim. Configuration (from `vision_config`):

```json
{
  "hidden_size": 1024,
  "num_attention_heads": 16,
  "head_dim": 64,
  "patch_size": 14,
  "image_size": 1540,
  "num_hidden_layers": 24,
  "intermediate_size": 4096,
  "model_type": "pixtral",
  "rope_parameters": { "rope_theta": 10000, "rope_type": "default" }
}
```

Key facts:

- **Patch size 14.** Each 14×14 RGB patch becomes a 1024-dim token.
- **Image size up to 1540×1540.** Smaller images are kept at native resolution (no aspect-ratio distortion); larger images are tiled with the patches-of-images approach used by Pixtral. The maximum number of patches per image is $\lceil 1540/14 \rceil^2 = 110^2 = 12{,}100$.
- **24 layers, 16 heads, head_dim 64.** Standard Pixtral-style. The encoder is bidirectional (no causal mask), vision data has no notion of "future" tokens.
- **RoPE on positions.** Patches receive 2D RoPE based on (row, col) within the image. `rope_theta = 10000` (the standard ViT-style theta, not the LM's 1e6).

The encoder is initialized from the Pixtral-style vision tower of the **corresponding Ministral3-8B-Instruct-2512 VLM** (tech report sec 5.3), which has the exact same Pixtral-12B-class shape. This initialization is critical, the bottom of this lecture (§5) explains why.

---

## 3. The multimodal projector

After the encoder produces hidden states of shape `(num_patches, 1024)`, the projector maps each to LM input space. NLD's projector is:

```python
class NemotronLabsDiffusionVLMMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size  # = 2

        # Linear 1: 1024 × spatial_merge² → 4096
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * (self.spatial_merge_size ** 2),
            config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = nn.GELU()
        # Linear 2: 4096 → 4096
        self.linear_2 = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, image_features):
        # image_features: (num_patches, 1024)
        # Spatial-merge 2×2 patches into one merged-patch token
        merged = self._spatial_merge(image_features, self.spatial_merge_size)
        # merged: (num_patches / 4, 1024 * 4)
        hidden = self.linear_1(merged)
        hidden = self.act(hidden)
        out = self.linear_2(hidden)
        return out  # (num_patches / 4, 4096)
```

Three design choices:

### 3.1 Spatial merge 2×2

Pixtral's raw output is one token per 14×14 patch. For a 1540×1540 image: 12,100 tokens. That's a lot to pass through a 4096-d LM.

NLD merges every 2×2 patch grid into a single "merged-patch" token by concatenating the four hidden vectors:

```
Input:  4 patches × 1024 = 4096 floats per 2×2 grid
Output: 1 token × 4096 = 4096 floats (passes through Linear_1)
```

This reduces the per-image LM context from 12,100 to ~3,000 tokens. The exact reduction is 4× because each merge collapses 4 patches.

The merge is **deterministic and structural**, no learnable weights in the merge itself. It's just a reshape: `(H, W, 1024) → (H/2, W/2, 4096)`.

### 3.2 Why 4096 dim? (matching LM hidden)

The LM's input embedding is 4096-d. The projector's output is 4096-d. So projected vision tokens are inserted into the LM's input sequence as if they were word embeddings, no further transformation needed.

A 2-layer MLP (4096 → 4096) is enough to absorb the cross-modal distribution shift. Pixtral and Ministral3 (the LM base) were trained jointly via a similar projector, so the initialization makes the projector's job easier (Lecture 3.6 §5).

### 3.3 GELU vs SiLU

The projector uses GELU, the LM uses SwiGLU (silu-based). The choice reflects the upstream Pixtral convention: Pixtral's projector is GELU. NLD inherits this to make the Pixtral initialization compatible.

Changing the projector activation does not change performance to the same degree as the encoder initialization choice; the GELU choice is for compatibility with the source VLM's projector, not because GELU is strictly preferred over SiLU.

---

## 4. The asymmetric dual-stream

Here's where NLD-VLM's design becomes interesting. Recall (Lecture 3.2) that the joint training uses a dual-stream input `[noisy | clean]` of length $2L$. For text-only, vision tokens don't exist. But what about VLM?

**Naive approach:** dual-stream the vision tokens too. So if there are 1000 vision tokens and 100 text tokens, the dual stream has length $2 \cdot 1100 = 2200$.

**Problem:** vision tokens are *never masked*. The diffusion process applies to text tokens only, masking image patches is nonsensical (we can't denoise pixels). So the noisy-view's image positions are *identical* to the clean-view's image positions. We'd be doubling vision-token compute for no benefit.

**NLD's solution:** the asymmetric dual-stream. Vision tokens appear **only in the clean view**, not the noisy view.

Dual-stream layout (with $N_\text{vis}$ vision tokens, $L_\text{txt}$ text tokens):

```
        |--- noisy view ---|--- clean view ---|
position: 0 1 ... L_txt-1   L_txt L_txt+1 ... L_txt + N_vis + L_txt - 1
content:  t1 t2 ... t_L     V1 V2 ... V_Nvis t1 t2 ... t_L
                            ^^^^^^^^^^^^^^^^^
                            vision tokens (only here)
```

Total length: $L_\text{txt} + N_\text{vis} + L_\text{txt} = 2L_\text{txt} + N_\text{vis}$.

For a typical image-conversation: $L_\text{txt} = 500$, $N_\text{vis} = 3000$, dual-stream length = $2 \cdot 500 + 3000 = 4000$. The symmetric naive approach would have given $2 \cdot 3500 = 7000$ tokens, almost 2× more.

### 4.1 The attention mask under asymmetric dual-stream

The structured `block_diff_mask` now operates over a non-uniform layout. The code (line 460+, in `forward_process_complementary` or similar) constructs the mask with view-classification adapted:

```python
# Noisy view: positions [0, L_txt)
# Clean view, vision portion:    positions [L_txt, L_txt + N_vis)
# Clean view, text portion:      positions [L_txt + N_vis, 2L_txt + N_vis)

x0_flag_q  = (q_idx  >= L_txt)        # 0 = noisy, 1 = clean (vis or text)
is_vis_q   = (q_idx >= L_txt) & (q_idx < L_txt + N_vis)   # vision tokens
# ... similar for kv
```

The structured-mask rules adapt:

- **Noisy text → noisy text (M_BD):** same as before, bidirectional within block.
- **Noisy text → clean text (M_OBC):** same as before, offset-block-causal.
- **Noisy text → clean vision:** **always allowed** (no block-causal constraint). Vision tokens are the "always-attend-to" conditioning.
- **Clean text → clean text (M_BC):** block-causal as before.
- **Clean text → clean vision:** **always allowed**.
- **Clean vision → clean vision:** **bidirectional within image** (image-internal attention).
- **Clean vision → clean text:** **prohibited** (vision tokens should not attend to text, image encoding is independent of the conversation).

The "always-allowed" rules for noisy/clean-text to clean-vision implement the "vision is context for everything" intuition. The "prohibited" rule from vision to text ensures the vision tokens' representations don't depend on the chat (which would defeat caching).

### 4.2 Why this asymmetry matters

The asymmetric dual-stream:

- Reduces compute by ~2× compared to symmetric dual-stream (vision tokens are not doubled).
- Allows vision-feature caching: once the encoder runs and the projector projects, the resulting `(N_vis, 4096)` tokens can be cached. Future chat turns over the same image reuse this cache.
- Avoids the "what would noisy-vision even mean" semantic question. Vision tokens stay clean throughout.

Qualitatively, vision tokens only being present in the clean view skips roughly half of the per-vision-token attention work that a symmetric dual-stream would otherwise require. The tech report does not publish exact compute-savings numbers for this design, so we describe the effect qualitatively rather than quote a specific multiplier.

---

## 5. Exact-merge initialization

When training NLD-VLM from scratch, the initialization recipe (tech report sec 5.3) is:

1. **LM weights ← the NLD-8B *instruct* checkpoint** (the Joint-SFT'd version of `nvidia/Nemotron-Labs-Diffusion-8B`).
2. **Vision encoder weights ← the vision tower of `Ministral3-8B-Instruct-2512 VLM`** (the VLM counterpart of the same Ministral3 base that NLD initialized from).
3. **Projector weights ← the multimodal projector of the same `Ministral3-8B-Instruct-2512 VLM`**.
4. **Image-token embedding ← the existing `<image>` slot in the Ministral3 vocabulary** (no new vocabulary slot is added; the source VLM already defines this token).

The phrase "exact merge" refers to the fact that the source VLM (Ministral3-8B-Instruct-2512 VLM) was *built on the same Ministral3-8B base* that NLD-8B was continued-pretrained from. Therefore the projector's output space is already aligned with the LM's input embedding space at the architectural level: shapes match, hidden dimensions match, and the source projector was trained against an LM that the NLD LM differs from only by continued training on top of the same starting point. The tech report (sec 5.3) emphasises: "the merge is exact with no parameter mismatch or interpolation".

This is a stronger starting point than the typical VLM recipe (CLIP encoder + random projector + LLM): the encoder, projector, and LM were trained in a compatible setting before the merge, so the joint training only has to handle the diffusion-objective adaptation, not the modality-alignment from scratch.

The tech report (sec 5.3) does not publish a token count for the joint VLM training stage; we therefore avoid quoting a specific number here.

### 5.1 What does "exact merge" require?

For "exact merge" to work, the shapes must align:

- Pixtral vision encoder output: 1024-d.
- Pixtral projector output: 4096-d (its target LM dim).
- Ministral3 LM input: 4096-d.
- NLD-8B (starts from Ministral3): 4096-d.

So Pixtral's projector → Ministral3 LM is dim-compatible, and Ministral3 → NLD-8B is dim-compatible. All shapes match. The actual numeric values don't need to align perfectly, they just need to be in the right neighborhood, but the architectural shapes must match.

### 5.2 Counter-example: dim mismatch

If NLD-8B had used a different hidden dim (e.g., 5120), the "exact merge" would have failed. The projector would need a separate adapter or full retraining. NLD's choice to base on Ministral3 (rather than a custom-sized model) preserves dim compatibility.

This is a non-trivial design decision: NLD's authors made a choice that constrains the LM architecture for the sake of VLM convergence. The pay-off is a faster, cheaper VLM stage.

---

## 6. The `_embed_with_vision` helper

The function `_embed_with_vision` (around line 850 in the modeling file) handles the vision-aware embedding lookup:

```python
def _embed_with_vision(self, input_ids, pixel_values, image_sizes):
    """
    Replace <image> placeholder tokens in input_ids with the vision
    encoder's projected output for the corresponding image.
    """
    # 1. Get the standard word embedding for non-image tokens
    inputs_embeds = self.encoder.embed_tokens(input_ids)

    # 2. Run the vision encoder on each image
    vision_features = self.vision_model(pixel_values, image_sizes=image_sizes)
    # vision_features: list of (num_patches_i, 1024) tensors

    # 3. Project each image's features
    projected = [self.multi_modal_projector(f) for f in vision_features]
    # projected: list of (num_merged_patches_i, 4096) tensors

    # 4. Find <image> placeholder positions and splice in the projected features
    image_mask = (input_ids == self.config.image_token_id)
    inputs_embeds[image_mask] = torch.cat(projected, dim=0)

    return inputs_embeds
```

The key idea: the input `input_ids` contains an `<image>` placeholder token for each vision-token position (one placeholder per merged patch). The embedding lookup turns these into the LM's embedding for `<image>`, then we *overwrite* those positions with the projected vision features.

This pattern allows the LM to handle text and vision uniformly in subsequent layers, the vision tokens are just specially-initialized "input embeddings" that flow through the rest of the decoder unchanged.

### 6.1 Image-token ID

The `<image>` token is id `131072` (the last slot in the 131073-vocab). Special tokens are also handled uniformly: BOS, EOS, MASK, IMAGE all have specific ids, and the LM treats them all as "input embeddings", but only MASK is touched by the diffusion process.

### 6.2 Multi-image support

If a conversation contains multiple images, the input might be:

```
[BOS] How does this image compare to the next? <image_1> ... <image_2> ... [response]
```

The placeholders are: $N_\text{vis,1}$ tokens of `<image>` for image 1, then $N_\text{vis,2}$ tokens for image 2. The encoder runs once *per image*, and the projector projects each batch. The splice step (line 4 above) handles multiple images by iterating over the image-token positions in order.

NLD-VLM's tokenizer ships a multi-image template that handles this; users typically don't construct the placeholders manually.

---

## 7. Inference modes for VLM

The VLM supports the same three inference modes (AR / diffusion / self-spec) as the text model. The vision encoder runs once per input image; its output is fixed for the rest of the generation.

### 7.1 KV cache and vision tokens

When using KV caching, the vision tokens' K/V are part of the prefix cache (because the vision is part of the "clean prefix" of the conversation). The cache is set up once during prefill (which now includes the vision encoder forward) and then reused across diffusion-draft and AR-verify forwards.

This means: **vision encoder cost is a one-time prefix cost**, not a per-cycle cost. For a long conversation about an image, the encoder runs once and amortises across all subsequent generation cycles.

### 7.2 Self-spec on VLM

Self-spec works identically to text-only. The only difference: the prefix KV cache is longer (because it includes vision tokens). For a 3000-vision-token image, the cache is ~3000 + L_text tokens at prefix time.

At per-cycle inference, the cache is read once (~1 HBM load) and the draft / verify forwards each take a per-block fraction. So the per-cycle wall-clock at VLM is roughly the same as text-only, *if* the cache is in-HBM (which it usually is for short conversations).

For very long conversations or many cached images, the extra vision-token prefix in the KV cache becomes a real cost. The VLM benchmarks in tech report Table 9 record diffusion-mode TPF in the 2.5 to 3.6 range on multimodal evaluation suites, lower than the text-only diffusion-mode TPF of 2.57 reported in tech report Table 5. The difference reflects (a) image-task answers being shorter on average than text-task answers, which compresses the gain self-speculation can deliver, and (b) the extra cost of the long vision-token prefix in the KV cache.

---

## 8. The vision-side training

The Pixtral encoder is trained in two phases during the VLM stage:

**Phase 1, vision adapter warm-up (~5B tokens).** The LM is frozen; only the projector and the new `<image>` embedding train. Loss is the joint AR + diffusion. Goal: align the projector's output to the LM's input embedding distribution.

**Phase 2, full VLM training.** The LM, projector, and vision encoder are all updated. The joint loss continues. The encoder's learning rate is typically set lower than the LM's to avoid destroying the pretrained vision tower (the tech report does not publish the exact LR ratio for the VLM stage).

Phase 1 is short and cheap (~500 H100-hours). Phase 2 is longer (~5000 H100-hours).

By comparison, full pretraining of a comparable VLM from scratch (random init) typically takes orders of magnitude more compute than the post-merge fine-tuning needed here. The "exact merge" initialization is what makes the VLM stage cheap relative to either training a VLM from scratch or aligning a CLIP-style encoder with a freshly-randomised projector.

---

## 9. Practical examples

### 9.1 Loading and using NLD-VLM

```python
from transformers import AutoModelForCausalLM, AutoProcessor

repo = "nvidia/Nemotron-Labs-Diffusion-VLM-8B"
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True,
                                              torch_dtype="bfloat16", device_map="auto")

# Input: text + image
inputs = processor(
    text="Describe the contents of this image:",
    images=[some_pil_image],
    return_tensors="pt"
).to("cuda")

# Generate (with self-speculation)
out_ids, nfe = model.generate(
    prompt_ids=inputs.input_ids,
    max_new_tokens=200,
    steps=20,
    block_length=32,
    threshold=0.9,
    shift_logits=False,
    pixel_values=inputs.pixel_values,
    image_sizes=inputs.image_sizes,
    causal_context=True,
    temperature=0,
)

response = processor.decode(out_ids[0], skip_special_tokens=True)
```

### 9.2 Self-spec metrics on a VLM workload

Empirical numbers from NLD's VLM tech report on a "describe this image" benchmark:

| Mode | TPF | Wall-clock per response |
|---|---|---|
| AR | 1.0 | 200 ms |
| Block-diffusion | 2.5 | 80 ms |
| Linear self-spec | 4.5 | 45 ms |
| Quadratic self-spec | 5.5 | 38 ms |

Notice: TPF is lower than text-only (5–6 → 4.5) because vision-aware inference incurs more HBM traffic per cycle. The relative speedup over AR is still ~4.5×.

---

## 10. Exercises

1. The asymmetric dual-stream layout assumes vision tokens are immutable. What if some vision-related positions *did* need to be masked (e.g., a question-answer where the answer mentions a specific patch)? Design a hybrid layout.

2. The projector has ~30M parameters. Estimate what fraction of these are in the spatial-merge linear (`linear_1`) vs the post-merge linear (`linear_2`).

3. Suppose you wanted to support 4×4 (rather than 2×2) spatial merging. State the required code changes and the trade-off in vision-token count vs LM throughput.

4. The vision encoder runs at fp16/bf16. What's the impact on memory bandwidth of switching to fp8 for vision? Justify whether this is worth the engineering effort.

5. NLD-VLM's tech report claims that the asymmetric dual-stream saves "~2× compute vs symmetric". Derive this number from first principles for an image-conversation with 3000 vision tokens and 500 text tokens.

Solutions to (3), (5) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 11. Further reading

- **HF model card**: https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B
- **Pixtral 12B paper** (Mistral AI, 2024). https://mistral.ai/news/pixtral-12b/, for the encoder's design.
- **Llava-1.5 paper** (Liu et al., 2024), the "linear projector" VLM architecture that Pixtral and NLD-VLM both build on.
- **NLD Tech Report §5**, full VLM training pipeline including the exact-merge init and Phase-1/Phase-2 split.
- **Multimodal projector ablations** in the NLD repo's `multimodal_projector.py` (separate file), alternative projector designs.

Next, Lecture 3.7: benchmarks. We'll read NLD's published numbers against the SOL framework from Lecture 2.5 and identify which numbers are SOL-limited vs algorithm-limited.
