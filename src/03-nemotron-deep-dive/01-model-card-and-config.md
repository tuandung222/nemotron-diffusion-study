# 3.1 - Model card and config walkthrough

> **Goal of this lecture.** Read the Hugging Face model card and `config.json` of Nemotron-Labs-Diffusion (both 8B text and VLM-8B variants) field by field, and connect every numeric to the architectural concepts from Series 1–2. By the end you should be able to (i) reproduce the model architecture from the config alone, (ii) identify what every non-standard field controls, and (iii) spot the design choices that distinguish NLD from a vanilla Llama-style AR LM.

Background assumed: Series 1 (foundations), Series 2 (joint loss + dual-stream + self-speculation). Familiarity with one standard HF config (Llama / Mistral) for comparison.

References:
- HF page: https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B
- VLM page: https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B
- Tech report: https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_Diffusion_Tech_Report_v1.pdf

---

## 1. Where the weights come from

The text model `nvidia/Nemotron-Labs-Diffusion-8B` is trained from a Ministral3-8B-Instruct-2512 initialization. Specifically:

- **LM backbone:** initialized from Ministral3-8B-Instruct weights (decoder layers, embeddings, lm_head).
- **Mask token:** a single new token (id 100) is added to the vocabulary as `[MASK]`. Its embedding is initialized to the mean of the embedding matrix.
- **Joint pretraining:** ~600B tokens of additional joint-objective training on the same mixed corpus used for Ministral3. This is the "Stage 1" of NLD's training pipeline (Lecture 3.3).

The VLM extension `nvidia/Nemotron-Labs-Diffusion-VLM-8B` adds:

- A Pixtral-style vision encoder (24 layers, hidden 1024, image up to 1540×1540).
- A 2-layer MLP multimodal projector with 2×2 spatial merging.
- An asymmetric dual-stream layout (vision tokens only in the clean view).

The VLM LM-side is initialized from the text NLD-8B, then fine-tuned jointly with the vision tower over ~70B image-text tokens.

---

## 2. The text-model `config.json` in full

The complete config of `Nemotron-Labs-Diffusion-8B`:

```json
{
  "ar_loss_weight": 1.0,
  "architectures": ["NemotronLabsDiffusionModel"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attn_implementation": "sdpa",
  "auto_map": {
    "AutoConfig": "configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig",
    "AutoModel":  "modeling_nemotron_labs_diffusion.NemotronLabsDiffusionModel"
  },
  "block_size": 32,
  "bos_token_id": 1,
  "dlm_loss_weight": null,
  "dlm_paradigm": "bidirectional",
  "dp_varying_mask_ratio": false,
  "eos_token_id": 11,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "mask_token_id": 100,
  "max_position_embeddings": 262144,
  "mlp_bias": false,
  "model_type": "nemotron_labs_diffusion",
  "num_attention_heads": 32,
  "num_hidden_layers": 34,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_parameters": {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 16.0,
    "llama_4_scaling_beta": 0.1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 16384,
    "rope_theta": 1000000.0,
    "rope_type": "yarn"
  },
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "5.0.0",
  "use_cache": false,
  "vocab_size": 131072
}
```

We'll walk every field. Skip standard ones; focus on the NLD-specific ones.

### 2.1 The shape: a Llama-class 8B with GQA

```
hidden_size:                4096
num_hidden_layers:          34
num_attention_heads:        32
num_key_value_heads:         8
head_dim:                  128
intermediate_size:        14336
vocab_size:              131072
```

This is structurally identical to **Mistral / Ministral3-8B**. 32 Q-heads sharing 8 KV-heads (GQA group size 4), 34 transformer decoder layers, FFN expansion ratio 14336/4096 ≈ 3.5×, SwiGLU activations (`hidden_act: silu`).

Parameter count: 7.85B + 0.005B for the mask token embedding ≈ 8.0B total. The reason NLD-8B has the same FLOP/parameter budget as Ministral3-8B is intentional - the diffusion machinery is *free* in inference compute. Only the training cost increases (Lecture 3.3).

### 2.2 RoPE and context length

```
max_position_embeddings:        262144
rope_parameters.original_max_position_embeddings: 16384
rope_parameters.rope_type: "yarn"
rope_parameters.factor: 16.0
rope_parameters.rope_theta: 1000000.0
```

NLD uses **YaRN** RoPE scaling with factor 16. Trained at native context 16,384, extended to 262,144 (16× longer) via YaRN's interpolation/extrapolation hybrid. The `beta_fast = 32`, `beta_slow = 1` parameters control the high-frequency rotation extrapolation that YaRN uses to preserve high-precision positional encoding even at the extended context.

Practical implication: NLD-8B can prefill up to 256K tokens but the *bulk* of evaluation is in the 0–16K range. Long-context performance at 256K is described in the tech report §5.3 - accuracy drops to ~70% of the within-training-context value (typical for YaRN-scaled models).

The choice of `rope_theta = 1e6` matches the Ministral3 base.

### 2.3 The diffusion-specific knobs

The fields that distinguish NLD from a vanilla Mistral config:

```
ar_loss_weight:        1.0
dlm_loss_weight:       null   # actually 0.5 in VLM config; "null" here means use the default
dlm_paradigm:          "bidirectional"
block_size:            32
mask_token_id:         100
dp_varying_mask_ratio: false
```

| Field | Meaning |
|---|---|
| `ar_loss_weight` | Coefficient on the AR-loss term in the joint objective. NLD uses 1.0. |
| `dlm_loss_weight` | Coefficient on the diffusion-loss term. In the VLM `config.json` this is 0.5 (so $\alpha = 0.5$ in the notation of Lecture 2.2). In the text-model config it's `null`, which the loader interprets as "use the default" - which for `NemotronLabsDiffusionConfig` is also 0.5. |
| `dlm_paradigm` | "bidirectional" means the diffusion mode uses fully bidirectional attention within blocks (the standard MDLM / block-diffusion setting). Other valid values from the source code are "autoregressive" (degenerate - no diffusion) and "block_diff" (block-causal + bidirectional intra-block, used at training time). |
| `block_size` | The block size $B$ used for both training and inference. Tied to `block_length` at generation time. |
| `mask_token_id` | The token id used for `[MASK]` in the absorbing-state diffusion. NLD reserves id 100, the first available slot after the Ministral3 special tokens. |
| `dp_varying_mask_ratio` | If true, each DP rank sees a different masking ratio per batch (Lecture 3.3 §4). Disabled in the released text model; can be enabled when fine-tuning. |

The **structural insight** here is that the NLD config is *strictly a superset* of the Ministral3 config - all standard Mistral fields are preserved, plus a handful of NLD-specific knobs. You can load an NLD checkpoint with `AutoModelForCausalLM` using `trust_remote_code=True` and get a working AR-mode LM that's bit-equivalent to a fine-tuned Ministral3-8B-Instruct.

### 2.4 The auto_map and `trust_remote_code`

```
auto_map:
  AutoConfig: configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig
  AutoModel:  modeling_nemotron_labs_diffusion.NemotronLabsDiffusionModel
```

NLD ships **all of its forward / generation / loss code as `.py` files in the HF repo**, not in the `transformers` library. You must pass `trust_remote_code=True` when loading. The model classes live in:

- `configuration_nemotron_labs_diffusion.py` - the config class (subclass of `PretrainedConfig`).
- `modeling_nemotron_labs_diffusion.py` - the model (subclass of `Ministral3PreTrainedModel`).

For the VLM:
- `configuration_nemotron_labs_diffusion_vlm.py`
- `modeling_nemotron_labs_diffusion_vlm.py` - the central file we'll be reading line by line in Lectures 3.2 onwards.

The decision to live outside `transformers` reflects the modeling code's novelty: the dual-stream forward, the `set_attention_mode`, and the FlexAttention block-mask construction are not standard `transformers` building blocks. NLD's authors chose to keep the code in the model repo where they can iterate on it freely.

### 2.5 `attn_implementation: "sdpa"` - and the FlexAttention escape hatch

The text config sets `attn_implementation: "sdpa"` as default. This uses PyTorch's `scaled_dot_product_attention` for AR-mode and bidirectional-mode inference - which is well-supported across hardware.

But when the model needs the **structured 2L × 2L attention mask** (block-causal + block-diagonal + offset-block-causal), SDPA cannot express it. NLD falls back to **FlexAttention** for that case. The VLM config sets `attn_implementation: null`, which the loader resolves to FlexAttention by default. Lecture 3.2 §4 covers the mask-construction code in detail.

The choice between SDPA and FlexAttention is dynamic: at inference, the model can swap implementations as needed. Each attention layer has a `set_attention_mode()` method that controls which mask is built and which implementation is used.

### 2.6 `use_cache: false`

A surprising default. NLD ships with `use_cache: false`, meaning KV-cache reuse is *not* the default behaviour. This is because in dual-stream training the noisy and clean views are concatenated and re-encoded each step - there's no autoregressive history to cache.

At inference, the user explicitly enables KV caching when calling `generate()` or `sbd_inference_diffusion_quadratic()`. The cache is then used aggressively to share the clean-prefix state across self-speculation cycles.

### 2.7 Tied / untied embeddings

```
tie_word_embeddings: false
```

NLD does **not** tie the input and output embeddings. This matters for the diffusion mode: the input embedding for `[MASK]` is the same vector everywhere, but the output logit for `[MASK]` at generation time is *not* used (we never want to predict `[MASK]` as an output). Untying the embeddings lets the lm_head zero out the `[MASK]` logit independent of how `[MASK]` is consumed at input.

The lm_head has shape `[131072, 4096]` ≈ 537M params. Together with the embedding's 537M, the embeddings sum to ~1.07B. This is **13% of the total parameter count** - a significant memory chunk in the sub-10B regime.

---

## 3. The VLM `config.json`

Most fields are inherited from the text model. The VLM-specific additions:

```json
{
  "diff_loss_weight": 0.5,
  "dlm_arch": "encoder",
  "dlm_loss_weight": 0.5,
  "dlm_type": "llada",
  "complementary_mask": true,
  "global_loss_avg": true,
  "spatial_merge_size": 2,
  "projector_hidden_act": "gelu",
  "multimodal_projector_bias": false,
  "vision_feature_layer": -1,
  "vocab_size": 131073,
  "vision_config": { ... 24-layer Pixtral encoder ... }
}
```

| Field | Meaning |
|---|---|
| `diff_loss_weight`, `dlm_loss_weight` | Both 0.5; the diffusion-loss coefficient is the same as for the text-only model. |
| `dlm_arch: "encoder"` | The diffusion is computed by a separate "encoder" sub-module (the noisy-view branch). The clean-view branch is the AR branch. |
| `dlm_type: "llada"` | The masked-diffusion variant uses the LLaDA-style simplified loss (Lecture 1.3). |
| `complementary_mask: true` | Enables the complementary-mask training trick (Lecture 3.3): the noisy view masks positions $i$; the clean view's loss is computed on positions $\neg i$. |
| `global_loss_avg: true` | Average the loss over all positions in the batch (across DP ranks) rather than per-position. This stabilises the gradient when different ranks see different mask ratios. |
| `spatial_merge_size: 2` | After the Pixtral encoder, every 2×2 patch is merged into one token via the projector. |
| `projector_hidden_act: "gelu"` | The MLP projector uses GELU (Pixtral convention) rather than the SiLU/SwiGLU of the LM backbone. |
| `vision_feature_layer: -1` | The projector consumes the last-layer output of the Pixtral encoder. |
| `vocab_size: 131073` | One additional token over the text model: an `<image>` placeholder. |

The `vision_config` defines a 24-layer Pixtral encoder:

```
hidden_size:           1024
num_attention_heads:     16
head_dim:                64
patch_size:              14
image_size:            1540
intermediate_size:     4096
num_hidden_layers:       24
```

Image up to 1540×1540, patched into 14×14 tiles, giving up to $(1540/14)^2 = 12{,}100$ patches per image. After 2×2 spatial merge: up to ~3000 tokens per image. The projector then maps each 4096-dim merged-patch vector to a 4096-dim LM token (no dim change).

The vision tower is **initialized from `mistralai/Pixtral-12B-2409`'s encoder**, which has the same shape. The choice ensures the projected vision tokens land in approximately the right neighborhood of the LM's input embedding space - see the "exact merge initialization" in Lecture 3.6.

---

## 4. The undocumented fields: training-only knobs

Several config fields are not used at inference but appear in the released config:

```
adaptive_mask_rate:        false
ada_dlm_loss_ratio:        null
ada_perm_ratio_global:     null
ada_perm_ratio_per_block:  null
multi_sampling:            null
prefix_ratio:              0.8
random_length_prob:        0
tok_mask_half_life_ratio:  null
num_ar_layers:             0
num_diffusion_layers:      0
num_skip_loss_tokens:      0
```

These are toggles for variant training procedures explored during research:

- `prefix_ratio: 0.8` - During training, 80% of the sequence is treated as a "prefix" (no mask, full causal AR), and the remaining 20% is the candidate-block range that can be masked. Reflects the realistic chat workload: most tokens are the prompt; only the tail is generation.
- `random_length_prob: 0` - Whether to randomly truncate sequences during training. Disabled.
- `multi_sampling: null` - Whether to use multiple samples of the noisy view per training step. Disabled in production.
- `num_ar_layers, num_diffusion_layers: 0, 0` - Allows splitting the model into AR-only and diffusion-only sub-stacks. Both 0 means a *fully shared* stack (all 34 layers see both losses). This is the production setting.

The fact that all these are disabled in the production config reflects an **ablation-tested-and-converged** state. The simplest joint-loss recipe wins.

---

## 5. Counting parameters

A line-by-line count of the 8B text model:

| Component | Per-block | × 34 blocks | Bytes (BF16) |
|---|---|---|---|
| `q_proj` (4096 → 4096) | 16.78 M | 570 M | 1.14 GB |
| `k_proj` (4096 → 1024) | 4.19 M | 142 M | 0.28 GB |
| `v_proj` (4096 → 1024) | 4.19 M | 142 M | 0.28 GB |
| `o_proj` (4096 → 4096) | 16.78 M | 570 M | 1.14 GB |
| `gate_proj` (4096 → 14336) | 58.72 M | 1996 M | 3.99 GB |
| `up_proj` (4096 → 14336) | 58.72 M | 1996 M | 3.99 GB |
| `down_proj` (14336 → 4096) | 58.72 M | 1996 M | 3.99 GB |
| `input_layernorm` | 4.10 K | 0.14 M | 0.0003 GB |
| `post_attention_layernorm` | 4.10 K | 0.14 M | 0.0003 GB |
| **Per-block subtotal** | **218.1 M** | **7411 M** | **14.81 GB** |
| Embedding (131072 × 4096) | - | 537 M | 1.07 GB |
| Final RMSNorm | - | 4.1 K | 8 KB |
| `lm_head` (4096 → 131072) | - | 537 M | 1.07 GB |
| `[MASK]` embedding (init) | - | 4 K | 8 KB |
| **Total** | - | **~8.49 B** | **~16.97 GB** |

So the byte cost of NLD-8B in BF16 is **~17 GB**, dominated by 7.4B params in the transformer stack and 1.07B in the two embedding tables.

> **Sanity check.** This matches the published parameter count of 8.0B (rounded). The 0.5B difference is the second embedding (lm_head not tied), which Mistral-style configs typically share. The 17 GB byte cost is what Lecture 2.5's SOL formula uses for $W$ in `Tokens/s = b · BW / (2W)`.

---

## 6. What the config does *not* tell you

Three architectural facts that don't appear in `config.json`:

1. **The shared LoRA option.** The released NLD-8B can be fine-tuned with a LoRA on `o_proj` (rank 128, α 512) to improve drafter alignment. The LoRA weights are *not* in the base config - they ship as a separate adapter (Lecture 3.4 §3).

2. **The quadratic self-speculation expansion factor.** The quadratic mode expands the draft from $B = 32$ to $B(B+1)/2$ candidate positions per cycle. This is controlled by an inference-time flag `enc_config.self_spec_inference_mode = "quadratic"`, not a config field (Lecture 3.5).

3. **The DP-rank varying masking schedule.** When `dp_varying_mask_ratio: true`, each data-parallel rank uses a different mask ratio, scheduled by the rank index. The schedule itself (cosine, linear, …) is hardcoded in the trainer, not configurable here (Lecture 3.3 §4).

These are training/inference *behaviours* that you can't infer from `config.json` alone. The released model picks production-tested defaults; if you want to retrain from scratch with different choices, you'll need to read the trainer code (not shipped in the public model repo).

---

## 7. Loading and inspecting the model in Python

The following snippet loads NLD-8B and prints the architectural summary. Use this as a sanity check in your own environment:

```python
from transformers import AutoModelForCausalLM, AutoConfig

repo = "nvidia/Nemotron-Labs-Diffusion-8B"
cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
print(f"{cfg.hidden_size=}, {cfg.num_hidden_layers=}, "
      f"{cfg.num_attention_heads=}/{cfg.num_key_value_heads=}, "
      f"{cfg.intermediate_size=}, {cfg.vocab_size=}, "
      f"{cfg.block_size=}, {cfg.mask_token_id=}, "
      f"{cfg.dlm_paradigm=}, {cfg.ar_loss_weight=}, {cfg.dlm_loss_weight=}")

model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True,
                                              torch_dtype="bfloat16",
                                              device_map="auto")
total = sum(p.numel() for p in model.parameters())
print(f"Total params: {total / 1e9:.2f}B")
```

Expected output (on a single H100 or B200):

```
cfg.hidden_size=4096, cfg.num_hidden_layers=34, cfg.num_attention_heads=32/cfg.num_key_value_heads=8,
cfg.intermediate_size=14336, cfg.vocab_size=131072,
cfg.block_size=32, cfg.mask_token_id=100,
cfg.dlm_paradigm='bidirectional', cfg.ar_loss_weight=1.0, cfg.dlm_loss_weight=None
Total params: 8.49B
```

The 8.49B includes the lm_head; the tech report quotes 8.0B for the "transformer backbone + embedding" only.

---

## 8. Exercises

1. The text config has `dlm_loss_weight: null` and the VLM has `dlm_loss_weight: 0.5`. Trace the code path: where does the text model's actual default value get resolved? Hint: check `configuration_nemotron_labs_diffusion.py` and the `__init__` of `NemotronLabsDiffusionConfig`.

2. Compute the byte cost of NLD-8B in FP8 vs BF16. State the implied SOL speedup on B200 at $b = 1$.

3. The `lm_head` is not tied to the input embedding. Estimate the memory savings if it were tied (i.e., `tie_word_embeddings: true`). What's the inference-time trade-off?

4. The `block_size = 32` is hardcoded in the released config. Suppose you wanted to fine-tune NLD with $B = 64$. What other config fields (and trainer code) would you need to modify? Hint: check `max_position_embeddings` and the position-embedding tables.

5. Verify the parameter count in §5 by loading the model and grouping parameters by `name.split(".")[1]` (i.e., by layer type). Report the total per layer type.

Solutions to (1), (2) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 9. Further reading

- **HF Mistral docs** for the standard fields (`hidden_size`, `num_attention_heads`, …). https://huggingface.co/docs/transformers/main/en/model_doc/mistral
- **YaRN paper** (Peng et al., 2023). [arXiv:2309.00071](https://arxiv.org/abs/2309.00071). For `rope_type: yarn` and the rope parameters.
- **Pixtral 12B paper** (Mistral AI, 2024). https://mistral.ai/news/pixtral-12b/ - for `vision_config` defaults.
- **Nemotron-Labs-Diffusion Tech Report**, §3 (architecture) and §4 (training pipeline).
- **HuggingFace `auto_map`** and `trust_remote_code` docs: how externally-hosted modeling files are loaded.

Next, Lecture 3.2: read `set_attention_mode` and `compute_block_mask` line by line, and trace how the same forward serves AR, bidirectional, and the dual-stream `block_diff` modes.
