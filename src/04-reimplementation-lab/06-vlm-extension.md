# Lab 4.6 VLM extension with a tiny vision encoder

> **Notebook:** [`notebooks/06-vlm-extension.ipynb`](https://github.com/tuandung222/nemotron-diffusion-study/blob/main/notebooks/06-vlm-extension.ipynb)
>
> **Runtime:** CPU < 5 min · GPU < 1 min · **Params:** 0.22M

The capstone lab: add **vision conditioning** to the joint AR + diffusion model from Lab 4.3. We follow NLD-VLM's recipe at miniature scale.

## The task

A synthetic image classification problem:

- Input: a 2 × 2 image, either all-white, all-black, or checker pattern.
- Question: " what is it? "
- Expected answer: "all white", "all black", or "checker".

The model must use the image content to produce the correct text answer.

## What you build

### Vision encoder + projector

```python
class TinyVisionEncoder(nn.Module):
    def __init__(self, embed_dim=64, n_vision_tokens=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.proj1 = nn.Linear(32, embed_dim * 2)
        self.proj2 = nn.Linear(embed_dim * 2, embed_dim)
```

2 conv + 2 linear layers. Outputs **4 vision tokens** in the LM embedding space.

### `<IMG>` placeholder

A new vocabulary token `<IMG>`. When the text contains `<IMG>` × 4 in a row, the model replaces those embeddings with the 4 vision-encoder outputs. This is the same scheme NLD-VLM uses (Lecture 3.6 §6).

### Asymmetric dual-stream mask

The key difference from Lab 4.3: **vision tokens are only in the clean view**.

Layout: `[noisy_text (L_txt) | vision (N_vis) | clean_text (L_txt)]` → total `2*L_txt + N_vis`.

Rules (`make_asym_dual_mask`):
- Noisy text → noisy text in same block (M_BD).
- Noisy text → all vision tokens (vision is always-attended-to).
- Noisy text → clean text in previous blocks (M_OBC).
- Vision ↔ vision (full attention within vision tokens).
- **Vision → text** (any) **is forbidden**, vision is encoded independently of text context.
- Clean text → vision (yes, text uses vision context).
- Clean text → clean text causal by block (M_BC).
- Clean text → noisy text never.

The mask matrix at `L_txt=4, N_vis=2, block_size=2`:

```
Layout: [noisy_text(4) | vision(2) | clean_text(4)]
[[1 1 0 0 1 1 0 0 0 0]    ← noisy block 0
 [1 1 0 0 1 1 0 0 0 0]
 [0 0 1 1 1 1 1 1 0 0]    ← noisy block 1, sees vision + clean block 0
 [0 0 1 1 1 1 1 1 0 0]
 [0 0 0 0 1 1 0 0 0 0]    ← vision tokens, see only other vision tokens
 [0 0 0 0 1 1 0 0 0 0]
 [0 0 0 0 1 1 1 1 0 0]    ← clean block 0, sees vision + itself
 [0 0 0 0 1 1 1 1 0 0]
 [0 0 0 0 1 1 1 1 1 1]    ← clean block 1, sees vision + clean blocks ≤ 1
 [0 0 0 0 1 1 1 1 1 1]]
```

Note rows 4-5 (vision): they only attend to other vision tokens. Vision → text is False because the vision encoder is independent.

### Joint training step

Same template as Lab 4.3, but the diffusion forward inserts vision tokens between the noisy and clean halves:

```python
dual = torch.cat([noisy_text, vision_placeholder, clean_text], dim=-1)
asym_mask = make_asym_dual_mask(L_txt, N_vis, block_size, device)
logits = model(dual, img, asym_mask)
```

The AR loss is text-only (vision tokens are skipped for AR, pedagogical simplification; NLD's AR uses vision too).

## What you measure

- Joint loss curve: both AR and diffusion losses drop together.
- **Classification accuracy** on the synthetic task: ~85-90% (3-char prefix match) after 400 steps.
- Sample predictions per pattern.

## Sanity assertions

The notebook verifies:
- Mask shape: `(2*L_txt + N_vis, 2*L_txt + N_vis)`.
- Vision → text always False (rule 5).
- Vision ↔ vision always True.
- Clean text → vision always True.
- Noisy text → vision always True.
- Noisy text across blocks always False.
- Final forward shape: `(B, 2*L_txt + N_vis, vocab_size)`.

## What's next (beyond this series)

The labs end here. To scale up:

1. Replace `TinyGPT` with a real Llama-3 / Ministral3-style transformer (8B, RoPE/YaRN, GQA, RMSNorm).
2. Replace `TinyVisionEncoder` with Pixtral 12B's encoder (24 layers, 1024 hid, patch 14, dynamic resolution).
3. Replace ad-hoc mask construction with PyTorch 2.5 `FlexAttention.create_block_mask` (Lecture 3.5).
4. Train on real data: Tulu-mix-3 for text-only, LLaVA-mix for VLM.
5. Add the LoRA drafter alignment (Lecture 3.4): rank 128, α 512, target `o_proj` only.

Each step is a 10-100× scale-up, but the architectural patterns are identical to what you built here.

## Prerequisites

- Labs 4.1–4.5.
- Lectures 3.6 (VLM extension).
