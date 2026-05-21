# Series 3 — Nemotron-Labs-Diffusion deep dive *(planned)*

This series is not yet written. It will assume Series 1 and Series 2 as background.

Provisional table of contents:

1. **3.1 — Model card and `config.json` walkthrough**
   - Every field in `nvidia/Nemotron-Labs-Diffusion-VLM-8B/config.json` explained.
   - The relationship between NLD-8B (text-only) and NLD-VLM-8B.

2. **3.2 — `compute_block_mask`, `block_diff_mask`, `sbd_block_diff_mask`**
   - Reading the actual source of `modeling_nemotron_labs_diffusion_vlm.py`.
   - `set_attention_mode("ar" | "block_diff" | "bidirectional")`.

3. **3.3 — Training pipeline**
   - Stage 1: continual pretrain from Ministral3-8B-Instruct on 1T tokens.
   - Stage 2: instruction tuning + diffusion-specific examples.
   - Global loss averaging, DP-rank varying masking.

4. **3.4 — Linear self-speculation with LoRA**
   - The LK-hybrid drafter alignment loss.
   - Rank-128 LoRA on `o_proj`.

5. **3.5 — Quadratic self-speculation with FlexAttention**
   - The single-forward draft+verify pattern.
   - FlexAttention mask construction.

6. **3.6 — VLM extension**
   - Pixtral encoder (24 layers, 1024-hidden, patch 14, $\leq$ 1540×1540).
   - 2-layer MLP projector with 2×2 spatial merge.
   - The asymmetric dual-stream: vision tokens only on the clean side.
   - "Exact-merge" initialization from Ministral3-8B-Instruct.

7. **3.7 — Reading the benchmarks**
   - Per-device throughput (DGX Spark, H100, B200, GB200).
   - Per-mode TPF and acceptance length.
   - VLM-specific eval (MMMU, MathVista, MMBench).

This series will be written after Series 2.
