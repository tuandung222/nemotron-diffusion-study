# Series 4 — Re-implementation lab *(planned)*

This series is not yet written. It will consist of progressive Jupyter notebooks that build a working tri-mode VLM in miniature.

Provisional notebook list:

1. **Lab 1 — Minimal AR transformer (review).** Small GPT-style model on TinyStories. 30 minutes of training to a baseline.

2. **Lab 2 — MDLM training.** Convert Lab 1's model to a masked diffusion LM by changing the loss and the attention mask. Reproduce the MDLM-style training and compare perplexity to Lab 1.

3. **Lab 3 — Block diffusion.** Add the `compute_block_mask` machinery. Inference loop with confidence-threshold sampling.

4. **Lab 4 — Joint AR + diffusion (single model, dual-stream).** Implement the M_BD / M_OBC / M_BC mask construction and the dual-stream loss. Verify that AR accuracy is preserved while diffusion TPF $> 1$.

5. **Lab 5 — Linear self-speculation.** Implement DRAFT (diffusion) + VERIFY (AR) on shared backbone weights. Measure acceptance length and TPF.

6. **Lab 6 — Vision encoder integration.** Add a tiny CNN-based image encoder and projector. Asymmetric dual-stream where vision tokens are only on the clean side.

7. **Lab 7 — End-to-end mini-NLD-VLM.** Put it all together. Train on a small VQA-style dataset.

All notebooks target a single A100/L40S GPU at ≤ 4 GB VRAM so they run in Colab.

Notebooks will be added to `notebooks/` after the corresponding lectures in Series 2/3 are written.

> Note: these are *re-implementation* exercises for understanding, not reproduction attempts. NLD-VLM-8B was trained on thousands of GPUs over multiple weeks. The labs use models with $\sim$1M parameters and toy datasets, but follow the same architectural pattern step-for-step.
