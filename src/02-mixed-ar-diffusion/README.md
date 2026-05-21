# Series 2 — Mixed AR–Diffusion Language Models *(planned)*

This series is not yet written. It will pick up where Series 1 ends.

Provisional table of contents:

1. **2.1 — Motivation: why unify AR and diffusion?**
   - The "no free lunch" argument from Series 1, lecture 1.6.
   - Empirical evidence that joint training doesn't hurt AR accuracy (NLD Table 2 ablation).
   - The new capabilities unlocked: self-speculation, infilling, mode switching.

2. **2.2 — Joint training objective, dual-stream input, structured attention masks**
   - The $\alpha$-weighted joint loss $\mathcal{L} = \mathcal{L}_{\text{AR}} + \alpha \cdot \mathcal{L}_{\text{diff}}$.
   - The dual-stream layout `[noisy_view ‖ clean_view]` of length $2L$.
   - The M_BD / M_OBC / M_BC mask decomposition with a worked $2L \times 2L$ example.
   - The implementation in `block_diff_mask` and `sbd_block_diff_mask`.

3. **2.3 — Self-speculation = DRAFT (diffusion) + VERIFY (AR) sharing the KV cache**
   - Linear self-speculation algorithm.
   - The LoRA-enhanced drafter (rank 128, $\alpha = 512$, target `o_proj`).
   - The LK-hybrid distribution-matching loss for drafter alignment.

4. **2.4 — Self-speculation vs MTP vs EAGLE-3 vs Medusa**
   - Architecture comparison: shared backbone vs separate drafter.
   - Acceptance length analysis.
   - When each wins.

5. **2.5 — Speed-of-light analysis at scale**
   - SOL upper bound, derived from the trained model's marginal accuracy.
   - How close NLD gets (~85%).
   - The "remaining gap" — what would close it.

This series will be written after the user reviews Series 1.
