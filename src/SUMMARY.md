# Summary

[Introduction](./00-introduction.md)

# Series 1 — Diffusion Language Models: Foundations

- [1.1 — From AR to non-AR: why parallel decoding matters](./01-foundations/01-ar-to-non-ar.md)
- [1.2 — Discrete diffusion 101: D3PM and absorbing-state diffusion](./01-foundations/02-discrete-diffusion-101.md)
- [1.3 — Masked Diffusion Models: SEDD, MDLM, LLaDA](./01-foundations/03-masked-diffusion-models.md)
- [1.4 — Block diffusion: from full-sequence to block-wise denoising](./01-foundations/04-block-diffusion.md)
- [1.5 — Sampling strategies for diffusion LMs](./01-foundations/05-sampling-strategies.md)
- [1.6 — AR vs Diffusion LMs: a head-to-head](./01-foundations/06-comparing-ar-vs-diffusion.md)

# Series 2 — Mixed AR–Diffusion Language Models

- [2.1 — Motivation: why unify AR and diffusion?](./02-mixed-ar-diffusion/01-motivation.md)
- [2.2 — Joint loss, dual-stream input, and the structured attention mask](./02-mixed-ar-diffusion/02-joint-loss-dual-stream.md)
- [2.3 — Self-speculation: drafting with diffusion, verifying with AR](./02-mixed-ar-diffusion/03-self-speculation.md)
- [2.4 — Comparisons: self-speculation vs MTP, EAGLE, Medusa, block-diffusion](./02-mixed-ar-diffusion/04-comparisons.md)
- [2.5 — Speed-of-Light analysis: the theoretical ceiling of parallel decoding](./02-mixed-ar-diffusion/05-speed-of-light.md)

# Series 3 — Nemotron-Labs-Diffusion deep dive

- [3.1 — Model card and config walkthrough](./03-nemotron-deep-dive/01-model-card-and-config.md)
- [3.2 — Tri-mode inference: set_attention_mode deep dive](./03-nemotron-deep-dive/02-tri-mode-inference.md)
- [3.3 — The joint training pipeline: curriculum + DP-rank varying masking](./03-nemotron-deep-dive/03-joint-training-pipeline.md)
- [3.4 — Linear self-speculation + LoRA drafter alignment](./03-nemotron-deep-dive/04-linear-self-spec-lora.md)
- [3.5 — Quadratic self-speculation + FlexAttention](./03-nemotron-deep-dive/05-quadratic-self-spec.md)
- [3.6 — VLM extension: Pixtral encoder + asymmetric dual-stream](./03-nemotron-deep-dive/06-vlm-extension.md)
- [3.7 — Benchmarks and deployment economics](./03-nemotron-deep-dive/07-benchmarks-and-deployment.md)

# Series 4 — Re-implementation lab *(planned)*

- [4.0 — How to run the labs](./04-reimplementation-lab/README.md)

---

[Appendix A — Glossary](./appendix/glossary.md)
[Appendix B — Reading list](./appendix/reading-list.md)
