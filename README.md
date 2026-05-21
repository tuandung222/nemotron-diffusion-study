# nemotron-diffusion-study

A self-contained set of lecture notes and code labs for **diffusion language models**, **mixed AR–diffusion training**, and a **deep dive into NVIDIA's Nemotron-Labs-Diffusion (NLD)** model family — including the **VLM** variant.

> Audience: NLP / LLM data scientists who are already comfortable with autoregressive transformers, KV caches, and standard training pipelines. The notes assume that background and focus on what's *different* about diffusion LMs and how a single tri-mode model unifies AR, diffusion, and self-speculation.

**Online (rendered with KaTeX):** <https://tuandung222.github.io/nemotron-diffusion-study/>

## What's inside

The repo is structured as four progressive series. **All four series are complete** (24 lectures + 2 appendices + 6 runnable notebooks ≈ 59,400 words of prose + ~2,500 lines of PyTorch).

| Series | Title | Format | Lectures | Words |
|---|---|---|---|---|
| **1** | [Diffusion Language Models — Foundations](src/01-foundations/) | mdBook | 6 lectures + 2 appendices | ~19,600 |
| **2** | [Mixed AR–Diffusion Language Models](src/02-mixed-ar-diffusion/) | mdBook | 5 lectures | ~14,500 |
| **3** | [Nemotron-Labs-Diffusion deep dive (incl. VLM)](src/03-nemotron-deep-dive/) | mdBook | 7 lectures | ~22,000 |
| **4** | [Re-implementation lab — minimal PyTorch from scratch](notebooks/) | Jupyter + mdBook | 6 notebooks + 7 chapter pages | ~3,300 prose + ~2,500 LOC |

The first three series are written as mdBook chapters and rendered to a navigable site (see "Reading the book" below). Series 4 lives under [`notebooks/`](notebooks/) as runnable `.ipynb` files; each one runs end-to-end on CPU in under 15 minutes.

### Series 1 — Foundations

Builds the math of diffusion LMs from scratch, assuming you already know AR transformers and KV caches.

- [1.1 — From AR to non-AR: why parallel decoding matters](https://tuandung222.github.io/nemotron-diffusion-study/01-foundations/01-ar-to-non-ar.html)
- [1.2 — Discrete diffusion 101: D3PM and absorbing-state diffusion](https://tuandung222.github.io/nemotron-diffusion-study/01-foundations/02-discrete-diffusion-101.html)
- [1.3 — Masked Diffusion Models: SEDD, MDLM, LLaDA](https://tuandung222.github.io/nemotron-diffusion-study/01-foundations/03-masked-diffusion-models.html)
- [1.4 — Block diffusion: from full-sequence to block-wise denoising](https://tuandung222.github.io/nemotron-diffusion-study/01-foundations/04-block-diffusion.html)
- [1.5 — Sampling strategies for diffusion LMs](https://tuandung222.github.io/nemotron-diffusion-study/01-foundations/05-sampling-strategies.html)
- [1.6 — AR vs Diffusion LMs: a head-to-head](https://tuandung222.github.io/nemotron-diffusion-study/01-foundations/06-comparing-ar-vs-diffusion.html)
- [Appendix A — Glossary](https://tuandung222.github.io/nemotron-diffusion-study/appendix/glossary.html)
- [Appendix B — Reading list + exercise solutions](https://tuandung222.github.io/nemotron-diffusion-study/appendix/reading-list.html)

### Series 2 — Mixed AR–Diffusion Language Models

Derives the joint training objective, dual-stream input layout, and the structured `M_BD ∪ M_OBC ∪ M_BC` attention mask. Introduces self-speculation and the Speed-of-Light upper bound.

- [2.1 — Motivation: why unify AR and diffusion?](https://tuandung222.github.io/nemotron-diffusion-study/02-mixed-ar-diffusion/01-motivation.html)
- [2.2 — Joint loss, dual-stream input, and the structured attention mask](https://tuandung222.github.io/nemotron-diffusion-study/02-mixed-ar-diffusion/02-joint-loss-dual-stream.html)
- [2.3 — Self-speculation: drafting with diffusion, verifying with AR](https://tuandung222.github.io/nemotron-diffusion-study/02-mixed-ar-diffusion/03-self-speculation.html)
- [2.4 — Comparisons: self-speculation vs MTP, EAGLE, Medusa, block-diffusion](https://tuandung222.github.io/nemotron-diffusion-study/02-mixed-ar-diffusion/04-comparisons.html)
- [2.5 — Speed-of-Light analysis: the theoretical ceiling of parallel decoding](https://tuandung222.github.io/nemotron-diffusion-study/02-mixed-ar-diffusion/05-speed-of-light.html)

### Series 3 — Nemotron-Labs-Diffusion deep dive

Code-level walkthrough of the NVIDIA NLD model family: `config.json`, `set_attention_mode`, `compute_block_mask`, joint training pipeline, linear/quadratic self-speculation with FlexAttention, the Pixtral-based VLM extension, and deployment economics.

- [3.1 — Model card and config walkthrough](https://tuandung222.github.io/nemotron-diffusion-study/03-nemotron-deep-dive/01-model-card-and-config.html)
- [3.2 — Tri-mode inference: `set_attention_mode` deep dive](https://tuandung222.github.io/nemotron-diffusion-study/03-nemotron-deep-dive/02-tri-mode-inference.html)
- [3.3 — The joint training pipeline: curriculum + DP-rank varying masking](https://tuandung222.github.io/nemotron-diffusion-study/03-nemotron-deep-dive/03-joint-training-pipeline.html)
- [3.4 — Linear self-speculation + LoRA drafter alignment](https://tuandung222.github.io/nemotron-diffusion-study/03-nemotron-deep-dive/04-linear-self-spec-lora.html)
- [3.5 — Quadratic self-speculation + FlexAttention](https://tuandung222.github.io/nemotron-diffusion-study/03-nemotron-deep-dive/05-quadratic-self-spec-flexattention.html)
- [3.6 — VLM extension: Pixtral encoder + asymmetric dual-stream](https://tuandung222.github.io/nemotron-diffusion-study/03-nemotron-deep-dive/06-vlm-extension.html)
- [3.7 — Benchmarks and deployment economics](https://tuandung222.github.io/nemotron-diffusion-study/03-nemotron-deep-dive/07-benchmarks-and-deployment.html)

### Series 4 — Re-implementation lab (Jupyter notebooks)

Six progressive notebooks that build a tri-mode (AR / block-diffusion / self-speculation) language model from scratch in pure PyTorch, then extend it to a VLM. Each notebook runs on CPU in <15 minutes, has pre-saved outputs, and ends with assertions consumed by the next lab.

| Notebook | Topic | Params | CPU runtime |
|---|---|---|---|
| [4.1](notebooks/01-minimal-ar-transformer.ipynb) | Minimal AR transformer | 0.81M | ~30 min |
| [4.2](notebooks/02-diffusion-masking-and-loss.ipynb) | Add diffusion masking + MDLM loss | 0.81M | ~40 min |
| [4.3](notebooks/03-dual-stream-training.ipynb) | Dual-stream training + 2L×2L structured attention mask | 0.83M | ~3 min |
| [4.4](notebooks/04-tri-mode-generate.ipynb) | Tri-mode `generate()` dispatcher | 0.83M | ~3 min |
| [4.5](notebooks/05-self-speculation-shared-kv.ipynb) | Self-speculation with shared KV cache | 0.83M | ~90 s |
| [4.6](notebooks/06-vlm-extension.ipynb) | VLM extension with tiny vision encoder | 0.22M | ~5 min |

Companion chapter pages with prose write-up are at [`src/04-reimplementation-lab/`](src/04-reimplementation-lab/) and on the [rendered site](https://tuandung222.github.io/nemotron-diffusion-study/04-reimplementation-lab/index.html).

## Reading the book

**Online (rendered with KaTeX):** <https://tuandung222.github.io/nemotron-diffusion-study/>

**Locally:**

```bash
cargo install mdbook mdbook-katex
git clone https://github.com/tuandung222/nemotron-diffusion-study.git
cd nemotron-diffusion-study
mdbook serve --open
```

Or just browse [`src/`](src/) on GitHub — every page is plain markdown (math will appear as raw `$...$` on GitHub since GitHub's Markdown renderer doesn't run our KaTeX preprocessor).

## Running the notebooks

```bash
cd notebooks/
pip install -r requirements.txt
jupyter lab
```

Each notebook is self-contained and runs CPU-only. They use a small Tiny Shakespeare corpus and are designed to fit in <15 minutes per lab. See [`notebooks/README.md`](notebooks/README.md) for details.

## Notes on math rendering

The book uses [`mdbook-katex`](https://github.com/lzanini/mdbook-katex) as a preprocessor (configured in `book.toml`) so inline math with underscores like `$\mathbf{x}_t$` renders correctly. The preprocessor runs **before** mdBook's Markdown parser, which prevents `_` characters inside math from being interpreted as italic delimiters. If you notice raw `$...$` on the rendered site, **hard-reload the page** (Ctrl+Shift+R) — your browser is probably caching an older MathJax-based build.

## Citing the underlying work

Every lecture cites primary sources inline. The main reference for the model studied is:

> Fu, Y., Whalen, L., Garg, A., Wu, C., et al. *Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding*. NVIDIA Technical Report, 2026. [PDF](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_Diffusion_Tech_Report_v1.pdf)

Model cards on Hugging Face:

- [`nvidia/Nemotron-Labs-Diffusion-8B`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B)
- [`nvidia/Nemotron-Labs-Diffusion-VLM-8B`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B)

## License

These notes are released under [CC-BY-4.0](LICENSE). Code snippets are MIT.
