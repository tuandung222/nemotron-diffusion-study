# nemotron-diffusion-study

A self-contained set of lecture notes and code labs for **diffusion language models**, **mixed AR–diffusion training**, and a **deep dive into NVIDIA's Nemotron-Labs-Diffusion (NLD)** model family — including the **VLM** variant.

> Audience: NLP / LLM data scientists who are already comfortable with autoregressive transformers, KV caches, and standard training pipelines. The notes assume that background and focus on what's *different* about diffusion LMs and how a single tri-mode model unifies AR, diffusion, and self-speculation.

## What's inside

The repo is structured as four progressive series:

| Series | Title | Format | Status |
|---|---|---|---|
| 1 | Diffusion Language Models — Foundations | Markdown (mdBook) | drafting |
| 2 | Mixed AR–Diffusion Language Models | Markdown (mdBook) | planned |
| 3 | Nemotron-Labs-Diffusion deep dive (incl. VLM) | Markdown (mdBook) | planned |
| 4 | Re-implementation lab — minimal PyTorch from scratch | Jupyter notebooks | planned |

The first three series are written as mdBook chapters and rendered to a navigable site (see "Reading the book" below). Series 4 lives under [`notebooks/`](notebooks/) as runnable `.ipynb` files.

## Reading the book

Online (rendered): https://tuandung222.github.io/nemotron-diffusion-study/

Locally:

```bash
cargo install mdbook
mdbook serve --open
```

Or just browse [`src/`](src/) on GitHub — every page is plain markdown.

## Citing the underlying work

Every lecture cites primary sources inline. The main reference for the model studied is:

> Fu, Y., Whalen, L., Garg, A., Wu, C., et al. *Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding*. NVIDIA Technical Report, 2026. [PDF](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_Diffusion_Tech_Report_v1.pdf)

## License

These notes are released under [CC-BY-4.0](LICENSE). Code snippets are MIT.
