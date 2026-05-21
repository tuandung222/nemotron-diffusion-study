# Notebooks — Series 4: Re-implementation lab

Six progressive Jupyter notebooks that build a working tri-mode VLM in miniature. Each notebook is self-contained and builds on the previous one.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Then open the notebooks in order.

## Lab list

| # | Notebook | Concept | Runtime (CPU) |
|---|---|---|---|
| 4.1 | [`01-minimal-ar-transformer.ipynb`](./01-minimal-ar-transformer.ipynb) | Baseline AR transformer | < 30 s |
| 4.2 | [`02-diffusion-masking-and-loss.ipynb`](./02-diffusion-masking-and-loss.ipynb) | MDLM-style diffusion training | < 90 s |
| 4.3 | [`03-dual-stream-training.ipynb`](./03-dual-stream-training.ipynb) | Joint AR + diffusion with structured mask | < 3 min |
| 4.4 | [`04-tri-mode-generate.ipynb`](./04-tri-mode-generate.ipynb) | `generate(mode=...)` for all three modes | < 3 min |
| 4.5 | [`05-self-speculation-shared-kv.ipynb`](./05-self-speculation-shared-kv.ipynb) | KV cache + shared-cache speculation | < 90 s |
| 4.6 | [`06-vlm-extension.ipynb`](./06-vlm-extension.ipynb) | Tiny VLM with asymmetric dual-stream | < 5 min |

## Dependencies

- Python 3.10+
- PyTorch 2.0+ (any version; we don't use FlexAttention since that requires PyTorch 2.5+)
- NumPy
- Jupyter

Install via `pip install -r requirements.txt`.

## Design notes

- **CPU-friendly.** Every notebook runs on CPU. GPU just speeds it up.
- **Toy scale.** Models are 0.2–0.8M parameters on character-level corpora — the architectures and training code mirror NLD-8B step-for-step, only the scale differs.
- **Assertions in every notebook.** Final cells verify mask structure, model shapes, and intermediate results.

For prose discussion, see the corresponding chapter in the mdBook (e.g. `src/04-reimplementation-lab/01-…md`).
