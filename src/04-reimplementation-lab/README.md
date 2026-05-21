# Series 4 - Re-implementation lab

This series consists of six progressive Jupyter notebooks that build a working **tri-mode VLM** in miniature. Each lab is self-contained, builds on the previous one, and runs end-to-end on CPU in under 15 minutes (GPU: < 1 minute).

## How to run

```bash
cd notebooks/
pip install -r requirements.txt   # torch, numpy
jupyter lab                        # or jupyter notebook
```

Open each notebook in order (`01-…` → `06-…`). All cells are executable as-is.

## Lab overview

| Lab | Title | Key concept | Model size |
|-----|-------|------------|------------|
| 4.1 | Minimal AR transformer | Baseline: causal attention, next-token CE loss | 0.8M |
| 4.2 | Diffusion masking + loss | MDLM-style mask-then-denoise; bidirectional attention | 0.8M |
| 4.3 | Dual-stream training | Joint AR + diffusion loss; structured 2L × 2L mask (M_BD ∪ M_OBC ∪ M_BC) | 0.8M |
| 4.4 | Tri-mode generate() | One model, three decoding modes; side-by-side TPF comparison | 0.8M |
| 4.5 | Self-speculation + KV cache | Cache clone/crop; shared-cache draft → verify cycle | 0.8M |
| 4.6 | VLM extension | Tiny vision encoder; asymmetric dual-stream; image classification | 0.2M |

## Design principles

- **Toy scale, real patterns.** Models are 0.2–0.8M parameters on a character-level corpus. The architecture and training code mirror NLD-8B step-for-step - only the scale differs.
- **No GPU required.** Every notebook runs on CPU. GPU just makes it faster.
- **Each lab has assertions.** Final cells verify shapes, mask structure, and intermediate results to catch bugs early.
- **Progressive complexity.** Lab 1 is a standard GPT. Lab 6 is a VLM with asymmetric dual-stream. Each lab adds one concept.

> **Note:** These are *re-implementation* exercises for understanding, not reproduction attempts. NLD-VLM-8B was trained on thousands of GPUs over multiple weeks.
