# Lab 4.5 - Self-speculation with shared KV cache

> **Notebook:** [`notebooks/05-self-speculation-shared-kv.ipynb`](https://github.com/tuandung222/nemotron-diffusion-study/blob/main/notebooks/05-self-speculation-shared-kv.ipynb)
>
> **Runtime:** CPU < 90 s · GPU < 15 s · **Params:** 0.83M

This lab implements the **KV cache** machinery that makes self-speculation actually fast in wall-clock. We introduce `clone()` and `crop()` operations - the two primitives NLD uses (Lecture 3.4 §3).

## What you build

### LayerCache and ModelCache

```python
class LayerCache:
    def __init__(self):
        self.k = None  # (B, H, T, D)
        self.v = None

    def append(self, new_k, new_v):
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)

    def clone(self):
        '''Shallow copy - shares underlying storage. Subsequent torch.cat
        creates a new tensor, so the original is unaffected (copy-on-extend).'''
        new = LayerCache()
        new.k, new.v = self.k, self.v
        return new

    def crop(self, length):
        self.k = self.k[:, :, :length, :].contiguous()
        self.v = self.v[:, :, :length, :].contiguous()
```

`ModelCache` is just `n_layer` copies of `LayerCache` plumbed through every block.

### Cached AR generation

We compare AR generation with and without the cache:

| mode | wall-clock (50 tokens) | tokens/s |
|---|---|---|
| no-cache | 0.15 s | 335 |
| with-cache | 0.11 s | 459 |
| **speedup** | **1.37×** | |

The output matches exactly between the two - verifies cache correctness. Speedup is modest because the model is small and the prompt is short; at 8B params and 4K-token prompts, the speedup is 20–50×.

### Shared-cache speculation cycle

The cache mechanics:

1. **Prefill** the prompt → `base_cache.length == L_p`.
2. **Draft** K tokens by running greedy AR on `draft_cache = base_cache.clone()`. The clone shares storage; subsequent appends create new tensors, leaving `base_cache` untouched.
3. **Verify** by re-running the same K tokens through `verify_cache = base_cache.clone()` with the K drafts as input.
4. **Accept** longest prefix where `drafted[i] == verify_argmax[i]`.
5. **Commit**: `verify_cache.crop(L_p + n_accept)`; promote `verify_cache` to `base_cache`.

The notebook reports an average of **4.1 tokens committed per cycle** (out of K=4), demonstrating the mechanism works.

## What you measure

- AR cached vs uncached speedup.
- Output equivalence between cached and uncached.
- Cache `clone()` doesn't mutate the original.
- Cache `crop()` truncates correctly.
- Speculation tokens-per-cycle.

## Sanity assertions

```python
cache = model.new_cache()
# ... append 10 tokens via forward ...
cache2 = cache.clone()
# ... append 3 tokens to cache2 ...
assert cache.length == 10            # original unaffected
assert cache2.length == 13
cache2.crop(8)
assert cache2.length == 8
```

All must pass.

## What the next lab changes

- We add a **vision encoder** and **`<IMG>` placeholder mechanism**.
- We build the **asymmetric dual-stream mask** (vision in clean view only).

## Prerequisites

- Labs 4.1–4.4.
- Lectures 3.4 (linear self-spec + cache mechanics).
