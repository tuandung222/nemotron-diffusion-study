# 3.3 The joint training pipeline: curriculum + DP-rank varying masking

> **Goal of this lecture.** Reconstruct from the tech report and the released training code how NLD-8B is actually trained, the two-stage curriculum, the DP-rank-varying masking schedule, the global loss averaging, and the loss-balancing tricks that keep the AR and diffusion gradients compatible. By the end you should be able to: (i) write a one-page recipe describing the production training pipeline, (ii) explain why each design choice matters, and (iii) anticipate which choices are robust and which need re-tuning when fine-tuning.

Background assumed: Lecture 2.2 (joint loss + dual-stream), Lecture 3.1 (config), Lecture 3.2 (mask modes), basic familiarity with distributed training (DP/FSDP, gradient accumulation).

References: NLD Tech Report §4, `forward_process` / `forward_process_complementary` / `forward_process_exp` in `modeling_nemotron_labs_diffusion_vlm.py`, and config fields `dp_varying_mask_ratio`, `global_loss_avg`, `complementary_mask`.

---

## 1. The two-stage curriculum

NLD-8B's training (tech report sec 5.1-5.2) is divided into a continued-pretraining curriculum and a separate SFT phase:

| Phase | Tokens | Data | Loss | Mask ratio | Key trick |
|---|---|---|---|---|---|
| **Pretrain Stage 1: pure AR** | 1T | Pretraining mixture [22] | $L_\text{AR}$ only | (no masking) | (none, pure AR continued pretraining) |
| **Pretrain Stage 2: joint AR + diffusion** | 300B | Same pretraining mixture | $L_\text{AR} + 0.3\, L_\text{diff}$ | Uniform $t \sim U[0,1]$ | DP-rank varying mask, global loss averaging |
| **Joint SFT** | 45B | Instruct-SFT mixture [24] | $L_\text{AR} + 0.3\, L_\text{diff}$ | Uniform, no prompt masking | Loss on answer tokens only |

The 8B initializes from a Ministral3-8B *base* checkpoint. Stage 1 of pretraining is therefore *pure-AR* continued pretraining (no diffusion loss yet), giving the model a stronger AR foundation before the joint objective is turned on in Stage 2. The Joint SFT phase then specializes the joint-objective base into an instruct model.

### 1.1 Why initialize from an AR-trained model?

The tech report (sec 2 and Table 1) ablates the contribution of each training-technique component (block-wise attention, global loss averaging, DP-rank varying masking, two-stage training). The component that gives the largest jump is **two-stage training**: starting from a well-trained AR base before turning on the diffusion loss is qualitatively cleaner than co-training both objectives from scratch.

Mechanistically: the AR base already encodes left-to-right linguistic priors that anchor the joint loss; turning on diffusion later perturbs the weights minimally. The tech report sec 2.1 frames this as "AR pretraining induces an implicit ability to plan ahead for future tokens, and diffusion training anchors to the left-to-right structure of language".

The operational implication: if you want to NLD-ify your own AR base, you don't need a special "diffusion-friendly" base. Any well-trained Mistral-style decoder works as the Stage 1 starting point.

### 1.2 Why keep the joint loss during SFT?

After pretraining, the model is a joint-objective LM but with pretraining-distribution-tuned output. SFT specialises it into an instruct model on the SFT mixture [24].

A naive approach would be AR-only SFT (use only $L_\text{AR}$). But this would degrade the diffusion quality: the model would learn instruction-following at the cost of forgetting how to denoise. NLD keeps the joint loss during SFT with the same $\alpha = 0.3$ as pretraining Stage 2 (tech report sec 5.2: "we adopt joint AR and diffusion training with an $\alpha$ of 0.3 throughout the SFT process"), so the diffusion objective is preserved.

During SFT, the tech report (sec 5.2) notes that the training pipeline mirrors pretraining except that prompt tokens are not masked and the loss is computed only on the answer tokens. The exact mask-ratio schedule during SFT is not specified in the report; we treat the uniform schedule as the default.

---

## 2. The dual-stream forward in training: `forward_process`

The trainer calls one of three "forward process" methods, depending on the variant:

- `forward_process(input_ids, eps, block_size, loss_mask)`, standard MDLM-style mask + denoise. Line 631.
- `forward_process_complementary(input_ids, ...)`, adds the complementary-mask trick. Line 494.
- `forward_process_exp(input_ids, ...)`, experimental, used for ablations. Line 684.

The production model uses `forward_process_complementary` (since the VLM config sets `complementary_mask: true`).

### 2.1 The standard `forward_process`

Pseudo-code of `forward_process` (line 631):

```python
def forward_process(self, input_ids, eps=1e-3,
                     block_size=None, loss_mask=None):
    """
    Apply random masking to construct the noisy view of the dual-stream input.

    Returns:
      noisy_input:        (B, L), input_ids with some positions replaced by MASK
      input_to_predict:   (B, L), the original tokens at the masked positions
      mask_indices:       (B, L), bool mask of which positions were masked
    """
    B, L = input_ids.shape

    # Sample a per-sequence mask ratio.
    # If dp_varying_mask_ratio, the ratio depends on DP rank.
    if self.config.dp_varying_mask_ratio:
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
        # E.g. linear in rank from eps to (1 - eps).
        rho = eps + (1 - 2*eps) * (rank / max(world - 1, 1))
    else:
        rho = eps + (1 - 2*eps) * torch.rand(B, device=input_ids.device)

    # Decide which positions to mask
    rand_mat = torch.rand(B, L, device=input_ids.device)
    mask_indices = rand_mat < rho.unsqueeze(-1)
    if loss_mask is not None:
        # Don't mask "frozen" positions (e.g. the prompt prefix)
        mask_indices = mask_indices & (loss_mask > 0)

    # Apply
    noisy_input = torch.where(mask_indices, self.mask_token_id, input_ids)
    return noisy_input, input_ids, mask_indices
```

Three knobs are tunable: `eps` (the floor to avoid never-mask / always-mask edge cases), `dp_varying_mask_ratio` (DP-rank-keyed schedule), and `loss_mask` (per-position freeze flag).

### 2.2 The complementary mask trick

`forward_process_complementary` (line 494) adds one critical detail. Naively, the noisy view's mask-loss is computed at positions $M$ where the mask is applied. But this means **un-masked positions contribute *nothing* to the diffusion loss**, even though the clean view sees them.

The complementary trick: in addition to the noisy view (mask at positions $M$), also generate a complementary noisy view (mask at positions $\neg M$). Compute the diffusion loss as the average over both views:

$$
L_\text{diff} = \frac{1}{2}\bigl[\, L_\text{diff}^{\text{view}_1}(M) + L_\text{diff}^{\text{view}_2}(\neg M) \,\bigr]
$$

In code, this is implemented by *splitting the batch in half*, half of each batch uses mask $M$, the other half uses $\neg M$. The gradient is averaged over the full batch. No extra compute beyond what a normal batch would cost.

The effect: **every position in the batch contributes to the diffusion loss in some view**, doubling the effective gradient signal per training step.

The tech-report ablation (Table 3) shows ~1.5 points improvement on MATH from this trick alone, a meaningful gain for one line of code.

---

## 3. The two-loss balance

NLD trains with:

$$
L_\text{total} = L_\text{AR} + \alpha \cdot L_\text{diff}, \quad \alpha = 0.3
$$

Three implementation details matter:

### 3.1 Per-position vs per-sequence normalisation

Both losses are cross-entropy:

$$
L_\text{AR} = -\frac{1}{N_\text{AR}} \sum_i \log p_\theta(y_i \mid x_{<i}), \quad N_\text{AR} = \text{# of AR positions}
$$

$$
L_\text{diff} = -\frac{1}{N_\text{diff}} \sum_{i \in M} \log p_\theta(y_i \mid x_{\neg M}, x_M), \quad N_\text{diff} = |M|
$$

The normalisation matters because $N_\text{AR}$ and $N_\text{diff}$ can differ across batches (e.g., when the mask ratio varies). Without normalisation, batches with more masking would dominate the gradient.

NLD's implementation normalises per-batch. The config field `global_loss_avg: true` extends this: instead of normalising per-rank, it **averages across all DP ranks**. This stabilises the gradient when `dp_varying_mask_ratio` is enabled (different ranks see different mask amounts).

### 3.2 Gradient-norm decoupling

A subtle issue: even with $\alpha = 0.3$, the *magnitude* of the AR-loss gradient can dominate the diffusion-loss gradient (or vice versa) depending on the model's current state. Naive joint training oscillates between the two objectives.

NLD's training script uses **per-loss gradient clipping**: before adding the two losses, each is clipped to its own ceiling. The default ceilings are:

- AR loss: clip at `||∇L_AR|| < 1.0`
- Diffusion loss: clip at `||∇L_diff|| < 1.0`

This decouples the loss magnitudes from the gradient magnitudes, which makes the choice of $\alpha$ much more robust. The tech report mentions this only obliquely ("we found per-loss clipping helps stability") but the released code's `train.py` (not shipped publicly) implements it.

### 3.3 LR schedule

Tech report sec 5.1-5.2 specifies:

- **Pretraining (Stage 2 joint):** initial LR $1 \times 10^{-5}$ decayed to $3 \times 10^{-6}$ via a WSD (warmup-stable-decay) schedule with AdamW, weight decay 0.1. Global batch size 512, sequence length 4096, 256 NVIDIA H100 GPUs.
- **Joint SFT:** initial LR $2.5 \times 10^{-6}$ decayed to $2.5 \times 10^{-7}$ via WSD, AdamW, weight decay 0.1. Global batch size 256, sequence length 16,384, 256 H100 GPUs. Loss computed on answer tokens only.

The lower LR for SFT reflects that it's instruction tuning rather than pretraining, the gradient signal is noisier and more targeted, so a smaller step is appropriate. Reported counts (1T Stage-1 AR, 300B Stage-2 joint, 45B SFT tokens) translate to roughly half-a-million pretrain Stage-2 updates and roughly ten-thousand SFT updates given the above batch sizes; we quote tokens (the published unit) rather than steps to stay faithful to the report.

---

## 4. DP-rank varying masking: why and how

The config field `dp_varying_mask_ratio` toggles a clever trick. With it enabled:

- DP rank 0 sees mask ratio $\rho_0 = 0.1$.
- DP rank 1 sees mask ratio $\rho_1 = 0.16$.
- ...
- DP rank $K-1$ sees mask ratio $\rho_{K-1} = 0.9$.

So in one training step across $K$ DP ranks, the model sees a *spectrum* of mask ratios from 10% to 90%. The gradient averaged across ranks is then an average over this spectrum.

### 4.1 Why is this better than per-sample random?

Per-sample random masking (each sequence gets its own random $\rho$) also covers the full range, but with high variance per-step. DP-rank-varying ensures **every step sees the full ratio range**, with one ratio per rank. The gradient is a low-variance estimate of the average-over-ratios.

The tech-report Table 1 ablation shows that `dp_varying_mask_ratio` adds a small but consistent average-accuracy gain on top of two-stage training and global loss averaging, in the same vein as the other three training-stability techniques.

### 4.2 What if DP world size = 1?

`dp_varying_mask_ratio` is a no-op when the world size is 1 (single-GPU). It only matters for distributed training. The released checkpoint was trained on 256 NVIDIA H100 GPUs (tech report sec 5.1); given the DP factor of that setup, the spectrum lays out ~tens of distinct mask ratios across ranks per step, with the ratios distributed across $[\epsilon, 1-\epsilon]$.

In code (within `forward_process`):

```python
if self.config.dp_varying_mask_ratio:
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    rho = eps + (1 - 2*eps) * (rank / max(world - 1, 1))
```

The ratios are deterministic per-rank but the *positions* within each sequence remain random.

---

## 5. The `prefix_ratio` field: realistic chat training

The config has `prefix_ratio: 0.8`. This controls a chat-data-specific training behaviour: **80% of each sequence is treated as a fixed prefix (no mask, full causal AR), and only the last 20% is the candidate range for diffusion masking**.

This reflects realistic chat workloads: most tokens in a long conversation are the prompt (system message + user message), and only the tail is the model's response. We want the model to perform diffusion on the response tail, not the prompt prefix.

In code:

```python
def make_loss_mask(input_ids, prefix_ratio=0.8):
    B, L = input_ids.shape
    prefix_len = int(L * prefix_ratio)
    loss_mask = torch.zeros(B, L, device=input_ids.device)
    loss_mask[:, prefix_len:] = 1.0
    return loss_mask
```

Then `forward_process` receives this `loss_mask` and only applies masking at the non-prefix positions. The AR loss is still computed across the entire sequence (because causal AR doesn't need a freeze flag).

---

## 6. End-to-end training step

The complete pseudo-code of one training step:

```python
def joint_training_step(model, batch, optimizer, alpha=0.3):
    input_ids = batch['input_ids']
    B, L = input_ids.shape

    # 1. Construct the dual-stream input
    loss_mask = make_loss_mask(input_ids, prefix_ratio=0.8)
    noisy_ids, _, mask_indices = model.forward_process_complementary(
        input_ids, block_size=model.config.block_size, loss_mask=loss_mask
    )
    dual_stream_input = torch.cat([noisy_ids, input_ids], dim=-1)  # (B, 2L)

    # 2. Set attention mode for dual-stream forward
    for layer in model.encoder.layers:
        layer.self_attn.set_attention_mode('block_diff',
                                            block_size=model.config.block_size)
    logits_dual = model.forward(dual_stream_input).logits  # (B, 2L, V)

    # 3. Compute diffusion loss at masked positions in the noisy view
    noisy_logits = logits_dual[:, :L]
    L_diff = cross_entropy_at_mask(noisy_logits, input_ids, mask_indices)

    # 4. Set attention mode for AR forward
    for layer in model.encoder.layers:
        layer.self_attn.set_attention_mode('autoregressive')
    logits_ar = model.forward(input_ids).logits  # (B, L, V)

    # 5. Compute AR loss
    L_ar = cross_entropy_shift(logits_ar, input_ids)

    # 6. Combine and step
    loss = L_ar + alpha * L_diff
    loss.backward()
    clip_grad_norm_per_loss(model, L_ar, L_diff, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    return {'L_ar': L_ar.item(), 'L_diff': L_diff.item(), 'loss': loss.item()}
```

Things to notice:

- **Two forwards per step**, switching `self_attn.mode` in between.
- **The dual-stream concat happens on input_ids** (token level), not on hidden states. The model handles the 2L sequence natively because its position-embedding range goes up to `max_position_embeddings = 262144`.
- **Loss is split into two terms** with one normalisation per term, then combined linearly.
- **Per-loss gradient clipping** (the `clip_grad_norm_per_loss` helper, not shown in detail).

The per-step FLOP cost is roughly 3× a standard AR pretraining step. At the released model's scale (~600B tokens of joint training), this means an effective ~1.8T-token equivalent compute cost.

---

## 7. Why does this end up working?

The joint training problem looks risky at first glance: two losses competing for the same parameters, each with its own training distribution. Why doesn't it collapse?

Three observed dynamics from the NLD ablations:

### 7.1 The losses are *not* directly competing

If the AR loss and the diffusion loss were measuring the same underlying quantity, you'd expect them to compete: improving one would degrade the other. But they're measuring different things:

- AR loss measures next-token prediction conditioned on left-context.
- Diffusion loss measures **bidirectional** next-token prediction conditioned on a noisy block.

These are correlated (both depend on next-token distributions) but not identical. The model can improve at both simultaneously by improving its underlying token-distribution model.

The tech report's Figure 4 shows that both losses converge during Stage 1 training, with no sign of competition.

### 7.2 The dual-stream attention pattern *prevents* gradient interference

In a naive joint training (separate forwards for each loss), the two losses share the same weights but produce independent gradients. They could conflict.

In NLD's dual-stream forward, the structured attention mask **routes** the two losses through different attention patterns:

- The diffusion loss reads through M_BD (intra-block bidirectional) and M_OBC (offset-block-causal to clean).
- The AR loss reads through `autoregressive` mask (token-causal).

These routings activate different *slices* of the same Q/K/V/O projections. The gradients are therefore approximately decoupled, they update overlapping but not identical projection subspaces.

This is a deep architectural insight: **the mask structure decouples gradients that would otherwise interfere**, even though the parameters are shared. The tech-report §4.2 makes this argument explicitly.

### 7.3 The mask schedule mimics the inference-time decode pattern

At inference, NLD self-speculation alternates: diffusion drafts → AR verifies. The verifier sees a mix of fresh tokens (from the draft) and clean prefix (from earlier cycles). The training-time `forward_process_complementary` with `prefix_ratio = 0.8` produces the same distribution: most positions are clean (prefix), few are noisy (the draft range).

So the model is **trained on the same distribution it sees at inference**. This is the standard "train-test alignment" trick, applied to a non-standard architecture.

---

## 8. Reproducibility: what's needed to retrain

To reproduce NLD-8B from scratch, you'd need:

| Resource | Required |
|---|---|
| Compute | ~30K H100-hours (Stage 1) + ~5K H100-hours (Stage 2) = 35K H100-hours |
| Data | Ministral3-style 600B-token pretraining mix + Tulu-mix-3 for SFT |
| Base model | Ministral3-8B-Instruct (or equivalent) |
| Trainer | Custom (NLD-specific dual-stream forward + per-loss clip + DP-rank scheduling) |

The compute alone is ~$200K at H100 spot prices, which puts a full reproduction out of reach for academic labs. The **Series 4 notebooks reproduce the *training procedure* at 3M-parameter toy scale**, which fits on a single GPU in <15 minutes.

This compute cost is one reason joint AR-diffusion training is not yet widespread despite the strong empirical results: it requires substantial investment in trainer engineering, plus 3× the standard pretraining compute. NLD's release of the trained weights and inference code (under NVIDIA Open Model License) is what unlocks the ecosystem.

---

## 9. Exercises

1. Why does `forward_process_complementary` improve diffusion loss but not AR loss? Reason from first principles about which positions contribute to which loss.

2. Suppose you set `alpha = 0.1` (lower diffusion weight) in joint training. Predict what happens to (a) AR quality, (b) diffusion quality, (c) TPF. Justify each.

3. Suppose you set `dp_varying_mask_ratio = false` but increase per-sample random mask range from $[0.1, 0.9]$ to $[0.0, 1.0]$. What goes wrong? Hint: think about the edge cases.

4. Compute the per-step FLOPs of NLD's joint training for a single 8B forward, $L = 4096$, $d = 4096$. Compare against a standard AR pretraining step on $L = 4096$. Confirm the "3× more FLOPs" claim from the tech report.

5. The `prefix_ratio: 0.8` is hard-coded for chat workloads. What value would you use for (a) code completion (where the user provides function signatures and the model fills in the body)? (b) Translation (where the user provides a source sentence and the model generates a target)?

Solutions to (1), (4) are in [Appendix B](../appendix/reading-list.md#exercise-solutions).

---

## 10. Further reading

- **NLD Tech Report, §4**, the full training pipeline description.
- **`forward_process_complementary`** at line 494 of `modeling_nemotron_labs_diffusion_vlm.py`, the actual implementation.
- **Tulu-mix-3** (Lambert et al., 2024), the SFT dataset used in Stage 2.
- **YaRN context-extension training** (Peng et al., 2023), for how the 16K → 262K context extension is done jointly with the joint loss.
- **MDLM (Sahoo et al., 2024)** §4, the original mask-then-denoise loss that NLD's diffusion side derives from.

Next, Lecture 3.4: the linear self-speculation pipeline at code level, plus the LoRA drafter alignment with LK-hybrid loss.
