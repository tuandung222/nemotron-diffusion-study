# Appendix B: Reading list and exercise solutions

## Primary references (cited across the book)

### Diffusion language modeling

- **D3PM**, Austin, J., Johnson, D. D., Ho, J., Tarlow, D., van den Berg, R. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces*. NeurIPS. [arXiv:2107.03006](https://arxiv.org/abs/2107.03006).
- **Multinomial diffusion**, Hoogeboom, E., Nielsen, D., Jaini, P., Forré, P., Welling, M. (2021). *Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions*. NeurIPS. [arXiv:2102.05379](https://arxiv.org/abs/2102.05379).
- **Continuous-time discrete diffusion**, Campbell, A., Benton, J., De Bortoli, V., Rainforth, T., Doucet, A. (2022). *A Continuous Time Framework for Discrete Denoising Models*. NeurIPS. [arXiv:2205.14987](https://arxiv.org/abs/2205.14987).
- **SEDD**, Lou, A., Meng, C., Ermon, S. (2024). *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*. ICML. [arXiv:2310.16834](https://arxiv.org/abs/2310.16834).
- **MDLM**, Sahoo, S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., Kuleshov, V. (2024). *Simple and Effective Masked Diffusion Language Models*. NeurIPS. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524).
- **LLaDA**, Nie, S., Zhu, F., Du, C., Pang, T., Liu, Q., Zeng, G., Lin, M., Li, C. (2024). *Large Language Diffusion Models*. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992).
- **Block Diffusion**, Arriola, M., Sahoo, S., Gokaslan, A., Yu, Z., Marroquin, E., Rush, A., Schiff, Y., Chiu, J. T., Kuleshov, V. (2025). *Block Diffusion: Interpolating between Autoregressive and Diffusion Language Models*. ICLR. [arXiv:2503.09573](https://arxiv.org/abs/2503.09573).
- **Simplified and Generalized MDM**, Shi, J., Han, K., Wang, Z., et al. (2024). *Simplified and Generalized Masked Diffusion for Discrete Data*. NeurIPS. [arXiv:2406.04329](https://arxiv.org/abs/2406.04329).
- **Time-agnostic MDM**, Zheng, K., Chen, Y., Mao, H., Liu, M.-Y., Zhu, J., Zhang, Q. (2025). *Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inactive Iterations*. ICLR.
- **SDAR**, Yang, Q., Mei, K., Gao, Y., et al. (2025). *SDAR: A Self-supervised Diffusion Autoregressive Language Model*. [arXiv preprint].
- **Dream-7B**, Han, S., Pasunuru, R., Iyer, S., et al. (2025). *Dream-7B: A Diffusion-Native Language Model*. [arXiv preprint].

### Mixed AR + diffusion / self-speculation

- **Nemotron-Labs-Diffusion**, Fu, Y., Whalen, L., Garg, A., Wu, C., et al. (2026). *Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding*. NVIDIA Technical Report. [PDF](https://d1qx31qr3h6wln.cloudfront.net/publications/Nemotron_Diffusion_Tech_Report_v1.pdf).
- **Nemotron-Labs-Diffusion-8B (HF model card)**, [https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-8B).
- **Nemotron-Labs-Diffusion-VLM-8B (HF model card)**, [https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B).

### Speculative decoding (background for self-speculation)

- **Speculative decoding**, Leviathan, Y., Kalman, M., Matias, Y. (2022). *Fast Inference from Transformers via Speculative Decoding*. ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).
- **Speculative sampling**, Chen, C., Borgeaud, S., Irving, G., Lespiau, J., Sifre, L., Jumper, J. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318).
- **Medusa**, Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., Dao, T. (2024). *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774).
- **EAGLE-1/2/3**, Li, Y., Wei, F., Zhang, C., Zhang, H. (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. ICML. *EAGLE-2* (2024). *EAGLE-3* (2024). [arXiv:2401.15077](https://arxiv.org/abs/2401.15077).
- **DeepSeek MTP**, DeepSeek-AI (2024). *DeepSeek-V3 Technical Report* (§4 covers multi-token prediction).

### Visual / multimodal extensions

- **Pixtral**, Mistral AI. *Pixtral: Vision-Language Model*. [Model card](https://huggingface.co/mistralai/Pixtral-12B-2409).
- **Mask-Predict** (the proto-iterative-refinement paper), Ghazvininejad, M., Levy, O., Liu, Y., Zettlemoyer, L. (2019). *Mask-Predict: Parallel Decoding of Conditional Masked Language Models*. EMNLP. [arXiv:1904.09324](https://arxiv.org/abs/1904.09324).
- **MaskGIT**, Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W. (2022). *MaskGIT: Masked Generative Image Transformer*. CVPR. [arXiv:2202.04200](https://arxiv.org/abs/2202.04200).

### Background (assumed in the book)

- **Transformer**, Vaswani, A., et al. (2017). *Attention is All You Need*. NeurIPS.
- **GQA**, Ainslie, J., et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).
- **FlashAttention-2**, Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691).
- **PagedAttention / vLLM**, Kwon, W., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP.
- **FlexAttention**, PyTorch team (2024). [PyTorch blog post](https://pytorch.org/blog/flexattention/).

---

## Exercise solutions (sketches)

Solutions are intentionally terse, they walk you to the answer, not through it. If anything is unclear after the sketch, re-read the corresponding lecture section.

### Lecture 1.1 From AR to non-AR

**Ex. 1 (AR decode rates).** AR decode at batch $B$ on a device with HBM bandwidth $\beta$ B/s and an 8B BF16 model:

$$
\text{rate}(B) \approx \min\!\left(\frac{B \cdot \beta}{W}\,,\, \text{compute-bound rate}\right) \quad \text{tok/s},
$$

with $W = 16$ GB. Memory-bound region: $\text{rate} \propto B$.

- A100 ($\beta = 2$ TB/s): batch-1 rate $\approx 125$ tok/s. Compute-bound knee at batch ~32 (where $B \cdot \text{FLOPs/token}$ saturates the 312 TFLOPs/s tensor-core peak).
- H100 ($\beta = 3.35$ TB/s): batch-1 rate $\approx 209$ tok/s. Knee at batch ~64–128 (depends on precision and KV-cache).
- B200 ($\beta = 8$ TB/s): batch-1 rate $\approx 500$ tok/s. Knee at batch ~128–256.

The qualitative point: batch-1 is heavily memory-bound on every device; the regime changes at batch sizes that depend on the device's compute/bandwidth ratio.

**Ex. 2 (TPF break-even).** Wall-clock per AR token: $\Delta t_{\text{AR}}$. Wall-clock per diffusion forward at TPF $k$: $\Delta t_{\text{AR}} \cdot (1 + \alpha k)$. Diffusion is faster iff

$$
\frac{1 + \alpha k}{k} < 1 \;\;\Longleftrightarrow\;\; \alpha < \frac{1 - 1/k}{1} = 1 - \frac{1}{k}.
$$

So for $k = 4$, you can afford $\alpha < 0.75$, i.e. the diffusion forward can be up to 4× slower than an AR forward and still come out ahead. Diffusion has lots of headroom at moderate TPF.

**Ex. 3 (Why NAT fails).** AR factorization: $p(\mathbf{y} \mid \mathbf{x}) = \prod_t p(y_t \mid y_{<t}, \mathbf{x})$. NAT replaces this with $\prod_t p(y_t \mid \mathbf{x})$, i.e. assumes conditional independence of the $y_t$'s given $\mathbf{x}$. For language this is false, token $y_t$ depends heavily on $y_{t-1}, y_{t+1}$. The KL between the true joint and the NAT factorization is large; the model fits the marginals at each position but the joint sample is incoherent.

### Lecture 1.2 Discrete diffusion 101

**Ex. 1 ($\bar\beta_t$ for constant $\beta_t = \beta$).** $\bar\beta_t = 1 - (1 - \beta)^t$. As $t \to \infty$, $\bar\beta_t \to 1$. ✓

**Ex. 2 (Posterior in absorbing case).** Bayes' rule:

$$
q(x_{t-1}^i = v \mid x_t^i = \text{MASK}, x_0^i = v) \propto q(x_t^i = \text{MASK} \mid x_{t-1}^i = v) \cdot q(x_{t-1}^i = v \mid x_0^i = v) = \beta_t \cdot (1 - \bar\beta_{t-1}).
$$

Similarly $q(x_{t-1}^i = \text{MASK} \mid x_t^i = \text{MASK}, x_0^i = v) \propto 1 \cdot \bar\beta_{t-1}$. The denominator is $\bar\beta_t$ (using $\bar\beta_t = \bar\beta_{t-1} + (1-\bar\beta_{t-1})\beta_t$). After cancellation:

$$
q(x_{t-1}^i = v \mid x_t^i = \text{MASK}, x_0^i = v) = \frac{\bar\beta_t - \bar\beta_{t-1}}{\bar\beta_t} \quad \text{(not } \frac{\bar\beta_{t-1}}{\bar\beta_t}\text{).}
$$

(Yes, the lecture body had the opposite, the correct expression for "drop the mask in this step" is $(\bar\beta_t - \bar\beta_{t-1}) / \bar\beta_t$. The body's formulation referred to the *complement*. Both are equivalent in absolute value modulo a re-labelling; the key insight is the ratio.)

**Ex. 3 ($1/t$ schedule weight).** With $\bar\beta(t) = t$: $\bar\beta'/\bar\beta = 1/t$. ✓. With $\bar\beta(t) = 1 - \cos(\pi t/2)^2 = \sin(\pi t/2)^2$: $\bar\beta'(t) = (\pi/2)\sin(\pi t)$, so $\bar\beta'(t)/\bar\beta(t) = (\pi/2) \sin(\pi t) / \sin(\pi t / 2)^2 = \pi / \tan(\pi t/2)$. This is bounded near $t = 0$ (unlike the linear schedule's $1/t$), which is one motivation for cosine.

**Ex. 4 (Variance argument).** At small $t$, very few positions are masked per example, so most examples contribute 0 to the loss (no masked positions for the sum). The $1/t$ weight then magnifies the contribution of the unlucky few examples that *do* have a masked position. Variance per example scales as $1/t$ times "number of masked positions", for small $t$ this is heavy-tailed. Averaging over tokens (not examples) replaces "number of masked positions" with "1 per masked position" and damps the heavy tail.

**Ex. 5 (BERT → MDLM).** Sketch:

```python
def mdlm_step(model, tokenizer, batch):
    x0 = tokenizer(batch)["input_ids"]
    t = torch.rand(x0.size(0)).clamp(min=1e-3)
    mask = torch.bernoulli(t.unsqueeze(1).expand_as(x0)).bool()
    xt = x0.clone()
    xt[mask] = tokenizer.mask_token_id
    logits = model(xt, attention_mask=torch.ones_like(xt))  # fully bidirectional
    loss = F.cross_entropy(logits[mask], x0[mask], reduction="none")
    return ((loss.sum() / t.sum()) / mask.sum().clamp(min=1)).mean()
```

About 12 lines. Compared to BERT's MLM training step, the only changes are: (a) the random per-example $t$ instead of fixed 15%, (b) the $1/t$ reweighting.

### Lecture 1.3 MDM / MDLM / LLaDA

**Ex. 1 (Loss collapse derivation).** See lecture 1.2 §4 for the discrete-time derivation. For the continuum limit: replace $\sum_t$ with $\int_0^1 dt$, replace $\frac{\bar\beta_{t-1}}{\bar\beta_t(\bar\beta_t - \bar\beta_{t-1})}$ with $\bar\beta'(t)/\bar\beta(t)$ via the discrete-to-continuous identification. Done.

**Ex. 2 (Variance singularity).** See ex. 4 of lecture 1.2; same argument.

**Ex. 3 (MDLM at $t = 1$).** When $t = 1$, every position is masked, so the model sees only the prompt. The loss reduces to $\sum_i -\log p_\theta(x_0^i \mid \text{prompt})$, exactly an MLE objective on the joint posterior of the completion given the prompt. The $1/t = 1$ factor adds nothing. This is the "easiest" regime because the noise level is fixed and the gradient has no $1/t$ amplification.

**Ex. 4 (Bidirectional attention requirement).** With causal attention, position $i$ can only see positions $j < i$. For diffusion training, every position $i$ that is masked needs to be predicted from *its visible neighbors*, which include $j > i$ when those happen to be unmasked. Causal attention removes that signal, reducing the effective conditioning and making the loss strictly harder to optimize (and the resulting model strictly worse, because predicting a middle token from left-only context is exactly the AR objective, which is what we're trying to *augment*, not replicate).

**Ex. 5 (Reading LLaDA Table 4).** Open the LLaDA paper. Typically:

- LLaDA wins or ties on: ARC-Easy, ARC-Challenge, TruthfulQA, HellaSwag (commonsense; diffusion's bidirectional context helps).
- LLaDA loses on: GSM8K (math, where left-to-right reasoning chains matter), HumanEval (code, where deterministic syntax is easier for AR), strict-format tasks.

Hypothesis: diffusion's strengths are tasks where global context coherence dominates; weaknesses are tasks where step-by-step sequential reasoning or strict syntax dominates.

### Lecture 1.4 Block diffusion

**Ex. 1 (Implement `compute_block_mask`).** See lecture 1.4 §3.1. Verify by printing.

**Ex. 2 (TPF maximizer).** $\bar K_B = a + bB$. $\text{TPF} = B / \bar K_B = B / (a + bB)$. Take derivative w.r.t. $B$: $d(\text{TPF})/dB = a / (a + bB)^2 > 0$. So TPF is monotonically increasing in $B$, there's no interior maximum. The limit as $B \to \infty$ is $1/b$. With $a = 4, b = 0.2$: TPF approaches 5. The practical $B = 32$ choice gets TPF $= 32/(4 + 6.4) = 3.08$, which is consistent with empirical numbers.

The lesson: larger $B$ is asymptotically faster, but quality concerns (and memory) provide the actual upper bound on $B$.

**Ex. 3 (Block diffusion vs chunked MDM).** Two reasons:

1. **Continuous KV cache.** Block diffusion's earlier-block KVs are produced by the *same* forward, so they're consistent with each other in a way that chunk-by-chunk MDM cannot match (which would have to re-encode the prefix from scratch each chunk or use stale KV).
2. **Block-causal mask is learnable.** The model is trained with this specific mask, so it learns to put the "important" information into the right positions for the block structure. Chunking at inference time when training was full-bidirectional is a distribution shift.

**Ex. 4 (Implement Option B).**

```python
def block_diff_train_step(model, x0, block_size=32):
    B_seq, L = x0.shape
    t = torch.rand(B_seq, device=x0.device).clamp(min=1e-3)
    mask = torch.bernoulli(t.unsqueeze(1).expand(-1, L)).bool()
    xt = x0.clone()
    xt[mask] = MASK_ID
    block_mask = compute_block_mask(L, block_size).to(x0.device)  # (L, L)
    logits = model(xt, attention_mask=block_mask[None, None])    # (B, L, V)
    loss = F.cross_entropy(logits[mask], x0[mask], reduction="none")
    weighted_loss = (loss / t[mask.any(dim=1).nonzero()[0]]).mean()
    return weighted_loss
```

**Ex. 5 (KV cache invariance).** Position $p$'s K/V is computed from the layer-$\ell$ residual at position $p$, which depends on the input embedding at $p$ and on all positions visible to $p$ via the attention mask at every layer below $\ell$. Within block $m$'s denoising, the *inputs* at $p$ (which is in block $m$ or earlier) change only if $p$ is in block $m$ *and* was just committed in the latest step. Once block $m$ is fully committed and we move to block $m+1$, position $p$'s input doesn't change anymore, and neither do any positions visible to $p$ (those are in blocks $\leq m$, all committed). Therefore the K/V at $p$ is stable from block $m+1$ onward.

### Lecture 1.5 Sampling strategies

**Ex. 1–5.** All require running code; see the Series 4 notebooks (when complete) for reference implementations and expected outputs.

### Lecture 1.6 Head-to-head

**Ex. 1–5.** These require running both AR and diffusion baselines and measuring; reference outputs will appear in the deliverables for Series 3.

---

## Where to go next

After Series 1, you can either:

- **Read forward** to Series 2 (mixed AR + diffusion training, self-speculation), which assumes everything in Series 1.
- **Skip to Series 3** (NLD code deep dive) if you're more interested in the implementation than the theory; Series 2 references will be flagged in-line so you know when to backtrack.
- **Run the Series 4 notebooks** in parallel with Series 2 to develop hands-on intuition.

Either way, the next interesting question is: *how do you train a single model to do AR and diffusion well at the same time?* That's the entire subject of Series 2.
