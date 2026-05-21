# 1.3 - Masked Diffusion Models: SEDD, MDLM, LLaDA

> **Goal of this lecture.** Walk through the three "modern" papers that turn the D3PM theory of lecture 1.2 into actually-working language models: **SEDD** (Lou et al., 2024), **MDLM** (Sahoo et al., 2024), and **LLaDA** (Nie et al., 2024). Each contributes a small but important refinement, and together they define the recipe that NLD inherits.

This lecture is mostly about **simplification**. Each successive paper drops something from D3PM until the loss collapses to a single line of PyTorch. By the end you should be comfortable saying *"masked diffusion training is just a one-line weighted cross-entropy with a random mask ratio"* - and know exactly which factors are doing the work.

---

## 1. The story arc

| Paper | Year | One-sentence contribution |
|---|---|---|
| **D3PM** | 2021 | Introduces the discrete-diffusion framework with arbitrary transition matrices. |
| **SEDD** | 2024 | Discrete score matching: estimate the *ratios* of marginals instead of the marginals directly. Sets a new SOTA among diffusion text LMs but requires a custom loss and a non-standard parameterization. |
| **MDLM** | 2024 | Shows that the absorbing-state D3PM ELBO, in continuous time, collapses to a **single, weight-free cross-entropy on masked positions**. Same accuracy as SEDD with a much simpler loss. |
| **LLaDA** | 2024 | Scales MDLM to 8B parameters, trains on a tokenized corpus comparable to AR LLMs, and shows competitive zero-shot benchmarks on reasoning. **This is the recipe NLD inherits.** |

After this lecture we'll have the absorbing-state, continuous-time, MDLM-style loss in our pocket - and we'll be ready to discuss block-wise factorization in lecture 1.4.

---

## 2. SEDD: discrete score matching

D3PM trains the network to predict $x_0$ from $x_t$, then plugs that prediction into the analytic posterior. SEDD takes a different route: it trains the network to predict **score ratios** in the discrete setting.

The continuous-diffusion score is $\nabla_x \log p_t(x)$. In discrete spaces, the analog is the **ratio** $\frac{p_t(y)}{p_t(x)}$ for pairs $(x, y)$ that differ in one coordinate. SEDD's loss is

$$
\mathcal{L}_{\text{SEDD}}(\theta) \;=\; \mathbb{E}_{t, x_t}\!\left[ \sum_{y \sim x_t} \omega_{xy}(t) \cdot \ell\!\left( s_\theta(x_t, t)_y\, ,\, \frac{q_t(y)}{q_t(x_t)} \right) \right],
$$

where $s_\theta(x_t, t)$ is the model's predicted score for each candidate neighbor $y$, and $\ell(\cdot, \cdot)$ is a score-entropy divergence. The construction is principled but the parameterization is awkward: you need to output a $V$-dimensional vector of *ratios* per position, not a probability distribution. Compute, memory, and post-hoc decoding all need custom code.

**SEDD's main empirical result:** matches or beats the strongest continuous text-as-image diffusion models (Plaid, GENIE) and approaches the AR baseline on small-scale text generation. The price is the custom loss.

You can skip SEDD as a practitioner; we mention it because it set the bar that MDLM had to clear.

---

## 3. MDLM: the simplification that broke everything open

Sahoo, Arriola, et al. (2024) - "Simple and Effective Masked Diffusion Language Models" - proved that you don't need score entropy or any of SEDD's machinery to get the same accuracy. You just need three things:

1. **Absorbing-state forward process** (a.k.a. masking).
2. **Continuous-time formulation** ($t \in [0, 1]$).
3. **Train the $x_0$-predictor directly with the simplified loss derived in lecture 1.2 §4.**

The MDLM loss, written exactly as it appears in the paper:

$$
\mathcal{L}_{\text{MDLM}}(\theta) \;=\; -\mathbb{E}_{\mathbf{x}_0,\, t \sim \mathcal{U}[0,1],\, \mathbf{x}_t \sim q(\cdot \mid \mathbf{x}_0, t)} \left[\, \frac{\bar\beta'(t)}{\bar\beta(t)}\, \sum_{i\, :\, x_t^i = \texttt{[MASK]}} \log \tilde{p}_\theta(x_0^i \mid \mathbf{x}_t)\, \right].
$$

With the linear schedule $\bar\beta(t) = t$ (which the paper uses), $\bar\beta'(t) / \bar\beta(t) = 1/t$:

$$
\boxed{\;
\mathcal{L}_{\text{MDLM}}(\theta) \;=\; \mathbb{E}_{\mathbf{x}_0,\, t,\, \mathbf{x}_t} \left[\, \frac{1}{t}\, \sum_{i\, :\, x_t^i = \texttt{[MASK]}} \mathrm{CE}\!\left( \tilde{p}_\theta(\cdot \mid \mathbf{x}_t),\, x_0^i \right)\, \right].
\;}
$$

Read this carefully:

- **For each example**, draw a random noise level $t \in [0, 1]$.
- **Mask each position independently** with probability $t$. (Linear schedule.)
- **Forward through the model**. Compute cross-entropy on the masked positions only.
- **Reweight by $1/t$** (this is what makes it a proper ELBO; without it the loss is biased).
- **Average over the batch**.

Less than ten lines of PyTorch:

```python
def mdlm_loss(model, x0, vocab_size, mask_token_id):
    B, L = x0.shape
    t = torch.rand(B, device=x0.device)              # noise level per example
    t = t.clamp(min=1e-3)                            # numerical safety
    mask_prob = t.unsqueeze(1).expand(-1, L)         # broadcast to (B, L)
    is_masked = torch.bernoulli(mask_prob).bool()    # (B, L)
    x_t = torch.where(is_masked, mask_token_id, x0)  # noisy input
    logits = model(x_t)                              # (B, L, V)
    nll = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        x0.reshape(-1),
        reduction="none",
    ).view(B, L)
    nll = nll * is_masked.float()                    # only masked positions
    per_example_loss = nll.sum(dim=1) / t            # 1/t reweighting
    return per_example_loss.mean()
```

That's it. That's the entire training loss of a state-of-the-art discrete diffusion LM.

### 3.1 Why the $1/t$ factor?

The $1/t$ factor is what makes this a principled lower bound on the data log-likelihood, not just an ad-hoc training objective. Without it, the small-$t$ regime (almost no masking) is underweighted, and the large-$t$ regime (almost all masking) is overweighted, biasing the model to denoise *more* than it should from heavily-masked contexts.

The $1/t$ factor is sometimes called the **diffusion ELBO weighting**, and it shows up identically in the continuous Gaussian diffusion case (Ho et al. 2020's "$\beta$-VDM" reweighting). It's a structural feature of variational diffusion.

### 3.2 Variance and stability

The $1/t$ factor has a downside: when $t$ is small (say $t = 0.01$), the weight is 100. Most batches will have a few examples with $t \approx 0$ and they will dominate the gradient. This causes:

1. **High gradient variance** - training is noisier than AR.
2. **A long training tail** - the loss spends a lot of time fitting the rare high-weight examples.

MDLM addresses (1) with **time-conditioning bias correction** (a learned per-$t$ bias on the logits) and uses a slightly higher learning rate. **Global loss averaging** (averaging cross-entropies over all *tokens* in the batch instead of all *examples*) is a different fix that NLD uses - we'll see it in Series 2.

### 3.3 What about the per-step KLs?

Doesn't the ELBO have additional KL terms $L_{t-1}$? Yes - and in the absorbing case the simplification of §4 of lecture 1.2 shows that *summing* the KLs across all $t$ gives exactly the integral form above (modulo discretization). The continuous-time MDLM loss *is* the ELBO, just rewritten as a single expectation. No information is lost.

---

## 4. LLaDA: scaling MDLM to 8B and beyond

Nie et al. (2024) - "Large Language Diffusion Models" - applied MDLM at LLM scale (1B and 8B parameters), trained from scratch on a token corpus comparable to standard pretraining datasets, and reported the first set of zero-shot benchmarks that placed diffusion LMs in the same conversation as 8B AR LLMs.

Key practical contributions:

1. **Architecture: standard decoder transformer.** No exotic modules. The model is essentially a LLaMA-style 8B with bidirectional attention during training.

2. **Tokenizer: BPE shared with AR LLMs.** Earlier diffusion LMs sometimes used character-level or smaller vocabularies. LLaDA uses a 100k-ish BPE that is API-compatible with the AR ecosystem. **This matters for NLD**, which uses Ministral3's tokenizer (vocab 131k) and inherits its embedding table directly.

3. **Inference recipe.** Confidence-based sampling with a fixed threshold $\tau$. This is the foundation of every modern diffusion-LM sampler, including NLD's default `threshold=0.9`.

4. **Found bugs in everyone's evaluation.** Reading the LLaDA paper carefully reveals that several prior diffusion LM benchmarks were measuring "perplexity on uniformly masked sequences" instead of "perplexity on the natural sequence", which is much easier. LLaDA introduced a corrected evaluation protocol that has since become standard.

### 4.1 LLaDA's inference loop

The paper's pseudocode (Algorithm 1) for LLaDA inference, simplified:

```
INPUT: prompt p (length L_p), output length L_y, num steps K, sampler S
OUTPUT: completion y

# initialize y as all [MASK]
y = [MASK] * L_y
xt = concat(p, y)

# schedule: K timesteps from t=1 down to t=0
schedule = [1 - (k / K) for k in range(K)]

for t_k in schedule:
    logits = model(xt)              # bidirectional forward
    # decide which positions to commit at this step
    commit_set = S(xt, logits, t_k)
    # commit: replace [MASK] with argmax (or sample)
    xt[commit_set] = argmax(logits[commit_set])

return xt[L_p:]
```

Two important details:

- **The prompt is fed into the same sequence as the completion**, with the completion initialized to all `[MASK]`. The model attends bidirectionally across the entire sequence.
- **The sampler $S$ is replaceable.** LLaDA reports results for several samplers; the default is confidence-thresholded greedy. We'll spend all of lecture 1.5 on this.

### 4.2 LLaDA's empirical findings

The paper's most cited results:

| Benchmark | Qwen2.5-7B (AR) | LLaDA-8B (Diffusion) | Note |
|---|---|---|---|
| GSM8K | 91.9 | 79.9 | AR leads on math. |
| MMLU | 74.9 | 65.5 | AR leads on knowledge. |
| MMLU-pro | 49.0 | 50.0 | Tie. |
| TruthfulQA | 53.1 | 51.0 | Tie. |
| HumanEval | 77.4 | 49.4 | AR leads on code. |

The story: LLaDA-8B trained from scratch is **roughly comparable to LLaMA-2-7B** on broad benchmarks, behind Qwen2.5 on math/code, and at the time of writing was the first credible 8B diffusion LM. The gap to AR was significant but no longer a chasm.

**This is one of the motivations for mixed AR-diffusion** (Series 2): you don't pick one objective and live with its weaknesses; you train both, and the model can run as whichever mode the task prefers. NLD's 8B instruct model in AR mode actually slightly *beats* Qwen3-8B on average accuracy, because joint training preserves AR quality, and its diffusion mode TPF is 2.57× higher. Tri-mode is a strict Pareto improvement when the joint training works.

---

## 5. SEDD vs MDLM vs LLaDA in one table

| Aspect | SEDD | MDLM | LLaDA |
|---|---|---|---|
| Parameterization | discrete score ratios | $x_0$-prediction (categorical) | $x_0$-prediction |
| Loss | discrete score entropy | weighted CE on masked positions | weighted CE on masked positions |
| Time domain | continuous | continuous | continuous |
| Forward process | absorbing or uniform | absorbing | absorbing |
| Scale demonstrated | 90M – 1B | up to 1B | **up to 8B** |
| Tokenizer | custom | BPE | BPE (LLaMA-compatible) |
| Code complexity | high | low | low (= MDLM + scaling) |
| Used by NLD? | No | Yes (loss family) | Yes (training recipe) |

What does NLD specifically inherit from each?

- From **D3PM**: the absorbing-state forward process, the conceptual framework.
- From **MDLM**: the simplified loss, continuous-time formulation, $1/t$ reweighting.
- From **LLaDA**: the 8B-scale recipe, the BPE-compatible tokenizer, the confidence-threshold sampler.
- **New in NLD**: the *block-wise* version of the loss (next lecture), the *joint* AR+diffusion training (Series 2), self-speculation (Series 2), and tri-mode inference.

You can see this in `config.json` of NLD-VLM-8B: `"dlm_type": "llada"`. The diffusion component is literally LLaDA-style; what NLD adds is the AR-side coupling and the block factorization.

---

## 6. Stress-testing the simplified picture

Let's check that the MDLM loss does what we think it does.

**Q: What happens when $t \to 0$?**
A: No positions are masked. The cross-entropy sum is empty, the loss is 0. The $1/t$ factor is a $0/0$ that becomes irrelevant in practice (multiplied by 0 number of terms). In code, you clamp $t \geq 10^{-3}$ to avoid numerical issues, and you sample $t$ from a distribution that doesn't put much mass at 0 (uniform is fine because the expected number of masked positions is $L \cdot t$, which is small for small $t$).

**Q: What happens when $t \to 1$?**
A: Every position is masked; the model is being asked to generate the entire sequence from scratch with no observed context (except the prompt, in a conditional generation setting). This is the hardest regime and contributes the most stable, well-conditioned gradient signal - but with weight $1/t \to 1$, not particularly upweighted. The bulk of the training signal comes from intermediate $t$.

**Q: Why isn't this just BERT?**
A: BERT uses a fixed mask ratio (15%), no $1/t$ reweighting, and the encoder is bidirectional but trained jointly with NSP. MDLM uses a *random* mask ratio per example $\sim \mathcal{U}[0,1]$, the $1/t$ reweighting, and trains a decoder-style transformer with bidirectional attention. The model architectures can be identical; the *training distribution* is the actual difference.

A useful intuition: **BERT is a diffusion LM trained at a single noise level**. MDLM is BERT trained across all noise levels with the principled per-level weighting. The samples you can generate from BERT by iteratively unmasking are exactly what you'd get from a one-shot diffusion sampler at $t = 0.15$.

**Q: Where does the architecture have to change?**
A: Only the attention mask. AR uses a causal lower-triangular mask; diffusion uses an all-ones mask (full bidirectional). The same set of weights produces different outputs because the *visible context* per position is different. In Series 2 we'll see how to train *both* attention patterns simultaneously and have the model excel at either.

---

## 7. Sanity-check exercise: write the MDLM loss yourself

Before moving on, take 15 minutes and implement the MDLM loss against a tiny GPT-2 baseline. Goal: convince yourself that the entire delta from AR training to diffusion training is on the order of **ten lines of code**.

Concretely:

1. Start from a HuggingFace `GPT2LMHeadModel`. Add `mask_token` to the tokenizer (it doesn't have one).
2. Replace the AR data collator with one that samples $t \sim \mathcal{U}[0,1]$, masks Bernoulli($t$) of positions, and provides $x_0$ as the label.
3. Replace the causal attention mask with a full bidirectional mask. (Trick: pass `attention_mask=None` and override `_attn_implementation` to skip the causal mask. Or just train a different model class.)
4. Write the loss as `(1/t) * F.cross_entropy(logits[masked], x0[masked], reduction="sum") / masked.sum().clamp(min=1)`.
5. Train for an hour on TinyStories or similar.

You will *not* get a good model from one hour of training, but you should see (a) the loss curve decrease and (b) samples that are syntactically reasonable English. The point is to internalize how trivial the training-loop delta is.

The Series 4 notebooks will do this from scratch in a more controlled setting.

---

## 8. What we now have

- The **MDLM loss**: a single, weight-free cross-entropy on masked positions, multiplied by $1/t$.
- The **LLaDA recipe**: scale MDLM to LLM size, use a BPE tokenizer, use confidence-threshold sampling.
- The realization that **the architecture is the same as an AR LM**; only the attention mask and the loss formula change.
- Reference to where each of these fits in NLD's config: `dlm_type: llada`, the loss inherited from MDLM, the tokenizer from Ministral3.

The next lecture explains why "full-sequence diffusion" - denoising all $L$ positions of a long sequence in parallel - is actually a bad idea for long contexts, and how block-wise factorization fixes it. This is the conceptual step that lets diffusion LMs reuse KV caches and that makes self-speculation possible.

---

## Further reading (primary sources)

- Lou, A., Meng, C., Ermon, S. *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*. ICML 2024. **The SEDD paper.**
- Sahoo, S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., Kuleshov, V. *Simple and Effective Masked Diffusion Language Models*. NeurIPS 2024. **The MDLM paper.** §3 has the loss derivation, §4 the empirical comparison to SEDD.
- Nie, S., Zhu, F., Du, C., Pang, T., Liu, Q., Zeng, G., Lin, M., Li, C. *Large Language Diffusion Models*. arXiv:2502.09992, 2024. **The LLaDA paper.** §3 is the inference recipe, §6 is the benchmarks.
- Ho, J., Jain, A., Abbeel, P. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. The continuous-image analog - useful as a comparison; you'll see the same $\beta$-VDM weight argument.

## Exercises

1. **Derive the loss collapse.** Starting from the discrete-time ELBO in lecture 1.2 §3.1, specialize to the absorbing process and the linear schedule $\bar\beta_t = t / T$, then take the continuum limit $T \to \infty$. You should recover the MDLM loss. (This is the cleanest path to the $1/t$ factor.)

2. **Variance of the $1/t$-weighted loss.** Show analytically that, for a fixed batch of sequences and noise levels drawn uniformly, the variance of the per-example MDLM loss has a $1/t$ singularity. Argue why this motivates "averaging over tokens in the batch" rather than "averaging over examples".

3. **MDLM ↔ AR equivalence at $t = 1$.** Show that when $t$ is sampled from a degenerate distribution concentrated at $t = 1$ (every position masked), the MDLM loss reduces to maximum-likelihood training of a generic categorical predictor - i.e. just MLE. Why is this the "easiest" case from the gradient-variance perspective?

4. **Why bidirectional attention at training time?** Argue why MDLM *requires* bidirectional attention (or at least, why pure causal attention is suboptimal). What would happen if you trained MDLM with causal attention? *(Hint: think about the conditioning available to predict each masked position.)*

5. **Reading LLaDA Table 4.** Open the LLaDA paper and read Table 4 (benchmark comparison). Identify two benchmarks where LLaDA outperforms LLaMA-2-7B and two where it loses by a wide margin. Propose a hypothesis for *why* - i.e. what is it about the benchmark that favors AR or diffusion?

Sketched solutions are in [Appendix B](../appendix/reading-list.md#exercise-solutions).
