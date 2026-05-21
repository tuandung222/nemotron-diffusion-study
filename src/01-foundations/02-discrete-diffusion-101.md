# 1.2 — Discrete diffusion 101: D3PM and absorbing-state diffusion

> **Goal of this lecture.** Build up the math of *discrete* diffusion processes from scratch, with enough rigour that the simplified masked-LM losses in the next lecture stop looking like magic. By the end, you should be able to:
>
> 1. Write down the forward / reverse processes for a discrete diffusion model.
> 2. Derive the variational ELBO and see what gets simplified in practice.
> 3. Explain why the **absorbing-state** ("mask token") special case is the one that survives in modern LMs, and why its loss collapses to "weighted masked language modeling".

This lecture is more math-dense than 1.1. The reward is that everything that follows (MDLM, LLaDA, NLD's diffusion loss) is then transparently a small variation of what we derive here.

Notation: discrete tokens take values in a vocabulary $\mathcal{V}$ with $|\mathcal{V}| = V$. We'll write a sequence of length $L$ as $\mathbf{x} = (x^1, \dots, x^L)$ with each $x^i \in \mathcal{V}$. We'll use superscripts for sequence position and subscripts for diffusion timestep. A categorical distribution over $\mathcal{V}$ is encoded as a one-hot vector $\mathbf{x} \in \{0,1\}^V$ (so "$\mathbf{x} = e_v$" means the token is $v$).

---

## 1. Why "discrete" diffusion needs its own theory

You've probably seen the continuous (image) version: choose noise $\epsilon \sim \mathcal{N}(0, I)$, define $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$, train a network to predict $\epsilon$, sample by running the reverse SDE. That entire machinery presupposes that "noise" is a Gaussian perturbation in $\mathbb{R}^d$ — which is meaningless on a token vocabulary. There's no natural metric that says "the word *cat* is close to the word *dog*"; the embedding space induces one, but that's a *learned* embedding, not something the diffusion process can rely on.

The right move is to give up on Gaussian noise and design a **stochastic Markov chain over the vocabulary itself**:

$$
q(x_t^i \mid x_{t-1}^i) \;=\; \text{Cat}\!\left(x_t^i;\, Q_t\, x_{t-1}^i\right),
$$

where $Q_t \in \mathbb{R}^{V \times V}$ is a column-stochastic transition matrix. That is, **at every diffusion step, each token transitions to another token according to a categorical distribution.** Different choices of $\{Q_t\}_{t=1}^T$ give qualitatively different diffusion models.

This is the D3PM framework introduced by Austin et al. (2021) and is the right starting point.

> **Important assumption.** D3PM diffuses each sequence position **independently** through the chain: $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \prod_i q(x_t^i \mid x_{t-1}^i)$. The model captures inter-position dependencies only through the *denoising network*, not the forward process. This is what makes discrete diffusion tractable.

---

## 2. The forward process and its closed form

The chain $\mathbf{x}_0 \to \mathbf{x}_1 \to \dots \to \mathbf{x}_T$ is Markov. From the recursion

$$
q(x_t \mid x_{t-1}) = \text{Cat}(x_t; Q_t x_{t-1}),
$$

we can compute $q(x_t \mid x_0)$ in closed form for any $t$:

$$
q(x_t \mid x_0) \;=\; \text{Cat}\!\left(x_t;\, \bar{Q}_t\, x_0\right), \quad \bar{Q}_t \;\triangleq\; Q_t Q_{t-1} \cdots Q_1.
$$

We can also compute the posterior over $x_{t-1}$ given $(x_0, x_t)$ — which we'll need for the ELBO:

$$
q(x_{t-1} \mid x_t, x_0) \;\propto\; q(x_t \mid x_{t-1}) \, q(x_{t-1} \mid x_0)
\;=\; \text{Cat}\!\left(x_{t-1};\, \frac{Q_t^\top x_t \,\odot\, \bar{Q}_{t-1} x_0}{x_t^\top \bar{Q}_t x_0}\right),
$$

where $\odot$ is elementwise product and the denominator is the normalizing constant. The key observation: **the posterior is itself categorical**, with weights given by a simple product of two known vectors. There is no need for sampling or for fancy score functions.

This is the whole reason D3PM is tractable: you choose the noise process, you derive the posterior in closed form, you train a network to match it.

### 2.1 Three canonical choices of $Q_t$

D3PM enumerates several useful $Q_t$ designs:

1. **Uniform.** Each token is held with probability $1 - \beta_t$ and replaced with a uniformly random token with probability $\beta_t$:
   $$
   Q_t \;=\; (1-\beta_t)\, I + \frac{\beta_t}{V}\, \mathbf{1}\mathbf{1}^\top.
   $$
   This is the discrete analog of Gaussian noise. As $t \to T$, $\bar{Q}_t \to \frac{1}{V}\mathbf{1}\mathbf{1}^\top$ and $q(x_T \mid x_0)$ approaches uniform over $\mathcal{V}$.

2. **Discrete Gaussian on a token-distance graph.** $Q_t$ defined to favor transitions to nearby tokens under some metric (e.g. continuous embedding). Useful for image tokenizers; rarely competitive for language because there's no canonical token metric.

3. **Absorbing state.** Add a special token `[MASK]` to $\mathcal{V}$. Each non-mask token transitions to `[MASK]` with probability $\beta_t$ and stays put with probability $1 - \beta_t$; `[MASK]` stays `[MASK]` forever. In block form (vocab ordered as `[mask]` first, then real tokens):
   $$
   Q_t \;=\; \begin{pmatrix} 1 & \beta_t \mathbf{1}^\top \\ 0 & (1-\beta_t) I \end{pmatrix}.
   $$
   As $t \to T$, every position is replaced by `[MASK]`. $\bar{Q}_t$ has the same structure with cumulative mask probability $\bar{\beta}_t = 1 - \prod_{s=1}^t (1-\beta_s)$.

**Option (3) is what almost every modern discrete diffusion LM uses, including LLaDA, MDLM, SDAR, and Nemotron-Labs-Diffusion.** The rest of this lecture and Lecture 1.3 will focus on it.

> **Why absorbing wins.** Uniform diffusion forces the network to learn "is this token original or replaced?" *and* "what was the original?" simultaneously. Absorbing diffusion factors the problem: every position is either visible (=original) or `[MASK]` (=unknown). At inference, the model has a binary indicator (per position) telling it which positions still need work. That's a much cleaner conditioning signal.

---

## 3. The reverse process and the ELBO

The forward process is fixed by construction. The reverse process is **learned**:

$$
p_\theta(x_{t-1} \mid x_t) \;\approx\; q(x_{t-1} \mid x_t).
$$

Since we only have $q(x_{t-1} \mid x_t, x_0)$ in closed form (not $q(x_{t-1} \mid x_t)$), we **parameterize** the reverse by having the network predict $x_0$ from $x_t$ — call this prediction $\tilde{p}_\theta(x_0 \mid x_t)$ — and then plug it into the posterior:

$$
p_\theta(x_{t-1} \mid x_t) \;=\; \sum_{x_0 \in \mathcal{V}} q(x_{t-1} \mid x_t, x_0) \, \tilde{p}_\theta(x_0 \mid x_t).
$$

(Read this as: "average the analytic posterior over the network's belief about what the clean token was.")

This is the **mean parameterization**. It's the standard choice because the network is doing a categorical classification ($V$-way) at each position, which is exactly what cross-entropy is for.

### 3.1 Variational lower bound

The training objective is the negative log-likelihood, lower-bounded by the variational ELBO. For diffusion models the ELBO decomposes into a sum over timesteps:

$$
\mathcal{L}_{\text{vlb}} \;=\; \underbrace{\mathbb{E}_q\!\left[ -\log p_\theta(x_0 \mid x_1) \right]}_{L_0\text{: clean reconstruction}}
\;+\; \sum_{t=2}^T \underbrace{\mathbb{E}_q\!\left[ \mathrm{KL}\!\left( q(x_{t-1} \mid x_t, x_0) \;\|\; p_\theta(x_{t-1} \mid x_t) \right) \right]}_{L_{t-1}\text{: per-step KL}}
\;+\; \underbrace{\mathrm{KL}\!\left( q(x_T \mid x_0) \;\|\; p(x_T) \right)}_{L_T\text{: prior matching, $\approx 0$}}.
$$

The prior term $L_T$ vanishes when $q(x_T \mid x_0)$ converges to a fixed reference (uniform, or all-masked) by design. The reconstruction $L_0$ and per-step KLs $L_{t-1}$ are the meaningful losses.

For continuous diffusion this is where one usually invokes "$\epsilon$-prediction" and a simplified loss. For discrete diffusion, **the same simplification falls out almost trivially** — and that's where the modern simplified losses come from.

### 3.2 Auxiliary $x_0$-prediction loss (D3PM's "$L_\lambda$")

D3PM observes that you can additionally train the $x_0$-predictor *directly* with cross-entropy, which provides a much stronger gradient signal than the per-step KLs alone:

$$
\mathcal{L}_{\text{aux}}(\theta) \;=\; -\mathbb{E}_{t, x_0, x_t}\!\left[ \log \tilde{p}_\theta(x_0 \mid x_t) \right].
$$

The actual D3PM training loss is

$$
\mathcal{L}_{\text{D3PM}}(\theta) \;=\; \mathcal{L}_{\text{vlb}}(\theta) \;+\; \lambda \cdot \mathcal{L}_{\text{aux}}(\theta),
$$

with $\lambda \sim 10^{-2}$ in the original paper. In practice almost every follow-up paper dropped the VLB term entirely and kept only the auxiliary loss with a *t*-dependent weighting — that's the "modern masked diffusion loss" we'll derive in §4.

---

## 4. The absorbing-state special case: discrete diffusion = weighted masked LM

Let's specialize everything to absorbing diffusion. Let $\bar\beta_t \in [0, 1]$ be the cumulative mask probability at timestep $t$:

$$
\bar\beta_t \;=\; 1 - \prod_{s=1}^t (1 - \beta_s), \qquad \bar\beta_0 = 0,\; \bar\beta_T = 1.
$$

The forward process at any timestep is just **independent Bernoulli masking**:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) \;=\; \prod_i \Big[ (1-\bar\beta_t)\, \mathbb{1}[x_t^i = x_0^i] + \bar\beta_t\, \mathbb{1}[x_t^i = \texttt{[MASK]}] \Big].
$$

So $x_t^i$ is either the original $x_0^i$ (with probability $1 - \bar\beta_t$) or `[MASK]` (with probability $\bar\beta_t$). **There's no continuous interpolation, no Gaussian noise — just random masking.**

The posterior $q(x_{t-1} \mid x_t, x_0)$ is also clean:

- If $x_t^i \neq \texttt{[MASK]}$: then $x_t^i = x_0^i$ (because once a position is visible, it can never become masked in the forward chain), so $x_{t-1}^i = x_0^i$ with probability 1.
- If $x_t^i = \texttt{[MASK]}$: then $x_{t-1}^i = x_0^i$ with probability $\frac{\bar\beta_{t-1}}{\bar\beta_t}$, and $x_{t-1}^i = \texttt{[MASK]}$ with probability $\frac{\bar\beta_t - \bar\beta_{t-1}}{\bar\beta_t}$.

Compute the per-step KL term and you get, after some algebra:

$$
\mathrm{KL}\!\left( q(x_{t-1}^i \mid x_t^i, x_0^i) \,\|\, p_\theta(x_{t-1}^i \mid x_t^i) \right) \;=\; -\frac{\bar\beta_{t-1}}{\bar\beta_t}\, \log \tilde{p}_\theta(x_0^i \mid x_t^i) \;+\; \text{constants in }\theta.
$$

Sum over positions $i$ and timesteps $t$, and the **entire ELBO collapses to a single weighted cross-entropy**:

$$
\mathcal{L}_{\text{abs}}(\theta) \;=\; -\, \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t}\!\left[ \sum_i w_t \cdot \mathbb{1}[x_t^i = \texttt{[MASK]}] \cdot \log \tilde{p}_\theta(x_0^i \mid \mathbf{x}_t) \right],
$$

where $w_t = \frac{\bar\beta_{t-1}}{\bar\beta_t \cdot (\bar\beta_t - \bar\beta_{t-1})}$ in the discrete-time derivation, simplifying further in the continuous-time limit.

**This is just masked language modeling**, but with two twists:

1. **The mask ratio $\bar\beta_t$ is random per batch** — drawn from the noise schedule rather than fixed at 15% as in BERT.
2. **There is a $t$-dependent weight $w_t$** on each masked position.

That's it. Once you've understood §1–§3, you've earned the right to think of absorbing diffusion as *masked LM with a random mask ratio and a principled per-step weighting*. This is the conceptual punchline of the entire chapter.

> **Practical consequence.** You can take any pretrained encoder-only masked LM (BERT, RoBERTa) or any decoder-only LM, and turn it into a discrete diffusion LM by changing the training loop to (a) sample a random mask ratio per batch, (b) mask that fraction of positions, (c) compute weighted cross-entropy on the masked positions. There is no new architecture required at training time. We'll see that NLD-VLM-8B does exactly this — it inherits its backbone weights from `Ministral3-8B` (a standard AR decoder LM) and continual-pretrains with the absorbing-diffusion loss layered on top.

---

## 5. Continuous-time formulation

D3PM is presented in discrete time ($T = 1000$ steps, schedules like cosine or linear). Modern papers re-derive everything in continuous time $t \in [0, 1]$, which removes some hyperparameters and exposes a cleaner picture.

Let $\bar\beta(t) \in [0,1]$ be a *continuous, non-decreasing* schedule with $\bar\beta(0) = 0$ and $\bar\beta(1) = 1$. The forward process is

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) \;=\; \prod_i \Big[ (1 - \bar\beta(t))\, \mathbb{1}[x_t^i = x_0^i] + \bar\beta(t)\, \mathbb{1}[x_t^i = \texttt{[MASK]}] \Big].
$$

In the continuous-time limit, the ELBO becomes an integral:

$$
\mathcal{L}_{\text{ct}}(\theta) \;=\; \mathbb{E}_{\mathbf{x}_0, t \sim \mathcal{U}[0,1]}\!\left[\, \frac{\bar\beta'(t)}{\bar\beta(t)} \sum_i \mathbb{1}[x_t^i = \texttt{[MASK]}] \cdot \big(-\log \tilde{p}_\theta(x_0^i \mid \mathbf{x}_t)\big)\, \right],
$$

where $\bar\beta'(t)$ is the derivative of the schedule. With the linear schedule $\bar\beta(t) = t$, we have $\bar\beta'(t) / \bar\beta(t) = 1/t$ — the famous "$1/t$ reweighting" you see in MDLM and downstream papers.

> **NLD config check.** Open the [config.json of `nvidia/Nemotron-Labs-Diffusion-VLM-8B`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B/blob/main/config.json). The diffusion paradigm is `"dlm_type": "llada"` — i.e. continuous-time absorbing diffusion with the simplified MDLM-style loss. You'll see references to a "$1/t$ reweighting" later in the paper that is exactly the $\bar\beta'(t)/\bar\beta(t)$ factor above.

### 5.1 Why continuous time matters

Three reasons:

1. **No timestep hyperparameter.** You sample $t \sim \mathcal{U}[0,1]$ instead of $t \in \{1, \dots, 1000\}$. One fewer schedule to design.
2. **Cleaner loss expression.** The $1/t$ factor is now a property of the schedule, not a discrete sum.
3. **Continuous-time inference can use any number of denoising steps** without retraining — you sample timesteps $t_1 > t_2 > \dots > t_K$ from a chosen inference schedule, and the model still works.

The downside is that the loss has higher variance for small $t$ (the $1/t$ blows up), which motivates the "global loss averaging" trick we'll see in Series 2: NLD averages per-token losses globally across the batch rather than per-sample, which damps the $1/t$ spikes.

---

## 6. Architecture during training

What does the actual training forward pass look like?

```
input:   x_0  = ["The", "cat", "sat", "on", "the", "mat"]
sample t ~ U(0,1)                # noise level
sample mask ~ Bernoulli(t) per position
build x_t = [<MASK> if mask[i] else x_0[i]]

# x_t is fed to the model
logits = model(x_t)              # shape (L, V)
loss = mean over masked positions of CE(logits[i], x_0[i])
loss *= (1 / t)                  # the schedule weight
backprop
```

**Two important points** that often trip up people coming from AR:

1. **The model is the same architecture as a decoder LM.** No special "diffusion head", no extra timestep embedding (the timestep enters only through the loss weighting and the mask ratio). NLD goes one step further and uses *literally the same weights* as a pre-existing AR LM (Ministral3-8B). The diffusion-vs-AR distinction at training time is entirely about (a) what positions are masked and (b) the attention mask.
2. **The attention mask is NOT causal.** Within a diffusion forward pass, every position can attend to every other position — including the future. This is essential, because predicting a masked position from the prefix alone is exactly the AR objective (which we still keep, in mixed AR-diffusion), but predicting it from *bidirectional context* is what makes diffusion useful. In Series 2 we'll see how NLD combines a causal stream and a bidirectional stream in one forward.

If you're picturing this as "BERT-style masked LM with a random mask ratio and a $1/t$ weight" you're correct, modulo the schedule. The conceptual gap from BERT to a diffusion LM is much narrower than the gap from AR LM to either.

---

## 7. Architecture during inference

Inference is where discrete diffusion looks different from masked LM:

```
output_len = N
x_T = ["<MASK>"] * N

for t in [t_1 > t_2 > ... > t_K]:
    logits = model(x_t)
    x_0_hat = argmax(logits, dim=-1)        # or sample
    # decide which positions to "commit" at this step
    commit_mask = sampler(x_t, logits, t)
    x_t = where(commit_mask, x_0_hat, x_t)  # unmask
    # optionally: remask some committed tokens (for refining)
return x_t
```

Two questions arise immediately:

1. **How many denoising steps $K$ do we need?** In principle $K = T$ (the training horizon) recovers the analytic posterior; in practice $K$ can be much smaller (like $K=8$ to $K=32$). The trade-off: smaller $K$ = faster, larger $K$ = closer to the "true" posterior and usually better quality. This is precisely the TPF vs accuracy curve we saw in lecture 1.1.

2. **How do we pick which positions to commit at each step?** This is the **sampler**, and it's where most of the modern engineering happens. The default "confidence-based" sampler — commit positions whose top-1 probability exceeds a threshold $\tau$ — is what NLD uses by default. We'll see in lecture 1.5 that better samplers (learned, ancestral, remasking) can substantially close the gap to the speed-of-light upper bound.

> **NLD inference defaults.** `block_length = 32`, `threshold = 0.9`, `steps = 512`. Translation: at each forward pass, the model is denoising a block of up to 32 masked positions, and commits any whose top-1 confidence is $\geq 0.9$. We'll see this code in Series 3.

---

## 8. What we've earned

After this lecture you should be able to:

1. **Write the forward process.** $q(\mathbf{x}_t \mid \mathbf{x}_0) = \prod_i \big[(1 - \bar\beta(t))\, \mathbb{1}[x_t^i = x_0^i] + \bar\beta(t)\, \mathbb{1}[x_t^i = \texttt{[MASK]}]\big]$.
2. **Write the training loss.** $\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, t}\!\left[ \frac{1}{t} \sum_{i : \text{masked}} -\log \tilde{p}_\theta(x_0^i \mid \mathbf{x}_t) \right]$.
3. **Describe inference.** Iteratively unmask positions in $K$ denoising steps using a sampler over confidences.
4. **Recognize the architecture cost.** Same transformer as an AR LM. The only architectural change at training time is the attention mask (bidirectional within the diffusion forward); at inference time the algorithm changes but the model graph doesn't.

The next lecture (1.3) walks through three specific instantiations of this framework — **SEDD**, **MDLM**, and **LLaDA** — to make the connections to specific modern papers explicit, and to highlight the design choices that NLD inherits.

---

## Further reading (primary sources)

- Austin, J., Johnson, D. D., Ho, J., Tarlow, D., van den Berg, R. *Structured Denoising Diffusion Models in Discrete State-Spaces*. NeurIPS 2021. **The D3PM paper.** Sections 3 and 4 derive the forward/reverse processes and the ELBO; Section 5 enumerates the choices of $Q_t$.
- Hoogeboom, E., Nielsen, D., Jaini, P., Forré, P., Welling, M. *Argmax Flows and Multinomial Diffusion*. NeurIPS 2021. Independently introduced uniform discrete diffusion; useful for contrast.
- Campbell, A., Benton, J., De Bortoli, V., Rainforth, T., Doucet, A. *A Continuous-Time Framework for Discrete Denoising Models*. NeurIPS 2022. Continuous-time generalization, including the $1/t$ schedule weight.
- Shi, J., Han, K., Wang, Z., et al. *Simplified and Generalized Masked Diffusion for Discrete Data*. NeurIPS 2024. Very useful "modern" treatment.

## Exercises

1. **D3PM forward process.** For the absorbing process with constant per-step mask probability $\beta_t = \beta$, derive $\bar\beta_t$ in closed form as a function of $t$ and $\beta$. Verify that $\bar\beta_t \to 1$ as $t \to \infty$.

2. **Posterior in closed form.** Verify the formula for $q(x_{t-1}^i \mid x_t^i, x_0^i)$ in the absorbing case (§4). Specifically, show that when $x_t^i = \texttt{[MASK]}$ and $x_0^i = v$, we have

   $$
   q(x_{t-1}^i = v \mid x_t^i = \texttt{[MASK]}, x_0^i = v) \;=\; \frac{\bar\beta_{t-1}}{\bar\beta_t}.
   $$

   *(Hint: use $q(x_{t-1} \mid x_0) \propto (1 - \bar\beta_{t-1}) \delta_{x_0} + \bar\beta_{t-1} \delta_\text{MASK}$ and $q(x_t \mid x_{t-1}) \propto (1 - \beta_t) \delta_{x_{t-1}} + \beta_t \delta_\text{MASK}$.)*

3. **Why $1/t$?** Substitute the linear schedule $\bar\beta(t) = t$ into the continuous-time loss in §5 and verify the $1/t$ reweighting. Then substitute the cosine schedule $\bar\beta(t) = 1 - \cos(\pi t / 2)^2$ and derive the corresponding weight.

4. **Variance of the diffusion loss.** Argue (heuristically) why the diffusion loss has higher variance at small $t$ than at large $t$, even though the *number* of masked positions is smaller at small $t$. This motivates the "global loss averaging" trick used by NLD, which we'll see in Series 2.

5. **From BERT to MDLM.** Take a pretrained BERT-base. Sketch the minimal code changes to its pretraining loop to turn it into an MDLM-style discrete diffusion LM. (Hint: it's about 30 lines of Python.)

Sketched solutions are in [Appendix B](../appendix/reading-list.md#exercise-solutions).
