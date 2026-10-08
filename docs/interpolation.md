# The interpolation between discriminative and generative training

> [!NOTE]
> Theory only: what the terms of the interpolated loss are, how their balance depends
> on the number of features, and what the parameter $\beta$ means. Nothing here
> depends on the code. The review of the plan, the experiments and the implementation
> are in [plan.md](plan.md) ("Planned: beta").

## 1. Setup: three losses and one identity

The model is a joint density $p_\theta(x, c)$ over $n$ continuous features
$x = (x_1, \dots, x_n)$ and a class $c \in \{1, \dots, C\}$. Every loss is a mean over
the data. Write $N = n + 1$ for the number of modelled variables (the $n$ features and
the class).

$$
L_\text{dis} = -\log p(c \mid x), \qquad
L_\text{gen} = -\log p(x, c), \qquad
L_\text{marg} = -\log p(x).
$$

By the chain rule, $p(x, c) = p(c \mid x)\, \prod_{i=1}^{n} p(x_i \mid x_{<i})$, so with
the per-feature conditional losses $\ell_i = -\log p(x_i \mid x_{<i})$:

$$
L_\text{gen} = L_\text{dis} + L_\text{marg}, \qquad
L_\text{marg} = \sum_{i=1}^{n} \ell_i = n\,\bar\ell,
$$

where $\bar\ell$ is the mean per-feature loss. The identity holds for any order of the
features. **$L_\text{gen}$ is a sum of $N$ per-variable terms: $n$ for the features and
one, $L_\text{dis}$, for the class.**

## 2. The alpha objective

$$
L(\alpha) = (1-\alpha)\,L_\text{dis} + \alpha\,L_\text{gen} = L_\text{dis} + \alpha\,L_\text{marg},
\qquad \alpha \in [0, 1].
$$

Per variable, the class term has weight 1 and each feature term has weight $\alpha$.
$\alpha = 0$ is a classifier and $\alpha = 1$ a joint density model.

## 3. Alpha is an exchange rate

Let $\theta^*(\alpha)$ be a minimiser of $L(\alpha)$ that moves smoothly with $\alpha$.
It satisfies

$$
\nabla L_\text{dis}(\theta^*) + \alpha\, \nabla L_\text{marg}(\theta^*) = 0.
$$

Follow both losses along the path of minimisers. With $\dot\theta = d\theta^*/d\alpha$:

$$
\frac{d}{d\alpha} L_\text{dis}(\theta^*) = \nabla L_\text{dis} \cdot \dot\theta
= -\alpha\, \nabla L_\text{marg} \cdot \dot\theta
= -\alpha\, \frac{d}{d\alpha} L_\text{marg}(\theta^*),
$$

that is,

$$
dL_\text{dis} = -\alpha\, dL_\text{marg}.
$$

**$\alpha$ is exactly the price** at which the trained model trades the two losses: it
gives up $\alpha$ nats of classifier loss for each nat of marginal NLL it gains. No
constant is involved. Since $L_\text{marg} = n\bar\ell$, one nat of the *mean
per-feature* loss is worth $\alpha n$ nats of $L_\text{dis}$:

$$
dL_\text{dis} = -\alpha n\, d\bar\ell.
$$

> [!IMPORTANT]
> **Hypothesis.** The trade-off between $L_\text{dis}$ and $\bar\ell$ has roughly the
> same shape whatever $n$. Then the interesting region of a sweep sits at a fixed value
> of $\alpha n$, not of $\alpha$. Section 4 argues for it; experiments must confirm it.

## 4. Why the price scales with n

### 4.1 The curvatures of the n feature terms add

Second derivatives add: $\nabla^2 L_\text{marg} = \sum_i \nabla^2 \ell_i$. The question
is whether the $n$ terms reinforce or cancel. Near the optimum, they reinforce.

**Lemma 1.** Let $q_\theta(y \mid z)$ be a conditional density, and let $y$ be drawn
from $q_\theta(\cdot \mid z)$ (the model fits the data). Then

$$
\mathbb{E}\big[-\nabla^2 \log q_\theta(y \mid z)\big]
= \mathbb{E}\big[\nabla \log q_\theta(y \mid z)\, \nabla \log q_\theta(y \mid z)^\top\big],
$$

which is positive semi-definite.

*Proof.* Differentiate $\int q_\theta(y \mid z)\, dy = 1$ with respect to $\theta$. Since
$\nabla q = q\, \nabla \log q$, this gives $\mathbb{E}[\nabla \log q] = 0$. Differentiate
again: $\int \big(\nabla q\, \nabla \log q^\top + q\, \nabla^2 \log q\big)\, dy = 0$, i.e.
$\mathbb{E}[\nabla \log q\, \nabla \log q^\top] + \mathbb{E}[\nabla^2 \log q] = 0$. The
left-hand matrix is an average of outer products $v v^\top$, hence positive
semi-definite. $\blacksquare$

Apply it with $y = x_i$, $z = x_{<i}$: near the optimum, each $\ell_i$ contributes a
curvature that is on average non-negative in every direction. The curvature of
$L_\text{marg}$ is therefore a sum of $n$ non-negative terms. In a direction of parameter
space that all $n$ conditionals depend on, it grows in proportion to $n$. The class term
$L_\text{dis}$ is one more term of the same kind (Lemma 1 with $y = c$, $z = x$).

In an MPS every conditional depends on every core: $p(x_i \mid x_{<i})$ is built from
the cores left of $i$ (which carry $x_{<i}$) and the cores right of $i$ (which
marginalise the rest). So the $n$ terms share the parameters.

### 4.2 Where the minimiser sits

Take one direction in parameter space, with coordinate $t$. Near the optimum, model the
two losses as parabolas: $L_\text{dis} \approx \tfrac{a}{2}(t - t_\text{d})^2$ and
$L_\text{marg} \approx \tfrac{b}{2}(t - t_\text{m})^2 + \text{const}$, where
$b = \sum_i b_i \approx n \bar b$ by 4.1. The minimiser of
$L_\text{dis} + \alpha L_\text{marg}$ is

$$
t^* = \frac{a\, t_\text{d} + \alpha b\, t_\text{m}}{a + \alpha b},
$$

a fraction $\alpha b / (a + \alpha b)$ of the way from the classifier's optimum
$t_\text{d}$ to the density's $t_\text{m}$. It is halfway at

$$
\alpha = \frac{a}{n \bar b}.
$$

**So the switch point scales like $1/n$**, and $\alpha n$ at the switch, $a/\bar b$,
does not depend on $n$. With many directions, each one switches at its own
$\alpha_u = a_u / b_u$. Directions the classifier does not need ($a_u \approx 0$)
switch at any $\alpha > 0$: the density improves "for free". Directions both losses
need switch late, and the accuracy drops. A sweep should therefore show **two knees**:
the density catches up at small $\alpha n$, the accuracy drops at large $\alpha n$.

### 4.3 Gradient noise grows only like the square root of n

**Lemma 2.** Let $g_i = \nabla \ell_i$ for one sample drawn from the model. For $i < j$,
$\mathbb{E}[g_i\, g_j^\top] = 0$.

*Proof.* $g_i$ depends only on $x_{\le i}$, and by the first step of Lemma 1,
$\mathbb{E}[g_j \mid x_{<j}] = 0$. So
$\mathbb{E}[g_i g_j^\top] = \mathbb{E}\big[g_i\, \mathbb{E}[g_j \mid x_{<j}]^\top\big] = 0$.
$\blacksquare$

The per-sample gradient of $L_\text{marg}$ is $\sum_i g_i$, a sum of $n$ uncorrelated
terms. Its squared norm grows like $n$, so the norm grows like $\sqrt n$. This is the
*noise* of a mini-batch gradient. It sets how much each optimiser step fluctuates, not
where the minimiser lies (the minimiser is where the *mean* gradient vanishes, §3).

Gradient-norm ratios therefore cannot measure the right scaling:
- on a trained model, the full-batch ratio
  $\lVert\nabla L_\text{dis}\rVert / \lVert\nabla L_\text{marg}\rVert$ equals $\alpha$ by
  the stationarity condition of §3;
- mini-batch gradient norms are dominated by the noise of Lemma 2;
- at initialisation, the ratio describes the initialisation.

### 4.4 Limits

- **Sharing.** The count $n$ applies where the contested directions are shared by all
  $n$ conditionals. If only about $\xi$ conditionals near a site respond (a correlation
  length $\xi \ll n$), the curvature scales like $\xi$. In general the switch point
  scales like $N^{-p}$ with $p \in [0, 1]$; $p = 1$ is the prior from 4.1, not a theorem.
- **The per-variable constant $a/\bar b$** depends on the data, the number of classes
  and the capacity. Correcting for $n$ removes the count, not these. The result is
  dimension-corrected, not dataset-invariant. (Example: on 2-D spirals with enough
  capacity there is no accuracy knee at all.)
- **Finite training.** Trained models are not exact minimisers, so the dynamics, where
  the noise of 4.3 does matter, also shape the measured knees.
- **JEM.** Lemmas 1 and 2 hold for any normalised density. JEM, however, follows an
  SGLD estimate of $\nabla L_\text{gen}$, whose size is set by the sampler, so its knees
  may sit elsewhere.

## 5. The beta objective

$$
L_\beta = (1-\beta)\, L_\text{dis} + \beta\, \frac{L_\text{gen}}{N}, \qquad \beta \in [0, 1].
$$

$L_\text{gen}/N$ is the generative loss per modelled variable.

**Per-variable weights.** Substituting §1, the class term gets weight
$1 - \beta + \beta/N$ and each feature term $\beta/N$. These $N$ weights **sum to one**,
so $L_\beta$ is a weighted mean of the $N$ per-variable losses. The share of the weight
on the features is $s = \beta n / N$ (close to $\beta$), and

$$
L_\beta = (1 - s)\, L_\text{dis} + s\, \bar\ell.
$$

**Relation to alpha.** In $L(\alpha)$ the generative term has $\alpha/(1-\alpha)$ times
the weight of the discriminative one. In $L_\beta$ the term $L_\text{gen}/N$ has
$\beta/(1-\beta)$ times the weight, so $L_\text{gen}$ itself has $N$ times less.
Writing $r(\cdot)$ for these weight ratios, the two objectives are proportional exactly
when

$$
r(\beta) = \frac{\beta}{1-\beta} = N \cdot \frac{\alpha}{1-\alpha} = N\, r(\alpha),
\qquad
\alpha(\beta) = \frac{\beta}{N - (N-1)\,\beta},
\qquad
\beta(\alpha) = \frac{N\alpha}{1 + (N-1)\,\alpha},
$$

and then

$$
L_\beta = c(\beta)\, L\big(\alpha(\beta)\big),
\qquad c(\beta) = 1 - \beta + \frac{\beta}{N} = \frac{1}{1 + n\,\alpha(\beta)}.
$$

- **Same minimisers.** $L_\beta$ is a positive multiple of $L(\alpha(\beta))$: a
  relabelling of $\alpha$, different for each $n$, plus a change of scale.
- **On a log scale, a shift.** $\ln r(\beta) = \ln r(\alpha) + \ln N$. Plotted against the
  log weight ratio, the $\beta$ axis is the $\alpha$ axis shifted by $\ln N$.
- **Endpoints.** $\beta = 0 \iff \alpha = 0$ and $\beta = 1 \iff \alpha = 1$.
- **Middle.** $\beta = \tfrac12 \iff \alpha = 1/(N+1)$: the classifier term and the
  per-variable generative term weigh the same.
- **The prediction of §4.** Knees at fixed $\alpha n$ mean knees at fixed $\beta$:
  $r(\beta) \approx N\alpha$ for small $\alpha$.
- **Why $N$ and not $n$.** With $L_\text{gen}/n$ the weights would sum to
  $1 + \beta/n$, not one, so $L_\beta$ would not be a mean. The two choices differ by
  $\ln(N/n)$ on the log scale: 0.41 on 2-D spirals, 0.007 on MNIST 12×12.
- **Testing the exponent.** The general form $\ln r(\beta_p) = \ln r(\alpha) + p \ln N$
  gives $\beta$ ($p = 1$), a $\sqrt N$ variant ($p = \tfrac12$) and $\alpha$ ($p = 0$).
- **Adversarial training and JEM** have the same form, with the attack in the
  discriminative term only:

  $$
  (1-\beta)\, L_\text{dis}(x_\text{adv}) + \frac{\beta}{N}\, L_\text{gen}(x)
  = c(\beta)\, \big[(1-\alpha)\, L_\text{dis}(x_\text{adv}) + \alpha\, L_\text{gen}(x)\big],
  $$

  with the same $\alpha(\beta)$ and $c(\beta)$ (check: $c(1-\alpha) = 1-\beta$ and
  $c\,\alpha = \beta/N$).

## 6. Units

| Quantity | Definition | Unit |
| --- | --- | --- |
| $L_\text{dis}$ | $-\log p(c \mid x)$ | nats per label (one variable) |
| $\bar\ell$ | $L_\text{marg} / n$ | nats per feature |
| $L_\text{gen}/N$ | $(L_\text{dis} + n\bar\ell)/N$ | nats per variable |
| $L_\beta$ | $(1-s)\,L_\text{dis} + s\,\bar\ell$, weights sum to one | nats per variable, at every $\beta$ |
| $L(\alpha)$ | $L_\text{dis} + n\alpha\,\bar\ell$, weights sum to $1 + n\alpha$ | no common unit: nats per label at $\alpha = 0$, nats per sample of $N$ variables at $\alpha = 1$ |
| $\log Z / N$ | log normaliser per site | nats per site |

- Bits instead of nats: divide by $\ln 2$.
- $\bar\ell$ is a *continuous* (differential) NLL, so it depends on the input range:
  rescaling every feature by a factor $k$ shifts $\bar\ell$ by $\ln k$. On $[-1, 1]$ the
  uniform density scores $\ln 2 \approx 0.69$ nats per feature, so
  $\bar\ell - \ln 2$ reads as "nats per feature better than uniform" when negative.
  $L_\text{dis}$ is discrete: at least 0, and $\ln C$ at chance.
- Hence values of $\bar\ell$ compare across datasets only within one embedding range,
  while $L_\text{dis}$ compares everywhere.
- JEM's $L_\text{gen}$ has no normaliser, so its per-variable value is known only up
  to a constant.

## 7. Alpha and beta on the datasets used so far

The log weight ratio $\ln r(\beta)$ of the ladder
$\beta \in \{0.01, 0.1, 0.5, 0.9, 0.99\}$ is about $-4.6, -2.2, 0, 2.2, 4.6$: evenly
spaced, a factor ~10 in $r$ per step, symmetric about $\beta = \tfrac12$. Shifts of the
log weight ratio from $\alpha$ to $\beta$: $\ln 3 = 1.1$ (spirals), $\ln 145 = 5.0$
(MNIST 12×12), $\ln 785 = 6.7$ (MNIST 28×28).

$\beta$ in $\alpha$:

| $\beta$ | 0.001 | 0.01 | 0.03 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.97 | 0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spirals ($N = 3$) | 0.00033 | 0.0034 | 0.01 | 0.036 | 0.1 | 0.25 | 0.5 | 0.75 | 0.92 | 0.97 |
| MNIST 12×12 ($N = 145$) | 6.9e-06 | 7e-05 | 0.00021 | 0.00077 | 0.0023 | 0.0068 | 0.02 | 0.058 | 0.18 | 0.41 |
| MNIST 28×28 ($N = 785$) | 1.3e-06 | 1.3e-05 | 3.9e-05 | 0.00014 | 0.00042 | 0.0013 | 0.0038 | 0.011 | 0.04 | 0.11 |

$\alpha$ in $\beta$:

| $\alpha$ | spirals | MNIST 12×12 | MNIST 28×28 |
| --- | --- | --- | --- |
| 1e-5, 1e-4, 1e-3 | 3e-5, 3e-4, 0.003 | 0.0014, 0.014, 0.13 | 0.0078, 0.073, 0.44 |
| 0.01 | 0.029 | 0.59 | 0.89 |
| 0.1 | 0.25 | 0.94 | 0.99 |
| 0.2 | 0.43 | 0.97 | 0.995 |
| 0.5 | 0.75 | 0.993 | 0.999 |
