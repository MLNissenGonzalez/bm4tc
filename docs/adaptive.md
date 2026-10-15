# Adaptive attacks and robustness ceilings

What the analysis's attacks are allowed to claim, and the bounds that check them (D96,
D97). The code is `bm4tc/analysis/adaptive.py` (the attacks) and
`bm4tc/analysis/ceiling.py` (the bounds).

## 1. Defences and what is measured

A defence is a triple (P, f, D):

- P, the purifier: likelihood purification P_δ (deterministic: no random start), Gibbs
  with k sweeps or SGLD (random), or the identity.
- f(z) = argmax_c p(c | z), the classifier.
- D, the detector: it accepts iff log p(·) ≥ τ_q, read on the input x or on P(x).
  τ_q is the q-th percentile of clean log p(x) on valid (D2). The attacker knows τ_q
  (white box).

The defended output is h(x) = ⊥ (reject) if D rejects, else f(P(x)). The attacker picks
x' in B∞(x, ε) ∩ [lo, hi]ⁿ. A test point (x, y) scores:

- **robust accuracy (no rejection):** 1[h(x') = y]. These are `rob` (P = identity, no
  D) and `purify`. For a random P the attacker fixes x' before the draw; the score is
  P_ω[h(x') = y].
- **selective robust accuracy** (Tramèr 2022): 1[h(x) = y] · 1[h(x') ∈ {y, ⊥}]. The
  clean point must be accepted and correct, and the attacked point must be rejected or
  correct. Detection is judged by this one number: a detection rate and an error rate
  among passed examples are not bounded by anything, and a detector that rejects
  everything scores 0 here, not 1.

## 2. Ceilings

### 2.1 On the test set, without rejection

Two test points of opposite class within 2ε of each other (L∞) share a point z within ε
of both. No h is right at z for both labels. This holds for a random h too, since the two
probabilities of being right sum to at most 1. So every edge of a matching in the
conflict graph forces one error:

    robust accuracy ≤ 1 − |M| / n.

For two classes the graph is bipartite, and the maximum matching gives the tightest bound
of this kind (König). It is also attained on the sample: the unmatched points form an
independent set, and labelling their ε-balls by class makes all of them robustly correct.

What it can be trusted for:

- It is a hard bound on any measured accuracy without rejection on that test set: any
  model, any purifier, any randomness. A measured value above it means that the attack
  missed adversarial examples that exist. It never means robustness.
- It is tight only for a classifier that knows the test points, so it is optimistic
  about what a model can reach: E[test ceiling] ≥ the population ceiling (2.3). On
  spirals at ε = 0.1 it is 0.676, against ≈ 0.652 for the population.
- It does not bound selective accuracy (2.2).
- The shared point z must lie in the input box, because attacks clamp to it. It does:
  the scaler clips valid and test points into the box (`clip=True`), and the midpoint of
  two points in a box is in the box.

### 2.2 With rejection: the ceiling at ε/2

Let x_i and x_j be of opposite class and within ε of each other. Suppose both were
selectively robust. Then x_j is accepted with label y_j. But x_j lies in B(x_i, ε), so it
must also be rejected or labelled y_i, a contradiction. The matching argument therefore
holds with threshold ε instead of 2ε:

    selective robust accuracy ≤ ceiling at ε/2.

This is the data-only form of "detecting adversarial examples is as hard as classifying
them at half the radius" (Tramèr 2022). It is tight on the sample (accept only within
r → 0 of the independent set). On spirals test (2000 points) the two ceilings are:

| ε | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|
| without rejection | 0.998 | 0.676 | 0.629 | 0.568 | 0.504 |
| with rejection | 1.000 | 0.998 | 0.766 | 0.676 | 0.629 |

Detection has far more room than classification. Whether a model uses that room is
what the adaptive detector attack must show.

### 2.3 Population ceiling (2-D data with a known law)

With equal priors, the best robust accuracy any classifier attains on the data
distribution is

    acc*(ε) = 1 − ½ · F_2ε(p₀, p₁),

where F_2ε is the largest mass that can be coupled between the class laws p₀ and p₁ with
every pair at most 2ε apart. This is optimal transport with the 0–1 cost 1[d > 2ε]
(Bhagoji, Cullina & Mittal 2019; Pydi & Jog 2020). The selective version is acc*(ε/2).
The test-set ceiling is the same quantity computed on the empirical laws.

The spirals law is the generator in `bm4tc/pipeline/data.py`:

- θ = 2π√U, so p(θ) = θ / (2π²) on [0, 2π];
- class 0 has mean μ₀(θ) = (2θ + π)(cos θ, sin θ), class 1 has μ₁ = −μ₀;
- the noise is N(0, σ² I) in raw units;
- the train-fitted MinMaxScaler is affine per axis, so the noise stays Gaussian and
  separable in model units. It also clips into the box, so the law is the clipped one:
  the outer cells hold the tails, and each class's cell masses sum to 1.

The mass of a class in a grid cell of side h is a quadrature over θ of a product of two
erf differences. For the whole grid that is one matrix product per class
(`DataHandler.class_masses`).

On the grid, F is a max-flow: from the source to the class-0 cells (capacity: their
mass), then to the class-1 cells within reach, then to the sink (capacity: their mass).
Cells blur distances by up to h per axis, which brackets F with certainty:

- connect cells whose farthest points are within 2ε: a lower bound on F;
- connect cells whose nearest points are within 2ε: an upper bound on F;
- hence acc* lies in [1 − F_hi/2, 1 − F_lo/2], and the bracket shrinks linearly with h.

Within reach means at most K cells away on each axis. Splitting that move into "≤ K
along x", then "≤ K along y" is exact, and costs O(cells · K) edges instead of
O(cells · K²). `population_ceiling` reports the bracket. The pipeline logs its upper end
as `rob_ceiling/pop/{eps}` (`rob_ceiling_sel/pop/{eps}` with rejection).

The default spirals (`configs/dataset/spirals.yaml`, legendre's [−1, 1]) at 400 cells
per axis. "Test" is the matching ceiling on the 2000 test points, "pop" the certified
bracket on acc*:

| ε | test | pop | test, selective | pop, selective |
|---|---|---|---|---|
| 0.025 | 1.000 | [1.000, 1.000] | 1.000 | [1.000, 1.000] |
| 0.05 | 0.998 | [0.978, 0.987] | 1.000 | [1.000, 1.000] |
| 0.075 | 0.766 | [0.709, 0.729] | 1.000 | [0.999, 0.999] |
| 0.1 | 0.676 | [0.651, 0.654] | 0.998 | [0.978, 0.987] |
| 0.125 | 0.655 | [0.625, 0.628] | 0.930 | [0.866, 0.900] |
| 0.15 | 0.629 | [0.598, 0.601] | 0.766 | [0.709, 0.729] |
| 0.175 | 0.599 | [0.569, 0.572] | 0.694 | [0.668, 0.673] |
| 0.2 | 0.568 | [0.540, 0.543] | 0.676 | [0.651, 0.654] |
| 0.25 | 0.504 | [0.500, 0.500] | 0.655 | [0.625, 0.628] |
| 0.3 | 0.504 | [0.500, 0.500] | 0.629 | [0.598, 0.601] |

Computed with `DataHandler.class_masses(np.linspace(-1, 1, 401))`, then
`population_ceiling(m[0], m[1], 0.005, 2 * eps)` (and `selective=True`); 3–30 s per value.
The pipeline uses 200 cells (`analysis.rob_ceiling_cells`, 1–3 s per value), which
widens the bracket to 0.005–0.017 where the curve is steep (ε ≈ 0.075, selective
ε ≈ 0.15). The test ceiling is 0.02–0.06 above the population one wherever either is
below 1, the optimism of §2.1.

The population ceiling is the reference for "how far from optimal". A test-split number
compares with it up to sampling error: with 2000 test points, one standard error on a
rate near 0.6 is ≈ 0.011. The test-set ceiling stays the per-sample hard flag.

### 2.4 MNIST

MNIST has no generating law, so it has no population ceiling. The matching bound still
holds for 10 classes, with a general matching, but it carries no information: on 28×28,
opposite-class images are ≈ 0.74 apart in L∞ (in [0, 1] pixel units; Yang et al. 2020,
"A closer look at accuracy vs. robustness"). That is far more than 2ε for every budget up
to 0.3, so the ceiling is 1. Trust in the MNIST adaptive numbers therefore comes from
validating the same attacks on spirals.

## 3. Adaptive attacks

The principles are those of Athalye, Carlini & Wagner (2018), Tramèr et al. (2020),
Bryniarski et al. (2022) and Croce et al. (2022): attack the whole defence, with its
real forward pass, and score the success criterion of the metric, not a proxy.

- **Loss:** the margin m(z) = log p(y | z) − max_{c' ≠ y} log p(c' | z), evaluated on
  the point that f actually classifies. The attack succeeds where m < 0. It descends on
  m.

- **Purifier attack, BPDA + EOT.** The forward pass runs the real purifier; the backward
  pass treats it as the identity, written straight-through as
  z = x' + (P(x') − x').detach().
  - Likelihood purification with L∞ sign steps: for a fixed sign pattern and fixed box
    clamps, P_δ(x) = clamp(x + c). Its Jacobian is the identity, or 0 on coordinates
    that the box clamps, so the identity backward is the exact gradient almost
    everywhere. **That is true and misleading** (E3, §5). The 40 sign steps of size
    2.5δ/40 oscillate around the ridge of the density, so P is a staircase of tiny
    translations: the identity on each step, but at any larger scale it snaps inputs
    onto the ridge (contracting across it). The identity backward then overstates how
    far moving x' across an arm moves P(x').
  - Gibbs and SGLD: with the random choices held fixed, the purified point is again
    x + const (both work in windows centred on the current state). So the identity
    backward is the pathwise gradient; the dependence of the draw probabilities on x
    (the score-function term) is dropped. EOT averages the margin over K draws per
    step. The same caveat applies: the more sweeps, the further the draws pull points
    towards the density, and the worse the identity backward fits.
  - A random defence's x' is chosen with the attack's own estimate, then scored with M
    fresh draws. The maximum over noisy estimates is biased; fresh draws are not.

- **Detector attack, selective PGD.** Per sample, the attack steps on log p − τ (up) while
  the input is rejected, and on the margin once it is accepted. There is no Lagrange
  weight to tune. log p is read where the detector reads it: on x', or on P(x') for
  purify + detect, which then combines both attacks.

- **For every attack:**
  - the first restart starts from the PGD example on the bare classifier (D98), so the
    attack can only improve on the transfer attack;
  - the best iterate per sample is kept: any iterate in the ball at which the defence
    errs (and accepts) is a valid adversarial example;
  - R random restarts;
  - the attack is done when the defence errs and accepts.

- **JOINT_PGD** (D72) adds log p(x) to the class loss with a fixed weight. It ignores the
  threshold and the purifier, so it is not an adaptive attack in this sense; it stays as
  a cheap extra attack.

## 4. Checking the attacks on spirals

There is no attack-specific reference (no grid search: D96). The checks are:

1. Every adaptive number lies under its test ceiling, and within about two standard
   errors of the population ceiling at most.
2. Saturation: doubling steps, restarts or EOT draws moves the numbers by less than
   0.005.
3. With P = identity and no detector, the adaptive attack reproduces PGD on the
   classifier.
4. The known gap closes: E2c's purified accuracy of 0.82 at ε = 0.1, above the 0.676
   ceiling, must fall under it.

Only once these hold do the same attacks report numbers on MNIST.

## 5. First pilot (E3, 2026-10-13): what it showed

`configs/studies/pilot_adaptive_spirals.yaml`, run on the eGPU. The analysis was read
on β = 0 at ε = 0.05, with steps 100, restarts 5 and EOT 16 (higher numbers mean a
weaker attack).

| defence | PGD on f, then the defence | adaptive |
|---|---|---|
| detector q1 / q5 / q10 / q20 (`rob_sel`) | 0.9870 / 0.9500 / 0.8970 / 0.8040 | 0.9865 / 0.9490 / 0.8970 / 0.8040 |
| likelihood purification d0.1 (`purify`) | 0.978 | **0.9925** |
| purify + detect q1 … q20 (`purify_sel`) | 0.978 … 0.9715 | **0.9915 … 0.986** |
| Gibbs k1 / k3 / k6 | 0.948 / 0.896 / 0.888 | 0.892 / 0.876 / **0.903** |

- **The detector attack works:** it is at least as strong as PGD at every threshold,
  in 8 s per threshold.
- **BPDA through likelihood purification is weaker than the transfer attack**, which
  fails check 4's spirit: an attack that knows the defence must not lose to one that
  ignores it. The cause is the staircase of §3.
- **Gibbs:** BPDA + EOT beats the transfer attack at 1 and 3 sweeps, and loses at 6.
- **Cost:** about 2.5 min per likelihood-purifier defence per budget; Gibbs takes
  130 / 239 / 416 s at k1 / k3 / k6. One run then takes about 2.5–3 h, mostly Gibbs.

Fixes considered:
1. **Warm-start the adaptive attack from the PGD example**, as one of its restarts, so
   it can only improve on the transfer attack: the attacker's best attack per point,
   as in AutoAttack's ensemble. **Adopted (D98).**
2. Also evaluate the margin at x' as well as at P(x') (a hybrid loss), so the
   gradient pulls towards f's boundary too. Cheap; a heuristic. Kept in reserve.
3. Unroll the purifier with smooth steps for the backward pass (the true contracting
   Jacobian). Principled, but more code and second-order autograd through
   tensorkrowch. Kept in reserve.

That first run was stopped. The next E3 (not yet run) is trimmed to its questions:
detection at q5 only, Gibbs at sweeps {1, 6} with EOT 8 (about 40 min per run, 1.5–2 h
for the study). It asks: is adaptive purified accuracy under the ceilings at every β,
in particular E2c's 0.82 at high β and ε = 0.1? Does detection survive the adaptive
attack, against its higher ceiling? Which β is best under adaptive attack? Then the
saturation ablation (steps, restarts, EOT) sets `analysis.adaptive`'s defaults.

## 6. E3 (2026-10-15): the trimmed pilot

Run on G21G01 (GPU 3, 6 units under MPS, worktree `runs/34acaa91`, 55 min for the
study); results in `pilots/cluster/pilot_adaptive_spirals/` (untracked). Spirals,
d10r6, E2c's lr, one seed, 6 β; detection at q5, likelihood purification at δ = 0.1,
Gibbs at 1 and 6 sweeps (250 points), adaptive attacks with D98's warm start, EOT 8.
Clean accuracy 1.0 (0.9995 at β ≥ 0.9); clean purified 1.0, clean Gibbs k6
0.976–1.0.

At ε = 0.1 (ceiling: test 0.676, population 0.655; selective population 0.989):

| β | PGD (`rob`) | purify: transfer / adaptive | Gibbs k1 | Gibbs k6 | detect q5 (`rob_sel`): transfer / adaptive |
|---|---|---|---|---|---|
| 0 | 0.579 | 0.610 / 0.465 | 0.676 / 0.335 | 0.696 / 0.514 | 0.822 / 0.720 |
| 0.01 | 0.511 | 0.622 / 0.443 | 0.648 / 0.325 | 0.704 / 0.475 | 0.893 / 0.881 |
| 0.1 | 0.559 | 0.689 / 0.421 | 0.684 / 0.319 | 0.732 / 0.471 | 0.925 / 0.902 |
| 0.5 | 0.523 | 0.716 / 0.493 | 0.728 / 0.350 | 0.744 / 0.512 | 0.939 / 0.908 |
| 0.9 | 0.543 | 0.828 / 0.555 | 0.824 / 0.369 | 0.828 / 0.555 | 0.947 / 0.914 |
| 1 | 0.544 | 0.842 / 0.397 | 0.832 / 0.396 | 0.844 / 0.505 | 0.945 / 0.918 |

At ε = 0.15 (ceiling 0.630 / 0.602; selective 0.74): PGD 0.35–0.45; purify transfer
0.46–0.70, adaptive 0.08–0.22; Gibbs k6 transfer 0.57–0.72, adaptive 0.21–0.30;
detection transfer 0.55–0.91, adaptive 0.26 (β 0) to 0.52 (β 1). At ε = 0.2 the same
pattern, lower. At ε = 0.05 everything is within 0.06 of the ceiling (0.989).

- **Every adaptive number is under both ceilings, at every β and ε.** The transfer
  numbers were not: purification at β ≥ 0.9 and ε = 0.1 (0.83–0.84, E2c's 0.82) and
  detection at ε = 0.15–0.2 (up to 0.91 and 0.84, ceilings 0.74 and 0.655) are above
  what any classifier can reach. They measured a weak attack, not the defence.
- **Purification does not hold under adaptive attack.** Likelihood purification and
  Gibbs fall to about the undefended model's accuracy at ε = 0.1 (0.32–0.56 against
  PGD's 0.51–0.58) and far below it at ε ≥ 0.15 (purify 0.04–0.22 against 0.34–0.45).
  The attack's input stays within ε of the clean point, like PGD's (projected onto
  the L∞ ball and the input box at every step; `test_points_stay_in_the_ball_and_the_box`).
  The likely reading: the purifier then moves that input by up to δ = 0.1 on its own,
  uphill in log p(x), and the attacker places its input where that climb ends on the
  wrong side of f's boundary. The purified point may lie up to ε + δ from the clean
  one, but the extra δ is the defence's own move, not the attacker's.
- **Detection holds best, and grows with β:** adaptive selective accuracy at ε = 0.1
  rises from 0.72 (β = 0) to 0.91–0.92 (β ≥ 0.5), under the selective ceiling 0.989;
  at ε = 0.15, 0.26 → 0.52 (ceiling 0.74). The detector attack (selective PGD) costs
  detection 0.01–0.1 at ε = 0.1 and 0.25–0.4 at ε ≥ 0.15 against the transfer attack.
- **D98 works:** the adaptive attack is never weaker than the transfer attack for the
  deterministic defences. The one exception, Gibbs k6 at β = 1 and ε = 0.05 (0.956 vs
  0.952), is one point of 250, within Gibbs's sampling noise.
- **β:** under adaptive attack, detection prefers β ≥ 0.5 (0.5, 0.9 and 1 agree within
  one seed); purification has no β where it beats the undefended model at ε ≥ 0.1.
- **float32 (D106):** no "negative beyond numerical tolerance" in any of the 6
  analyses (spirals models are 2 sites; not evidence for MNIST).
- A logging bug, results unaffected: the `gibbs` part reruns the attack without
  thresholds, and `next(iter(det.values()))` on the empty detection dict logged
  "Detection/attack failed: ; skipping" per budget (the adversarial examples were
  cached before it, so every number was computed). Fixed after E3.

## References

- A. Athalye, N. Carlini, D. Wagner. Obfuscated gradients give a false sense of security.
  ICML 2018.
- A. N. Bhagoji, D. Cullina, P. Mittal. Lower bounds on adversarial robustness from
  optimal transport. NeurIPS 2019.
- O. Bryniarski, N. Hingun, P. Pachuca, V. Wang, N. Carlini. Evading adversarial example
  detection defenses with orthogonal projected gradient descent. ICLR 2022.
- F. Croce, S. Gowal, T. Brunner, E. Shelhamer, M. Hein, T. Cemgil. Evaluating the
  adversarial robustness of adaptive test-time defenses. ICML 2022.
- M. S. Pydi, V. Jog. Adversarial risk via optimal transport and optimal couplings.
  ICML 2020.
- F. Tramèr. Detecting adversarial examples is (nearly) as hard as classifying them.
  ICML 2022.
- F. Tramèr, N. Carlini, W. Brendel, A. Madry. On adaptive attacks to adversarial example
  defenses. NeurIPS 2020.
- Y.-Y. Yang, C. Rashtchian, H. Zhang, R. Salakhutdinov, K. Chaudhuri. A closer look at
  accuracy vs. robustness. NeurIPS 2020.
