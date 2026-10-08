# β: review of the dimension-corrected objective

A review (2026-10-08) of "Planned: β" in [plan.md](plan.md), for Martin to read and fold
into plan.md, decisions.md and the paper. Nothing here is decided. Notation as in the
plan: n features, N = n + 1 modelled variables (the features and the class),
L_dis = −log p(c|x), L_gen = −log p(x, c), L_marg = −log p(x), each a mean over samples.

## Verdict

- **The reparametrisation is sound, and simpler than the plan writes it:**
  odds(β) = N · odds(α), i.e. logit β = logit α + ln N. β relabels α per dataset, with
  the same family of objectives and the same minimisers, so every α run is a β run (§3).
- **Q1 (is per-feature normalisation the right balance?): yes as a prior, but for a
  better reason.** α is an exact price, not a balance "up to a constant" (§1). The
  curvatures of the n per-feature terms add, so the price of the *mean* per-feature
  NLL is α·n: 1/N is the natural correction, not 1/√N (§2). √n is the scale of the
  gradient *noise*, which affects how the optimiser moves, not where the optimum lies.
  **Gradient-norm ratios cannot settle the question** (§2.3); experiment E2 can (§6). β
  removes the count n, but not the per-variable difficulty or the capacity: the
  objective becomes dimension-corrected, not dataset-invariant.
- **Q2 (N or n): N.** With N, the N per-variable weights sum to one, and β·n/N is
  exactly the share of the objective spent on modelling x (§3). The choice matters
  only on spirals (odds ×1.5).
- **Q3 (scale invariance): I disagree with changing the loss scale.** Every calibrated
  setting depends on today's scale: pilot A's norm penalty, JEM's gradient clip and
  energy penalty, Adam's ε compared with the tiny boundary-core gradients (D60), and the
  seams. **Recommended (C′): β is the only parameter in configs, names, keys and
  figures; the trainers optimise the normalised form (1−w)·L_dis + w·L_gen with
  w = α(β)** (§4).
- **Q4 (JEM):** the same map applies, because JEM's objective has the same linear form.
  JEM's logged L_gen is not an NLL, so JEM gets no per-variable number. Whether JEM's
  knees fall at the MPS's β is a result to report, not something to assume (§2.4).
- **Q5 (the paper):** one sentence and one row of numbers are enough (§5). In β, TPM's MNIST
  sweep α ∈ {0.01, 0.1, 0.2, 0.5} is β ∈ {0.59, 0.94, 0.97, 0.993}: all of it lies in
  the generative half.
- **Q6 (the ladder): expect two knees, not one balance point** (§5). As β grows, the
  density catches up first, at small β; the accuracy then drops at large β (on
  MNIST 12×12, somewhere between β = 0.59 and 0.94). The paper's interesting models
  sit between the two knees. Spirals has no accuracy knee at all. Locate both knees on
  MNIST before Phase 8 (E2), then fix the ladder.

## 1. α is a price

    L(α) = (1−α)·L_dis + α·L_gen = L_dis + α·L_marg        (L_gen = L_marg + L_dis)

At a minimiser θ*(α), ∇L_dis = −α·∇L_marg. Along the path of minimisers (envelope
theorem):

    dL_dis = −α · dL_marg

So α is exactly the exchange rate: the trained model gives up α nats of classifier
loss for each nat of marginal NLL it gains. No constant is involved. By the chain rule,
L_marg = n·ℓ̄, where ℓ̄ is the mean per-feature conditional NLL, so one nat of ℓ̄ is worth
α·n nats of L_dis. **The hypothesis behind β, stated exactly:** the trade-off between
L_dis and ℓ̄ has the same shape for every n, so the interesting region sits at a fixed α·n.

## 2. Why α·n, not α·√n

**2.1 Curvatures add.** Let s_i = ∇_θ log p(x_i | x_<i). The score of p(x) is Σ_i s_i,
and E[s_i | x_<i] = 0 under the model, so the cross terms vanish:

    F_marg = E[∇log p(x) ∇log p(x)ᵀ] = Σ_{i=1..n} F_i        F_gen = F_marg + F_dis

Each F_i is positive semi-definite. The Fisher of L_marg (its curvature near an
optimum) is therefore a sum of n terms that cannot cancel, and the classifier's F_dis
is one more term of the same kind (the class is the N-th variable). In an MPS every
conditional depends on every core, so the n terms share the parameters.

**2.2 The minimiser follows the curvature.** Take a direction in parameter space along
which L_dis has curvature a and L_marg has curvature b = Σ_i b_i ≈ n·b̄. Along it, the
minimiser of L_dis + α·L_marg sits a fraction α·b/(a + α·b) of the way from the
classifier's optimum to the density's, so it is halfway at α = a/(n·b̄). Hence
α_knee ∝ 1/n, and odds(β_knee) ≈ a/b̄ does not depend on n.

**2.3 Where √n comes from, and why the planned measurement fails.** The per-sample
score Σ_i s_i is a sum of orthogonal increments, so its norm grows like √n. That norm
is the gradient *noise*: it sets the optimiser's dynamics, not where the minimiser lies.
The plan's "between 1/n and 1/√n" mixes the two. The measurement the plan proposes,
|∇L_dis| against |∇L_marg|, cannot separate them:
- on a trained model, the full-batch ratio equals α by stationarity, so it checks
  convergence, not a scaling law;
- mini-batch norms measure the noise;
- at initialisation, the ratio describes the initialisation.

**2.4 Limits.**
- The count n applies where the classifier competes for capacity that all n
  conditionals share. Suppose only the ~ξ conditionals near a site respond to what the
  classifier needs (a correlation length ξ ≪ n). Then the price scales like ξ, so the
  exponent p in α_knee ∝ N^(−p) can lie anywhere in [0, 1]. p = 1 is the prior, not a
  theorem.
- β removes n but not a/b̄, which depends on the data, the number of classes and the
  capacity.
- Trained models are not exact minimisers. The measured knees therefore also reflect
  the dynamics, and there the gradient scales do matter. Pilot A shows this (E0).
- JEM: §2.1 holds for any density, but JEM follows a short-run SGLD estimate of
  ∇L_gen, whose size is set by the sampler. (The JEM paper optimises
  log p(y|x) + log p(x) = log p(x, y), which is our β = 1, and keeps ~93% on CIFAR-10.)
  Give both models the same β (same objective, D71) and report where each model's knees
  fall.

## 3. β

    L_β = (1−β)·L_dis + β·L_gen/N = c(β)·L(α(β)),       c(β) = 1 − β + β/N = 1/(1 + n·α)

    odds(β) = N·odds(α)   ⇔   logit β = logit α + ln N   ⇔   α(β) = β/(N − (N−1)·β)

- The endpoints stay: β = 0 ⇔ α = 0 and β = 1 ⇔ α = 1. In between, β = ½ ⇔ α = 1/(N+1).
- **How to read β, and why N:** L_β weights the class conditional by 1 − β + β/N and
  each feature conditional by β/N. These N weights sum to one, and **β·n/N (≈ β) is
  the share of the objective spent on modelling x**. With n in place of N, the weights
  do not sum to one. The two differ by ln(N/n) in logit: 0.41 on spirals, 0.007 on
  MNIST 12×12.
- **AT and JEM use the same map:** (1−β)·L_dis(x_adv) + (β/N)·L_gen(x) =
  c(β)·[(1−α)·L_dis(x_adv) + α·L_gen(x)], with the same α(β) and c(β) (since
  c·(1−α) = 1−β and c·α = β/N). The AT two-forward trick is written in α, so it needs no
  re-derivation.
- For the experiments, generalise to logit β_p = logit α + p·ln N: p = 1 is β, p = ½
  the √N variant, p = 0 is α.
- The per-variable NLL averages a density on [−1, 1] (the uniform density scores
  ln 2 = 0.69 per feature) with a discrete class term. That makes it a good weight, but
  only comparable as a number within one embedding range.

## 4. The loss scale (Q3): C or C′

L_β and L(α(β)) have the same minimisers. They differ by the factor c(β) ∈ [1/N, 1],
which is 1/785 at β = 1 on full MNIST. Adam is invariant to rescaling the whole loss and
to nothing else:

|Depends on the scale|Under C (optimise L_β)|
|---|---|
|Norm penalty, strength 0.1 (`NormRegularizer`)|its relative weight grows by 1/c, up to ×785 at β = 1; pilot A chose the target at this strength and today's scale (D84)|
|Adam ε = 1e-8 vs the ~1e-7 boundary-core gradients (D60)|for c < 1/10 (β ≳ 0.9 on MNIST) those gradients fall below ε, and those cores stop taking lr-sized steps: the optimiser then depends on β|
|JEM `grad_clip` 10, `energy_l2` 1e-4 (`core/jem/train.py:59`)|clipping fades out, and the energy penalty grows by 1/c as β → 1|
|Logged objectives, seam pins|all change|

The scale does not matter for selection, patience (a count, with strict <), median
pruning and HPO (all within one cell, where c is fixed), the lr ranges (Adam) or the
collapse checks (finiteness). The MPS trainer has no clipping and no weight decay.

- **C (as planned):** optimise L_β. To keep their meaning, the penalty, JEM's clip and
  JEM's energy penalty must be scaled by c(β); C then differs from C′ only in ε. Every
  seam with α > 0 needs a re-pin. Gain: at β = 1 the logged objective reads in nats per
  variable.
- **C′ (recommended):** β is the only parameter users see (configs, cell names, keys,
  figures). One function maps it to the weight w = α(β), and the trainers optimise
  (1−w)·L_dis + w·L_gen as today. The per-variable NLL becomes a logged or derived
  metric. Every calibration, pilot result and TPM number carries over exactly. Seams:
  α = 0.5 on spirals is exactly β = 0.75, so the JEM seams stay as they are; only the MPS
  AT seam (α = 0.01, which no round β gives) needs a re-pin.
- **D (cheapest):** keep α in the code and use β only to generate ladders and label
  axes. That leaves two names for one quantity, against "one parameter everywhere". Not
  recommended.

The best argument for C: it would make the NLL's gradient scale roughly independent of β
(O(1) curvature at both ends), whereas under C′ the scale grows about N-fold from β = 0
to β = 1, as it does today. Starting from scratch, C would be the better-conditioned
convention. But nothing here is calibrated for it, and Adam absorbs most of the
difference. The paper can state L_β (the readable form) and add that the code optimises
L_β/c(β), which has the same minimisers.

## 5. Ladders, knees and translations

**Two knees.** In the quadratic picture of §2.2, the parameter directions switch from
the classifier to the density one group at a time as β grows. First come the directions
the classifier hardly needs, and the density catches up. These directions are free at
any β > 0, given enough training time, so this knee may be set by the dynamics. Then
come the contested directions, and the accuracy drops. The evidence:

- MNIST 12×12, pilot A (d3r40, target n·ln d/2): test accuracy 98.1 / 76.2 / 73.8% at
  β = 0 / 0.94 / 1; test L_gen +62 / −138 / −142 nats.
- MNIST 12×12, TPM (d3r20): classification "markedly degrades beyond α ≈ 0.01"
  (β = 0.59, clean accuracy > 95%); at α = 0.5 (β = 0.993) clean accuracy is 56%.
- Spirals (d10r6), TPM: clean accuracy is near-perfect at every α. The capacity
  suffices, so there is no accuracy knee, only a density knee.

So on MNIST 12×12 the accuracy knee lies between β = 0.59 and 0.94. The density knee has
not been measured; it lies somewhere below 0.94, possibly below 0.01.

**The proposed ladder** {0, 0.01, 0.1, 0.5, 0.9, 0.99, 1} is evenly spaced in log-odds
(steps of ×~10) and symmetric about β = ½. On full MNIST it almost coincides with the
D84 α ladder (odds within ×1.4). On MNIST 12×12 it moves the D84 ladder up by ×6–8 in
odds. Its step from 0.5 to 0.9 spans both the known accuracy knee and TPM's best model
(β = 0.59), so it is too coarse there.

Between the two MNIST datasets the shift from α to β is ln(785/145) = 1.7 in logit,
less than one ×10 step (2.3). For MNIST alone, then, β changes little. It pays off as
one axis and one ladder across spirals and MNIST (a shift of 3.9–5.6, about two steps),
and as an exact translation of TPM.

The fine ladder of E2, in α:

|β|0.001|0.01|0.03|0.1|0.25|0.5|0.75|0.9|0.97|0.99|
|---|---|---|---|---|---|---|---|---|---|---|
|spirals (N = 3)|0.00033|0.0034|0.01|0.036|0.1|0.25|0.5|0.75|0.92|0.97|
|MNIST 12×12 (N = 145)|6.9e-06|7e-05|0.00021|0.00077|0.0023|0.0068|0.02|0.058|0.18|0.41|
|MNIST 28×28 (N = 785)|1.3e-06|1.3e-05|3.9e-05|0.00014|0.00042|0.0013|0.0038|0.011|0.04|0.11|

The α values used so far, in β:

|α|spirals|MNIST 12×12|MNIST 28×28|used in|
|---|---|---|---|---|
|1e-5, 1e-4, 1e-3|3e-5, 3e-4, 0.003|0.0014, 0.014, 0.13|0.0078, 0.073, 0.44|D84 ladder|
|0.01|0.029|0.59|0.89|TPM's best MNIST model; D82 AT grid|
|0.1|0.25|0.94|0.99|pilot A; D82 AT grid|
|0.5|0.75|0.993|0.999|TPM's "mixed" model|

A paper sentence (Q5): "We weight the two terms by β ∈ [0, 1], L_β = (1−β)·L_dis +
β·L_gen/N, where N = n + 1 counts the modelled variables, so that β·n/N is the share of
the objective spent on modelling x. Up to a positive factor this is the objective of
[TPM], with odds(α) = odds(β)/N; TPM's α ∈ {0.01, 0.1, 0.5} on 12×12 MNIST are
β ≈ {0.59, 0.94, 0.993}."

## 6. Experiments

**E0. Did pilot A's α = 0.1 runs converge? (Free: read `log.json` on the cluster.)**
On test, pilot A's α = 1 models (target n·ln d/2) score better on the α = 0.1 objective
than the α = 0.1 models themselves: −13.52 against −13.13 nats (means over 3 seeds).
They reach −142.3 against −137.8 test L_gen at a cost of only +0.07 in L_dis. So the
α = 0.1 runs stopped short of their own optimum. Check, per run:
- the best epoch against `max_epoch` = 100;
- whether `objective/valid` and the train `loss_gen` were still falling;
- the same comparison on the training split, which rules out generalisation.

If the curves were still falling, every intermediate-β cell of Phase 8 is limited by
its epoch budget. E2 must then either train longer or say that its knees are those of
budgeted training.

**E2. Locate the knees on MNIST and test the correction.** This decides the ladder; it
takes about two nights on G21G01. The studies are set up like the pilots:
- 1 seed, d3r20 (TPM's architecture; it fits 8 GB on full MNIST at batch 512);
- `max_epoch` 100 (or E0's answer), target n·ln d/2;
- analysis at ε = 0.1 only, no joint attack;
- the grid β ∈ {0, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 0.99, 1}: steps
  of ×3 in odds, twice as dense as the ladder, 12 cells.

The three studies:
- **E2a `pilot_beta12`**, MNIST 12×12: an lr-only HPO over `{log: [1e-4, 1e-2]}` (every
  lr pilot A picked lies inside), 6 trials, no pruning. This is random search, so all
  trials can start at once. Cost: 12 cells × 7 runs × ≤ 1.3 h ≈ 110 unit-h at most, one
  night at ~25 units.
- **E2b `pilot_beta28`**, full MNIST: no HPO; each cell takes the lr of E2a's cell at
  the same β. `hparams_from` matches cells by name, so this needs β implemented;
  without it, use one fixed lr, the median of E2a's picks. Cost: 12 runs × ≤ 7 h, one
  night.
- **E2c `pilot_beta_spirals`** (d10r6, normal HPO): a few minutes; gives the density
  knee at n = 2.

For each β, read the test accuracy, L_dis, ℓ̄ = L_gen/N, rob and purify at ε = 0.1, and
detection at q5. Plot each against logit β with the three datasets together, then again
against log α. Read off three points:
- the accuracy knee and the density knee: the β where acc and ℓ̄ each cross the middle
  of their range from β = 0 to β = 1, interpolated in logit β;
- β*, the β with the best purified accuracy (the paper's payoff; in TPM, α = 0.01).

What E2 decides:
1. **The normaliser.** For each knee, take the shift Δ = logit β_knee(28×28) −
   logit β_knee(12×12). p = 1 predicts Δ ≈ 0, p = ½ predicts +0.85, and p = 0 (α)
   predicts +1.7. One seed and the capacity difference (r = 20 on 784 sites against 144)
   blur Δ by about ±0.5. E2 therefore separates p = 1 from p = 0, but cannot cleanly
   separate either from ½. Keep N unless both knees shift by ≳ +1. A prediction to
   check: if p = 1, D82's α = 0.01 on full MNIST (β = 0.89) is at or past the accuracy
   knee, although on 12×12 the same α is the best model.
2. **The Phase 8 ladder:** at most 7 cells (cost), with at least 2 interior points
   between the density knee and the accuracy knee, and one point beyond each.
3. **The AT grid:** {0, β*}, plus one more value if affordable, decided together with
   pilot B's verdict. This answers the open question in AGENTS.md's next step 1: D82's
   α = 0.1 is β = 0.94 (12×12) and 0.99 (28×28), past the knee.

**E1 (optional, for the paper's "why N").** Take E2's checkpoints at
β ∈ {0.1, 0.5, 0.9} for 12×12 and 28×28. Over 256 test points, compute the per-sample
ratio E|∇log p(x)|² / E|∇log p(c|x)|², a ratio of Fisher traces. §2 predicts that it
grows ×5.4 (= 785/145) from 12×12 to 28×28, and that the ratio of norms grows ×2.3.
Caveat: a trace ratio depends on the parametrisation (the MPS gauge), so it supports
the argument but decides nothing. About 1 h on the laptop.

**Order:** E0 now, then decide C or C′ and implement it (under C′ this is mostly
renaming, one function and one re-pin). After pilot B, run E2 in β; E2a and E2c can start
earlier in α, using the table in §5. The ladder, the AT grid and the normaliser then
become a D-number. Phase 8 provides the rest for free: every NAT study runs the β
ladder, so the final cross-dataset and cross-capacity picture (r10–r40 on 12×12, JEM
beside MPS) comes from the studies themselves.

## 7. Implementation under C′ (where it differs from plan.md's list)

1. `objective.gen_weight(beta, n_vars)` = β/(N − (N−1)·β), exact at 0 and 1. It is the
   only place where β becomes a weight. N = data_dim + 1 for both models (the MPS's
   `n_features`).
2. `mixed_nll`, `mix`, `evaluate` and the AT trick keep their maths; their `alpha`
   argument is renamed to the weight `w`.
3. Trainer and JEMTrainer compute w = gen_weight(cfg.beta, N) once. Norm control is
   unchanged.
4. `TrainConfig.beta`, `Cell.beta`, names `b0.5`, grids, `paper.yaml`, `results.csv` and
   axis labels change as planned (D12 clean break).
5. Optionally, a metric for L_gen/N, built in `pipeline/metrics.py` (or derived in the
   figures).
6. Seams: the MPS AT seam gets a round β and is re-pinned in a commit of its own. The JEM
   seams (α = 0.5 → β = 0.75, w = 0.5 exactly) and the α = 0 seams stay bit-identical.
7. Tests: gen_weight at the endpoints and the odds identity; L_β = c·L(α(β)) on a batch.
