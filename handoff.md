# Handoff: sampling and tensorkrowch reuse (2026-10-14)

Session `ousterhout-sampling-efficiency`. Fold into AGENTS.md "State" (maintained by
the E2 session) and delete this file.

## State

- **Sampling done (D100–D103).** `SamplingMPS` (`bm4tc/core/sampling.py`) is tk
  develop's `condition`/`sample` on tk 1.1.6, made efficient: per-sample domains,
  each site's density matrix first, one layer for a sweep, uniform draw where no
  weight is usable, nodes kept between calls. `GibbsPurification` runs one
  `born.sample(in_condition=x, domain=windows)` per sweep; the figures'
  class samples are `condition` + `sample`. Gibbs per analysis run (250 points, 10
  sweeps): ≈ 9 s MNIST 12×12, ≈ 50 s full MNIST (did not fit before D100).
- **D104, the forward keeps its nodes.** `ConditionalBornMachine.forward` calls
  `contract()` without reset, recorded like tk's `trace`; `_nodes_key` says whether
  the forward or a `sample` structure holds the network. Bit-identical; eager step
  174 → 130 ms, eager analysis 81 → 53 s, captured training unchanged. Reserved
  memory +0.8 GiB per process at E2a's shape (allocated unchanged), cause not found.
- **tensorkrowch fork:** `../tensorkrowch`, branch `sampling-efficiency` (local), the
  sampler in develop's names (4 commits) and **`TK_NOTES.md`**: what we learned about
  tk internals (successors, forward's replay, trace's inverse memory, reset, node and
  edge rules, 1.1.6 vs develop, candidates). Read it first next session.
- **Not pushed (ask Martin):** bm4tc `7fe4a4e2`..HEAD on `ousterhout` (includes the
  E2 session's `3b6afd73`, `06d9af81`); tk `0a079fa`..`b640880`.
- compute.md marks the numbers D104 made stale and has the new ones ("Since D104").

## To test on the cluster (G21G01, a worktree at the D104 commit)

1. `pytest tests/unit/test_graphs.py` (captured vs eager, pools, E2a's shape): D94's
   rule, and the first run of D104 under torch 2.1 there.
2. `python -m tests.bench.bench_train_step` at every shape that will launch, captured,
   NAT and AT, comparing memory per process with compute.md's table:
   - MNIST 12×12 d3r40 `--batch 512 --beta 0.5` (E2a; AT with PGD-5 does not fit the
     laptop);
   - full MNIST `--features 784 --bond-dim 40 --batch 512 --micro-batch 256 --beta 0.5`
     (E2b; 5.34 / 7.58 GiB NAT / AT before, 9.3 GiB per process in E2b's real run);
   - d3r80 `--micro-batch 128` (did not fit: does it now?).
   The question is D104's +0.8 GiB reserved per process: if it holds, `--per-gpu`
   changes; if it is worse at 784 sites, find the cause before launching (allocator
   snapshot of the Trainer's warm-up and validation, `test_graphs.py`'s helpers).
3. `python -m tests.bench.bench_analysis --features 784 --bond-dim 40` (E2b's analysis
   needed ≥ 10.7 GiB).
4. One short `run ... --mps --per-gpu N` pilot: per-process memory in `nvidia-smi`
   with several units, as D95 measured it.
5. Gibbs on full MNIST in a real analysis (`analysis.sweep_purify.enabled`), timing
   against compute.md's ≈ 50 s per run.

## Next

1. The cluster checks above (Martin runs them; no agents on the HPC).
2. E3 (trimmed adaptive attacks on spirals, `setsid nohup`, scratch
   `BM4TC_DATA_ROOT`), then its saturation ablation.
3. **Next session: tensorkrowch, intensely** (start from `TK_NOTES.md`): candidates
   are a norm-keeping `renormalize` (log-norm as a node; would remove bm4tc's
   `_inline_contraction` override, the accumulator and the special forward), a
   public "contract with reuse" (forward's replay forbids Python side effects; trace's
   bookkeeping is reachable only through private flags), `reset`'s quadratic renaming
   and its reordering of `parameters()`, `@` on batch-only edges, `einsum`'s
   connected-edge rule, edge inheritance through result nodes, the env-first forward
   (B·n·r·d instead of B·n·r² stored). Then decide what goes upstream and how
   (PR to the fork's develop, then to tensorkrowch).

## Open items

- D104's reserved memory (above).
- `tests/e2e/test_stages.py::test_run_in_parallel_then_status` failed once with 4
  trials for 3 (the known worker race, plan.md: cap HPO workers per cell).
- AGENTS.md still says D1–D99 and lists the sampling as next.
