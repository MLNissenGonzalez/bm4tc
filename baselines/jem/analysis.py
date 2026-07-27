"""Post-hoc analysis for one JEM run, aligned with the MPS metric schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.datahandler import DataHandler

from .attacks import PGDConfig, pgd_classification, pgd_likelihood_aware
from .device import resolve_device
from .model import JEMMLP
from .purification import (
    PurificationConfig,
    gradient_purify,
    sgld_purify_snapshots,
)
from .sampler import ReplayBuffer, SGLDConfig, SGLDSampler
from .trainer import evaluate_classifier


def _checkpoint(run_dir: Path) -> Path:
    for candidate in (run_dir / "models/model.pt", run_dir / "models/model"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model checkpoint below {run_dir}")


def _load(run_dir: Path, device):
    cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")
    model, extra = JEMMLP.load(_checkpoint(run_dir), device=device)
    datahandler = DataHandler(cfg.dataset)
    datahandler.load()
    datahandler.split_and_rescale(model)
    datahandler.get_classification_loaders(batch_size=256)
    sgld_cfg = SGLDConfig(**OmegaConf.to_container(cfg.sampler, resolve=True))
    buffer = ReplayBuffer(
        sgld_cfg.buffer_size,
        datahandler.data_dim,
        model.input_range,
        seed=int(cfg.tracking.seed),
    )
    if "replay_buffer" in extra:
        buffer.load_state_dict(extra["replay_buffer"])
    return cfg, model, datahandler, SGLDSampler(sgld_cfg, buffer)


def _collect(loader) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*[(x, y) for x, y in loader])
    return torch.cat(xs), torch.cat(ys)


def _score_batches(model, data: torch.Tensor, device, batch_size=256) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            out.append(model.marginal_score(data[i : i + batch_size].to(device)).cpu())
    return torch.cat(out)


def _pred_batches(model, data: torch.Tensor, device, batch_size=256) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            out.append(model(data[i : i + batch_size].to(device)).argmax(1).cpu())
    return torch.cat(out)


def _attack_dataset(model, data, labels, attack_fn, attack_cfg, device, **kwargs):
    chunks = []
    for i in range(0, len(data), 256):
        x = data[i : i + 256].to(device)
        y = labels[i : i + 256].to(device)
        chunks.append(attack_fn(model, x, y, attack_cfg, **kwargs).cpu())
    return torch.cat(chunks)


def analyze_run(
    run_dir: str | Path,
    *,
    device: str = "auto",
    attack_eps: tuple[float, ...] = (0.1, 0.2, 0.3),
    percentiles: tuple[int, ...] = (1, 5, 10, 20),
    radii: tuple[float, ...] = (0.2, 0.3),
    sampling_radius: float = 0.2,
    sampling_sweeps: tuple[int, ...] = (1, 3, 5),
    sampling_steps_per_sweep: int = 20,
    sampling_step_size: float = 0.01,
    sampling_noise_std: float = 0.005,
    sampling_subsample: int | None = 1000,
    defense_subsample: int | None = None,
    threshold_split: str = "test",
    adaptive_score_weight: float = 1.0,
) -> dict[str, float | str]:
    run_dir = Path(run_dir)
    dev = resolve_device(device)
    cfg, model, dh, sampler = _load(run_dir, dev)
    model.eval()
    results: dict[str, float | str] = {}

    core = evaluate_classifier(model, dh.classification["test"], dev)
    results.update(core)
    results["uq_clean_accuracy"] = core["acc"]
    raw_alpha = cfg.trainer.get("alpha")
    alpha = float(raw_alpha) if raw_alpha is not None else float("nan")
    results["alpha"] = alpha
    results["parameters"] = float(model.count_parameters())
    sampling_sweeps = tuple(sorted(set(int(value) for value in sampling_sweeps)))
    if not sampling_sweeps or sampling_sweeps[0] < 1:
        raise ValueError("sampling_sweeps must contain positive integers")
    if (
        sampling_radius <= 0
        or sampling_steps_per_sweep < 1
        or sampling_step_size <= 0
        or sampling_noise_std < 0
    ):
        raise ValueError("sampling purification parameters are invalid")
    if sampling_subsample is not None and sampling_subsample < 1:
        raise ValueError("sampling_subsample must be positive or None")
    if defense_subsample is not None and defense_subsample < 1:
        raise ValueError("defense_subsample must be positive or None")
    results["sgld_purify_mode"] = "local_sweeps"
    results["sgld_purify_radius"] = float(sampling_radius)
    results["sgld_steps_per_sweep"] = float(sampling_steps_per_sweep)
    results["sgld_purify_step_size"] = float(sampling_step_size)
    results["sgld_purify_noise_std"] = float(sampling_noise_std)

    test_x, test_y = _collect(dh.classification["test"])
    # JEM cannot evaluate normalized joint NLL. For alpha>0 report the
    # decomposed CD surrogate using the saved, trained replay buffer.
    if np.isfinite(alpha) and alpha > 0:
        n_test = len(test_x)
        repeats = (n_test + len(sampler.buffer.data) - 1) // len(sampler.buffer.data)
        negatives = sampler.buffer.data.repeat(repeats, 1)[:n_test]
        with torch.no_grad():
            pos_joint, pos_score, neg_score = [], [], []
            for i in range(0, n_test, 256):
                xb = test_x[i : i + 256].to(dev)
                yb = test_y[i : i + 256].to(dev)
                logits = model(xb)
                pos_joint.append(logits.gather(1, yb[:, None]).squeeze(1).cpu())
                pos_score.append(torch.logsumexp(logits, dim=-1).cpu())
                neg_score.append(
                    model.marginal_score(negatives[i : i + 256].to(dev)).cpu()
                )
        positive_joint = float(torch.cat(pos_joint).mean())
        positive_marginal = float(torch.cat(pos_score).mean())
        negative_marginal = float(torch.cat(neg_score).mean())
        px_cd = negative_marginal - positive_marginal
        joint_cd = negative_marginal - positive_joint
        results.update(
            {
                "positive_joint_score": positive_joint,
                "positive_marginal_score": positive_marginal,
                "negative_marginal_score": negative_marginal,
                "px_cd_loss": px_cd,
                "joint_cd_loss": joint_cd,
                # Kept for the common MPS CSV schema, with kind made explicit.
                "gen_loss": joint_cd,
                "mixed_loss": results["dis_loss"] + alpha * px_cd,
                "gen_loss_kind": "joint_cd_replay_surrogate",
            }
        )
    else:
        results.update(
            {
                "positive_joint_score": float("nan"),
                "positive_marginal_score": float("nan"),
                "negative_marginal_score": float("nan"),
                "px_cd_loss": float("nan"),
                "joint_cd_loss": float("nan"),
                "gen_loss": float("nan"),
                "mixed_loss": results["dis_loss"],
                "gen_loss_kind": "not_applicable_alpha_zero",
            }
        )
    clean_score = _score_batches(model, test_x, dev).numpy()
    results["uq_clean_log_px_mean"] = float(clean_score.mean())
    if threshold_split == "test":
        threshold_scores = clean_score
    elif threshold_split == "valid":
        valid_x, _ = _collect(dh.classification["valid"])
        threshold_scores = _score_batches(model, valid_x, dev).numpy()
    else:
        raise ValueError("threshold_split must be 'test' or 'valid'")
    thresholds = {q: float(np.percentile(threshold_scores, q)) for q in percentiles}

    defense_idx = None
    if defense_subsample is not None and defense_subsample < len(test_x):
        generator = torch.Generator().manual_seed(0)
        defense_idx = torch.randperm(len(test_x), generator=generator)[
            :defense_subsample
        ]
        defense_x, defense_y = test_x[defense_idx], test_y[defense_idx]
    else:
        defense_x, defense_y = test_x, test_y

    sampling_idx = None
    if sampling_subsample is not None and sampling_subsample < len(test_x):
        generator = torch.Generator().manual_seed(0)
        sampling_idx = torch.randperm(len(test_x), generator=generator)[
            :sampling_subsample
        ]
        sampling_x, sampling_y = test_x[sampling_idx], test_y[sampling_idx]
    else:
        sampling_x, sampling_y = test_x, test_y
    results["sgld_purify_num_examples"] = float(len(sampling_x))

    # Clean-purification sanity checks, matching the MPS UQ analysis.
    for radius in radii:
        pur_cfg = PurificationConfig(radius=radius, num_steps=20)
        grad_chunks = []
        for i in range(0, len(defense_x), 256):
            batch = defense_x[i : i + 256].to(dev)
            grad_chunks.append(gradient_purify(model, batch, pur_cfg).cpu())
        grad_clean = torch.cat(grad_chunks)
        results[f"uq_clean_purify_acc/{radius}"] = float(
            (_pred_batches(model, grad_clean, dev) == defense_y).float().mean()
        )

    sampling_cfg = PurificationConfig(
        radius=sampling_radius,
        num_steps=sampling_steps_per_sweep,
        step_size=sampling_step_size,
        sgld_noise_std=sampling_noise_std,
    )
    clean_sampling_chunks = {sweep: [] for sweep in sampling_sweeps}
    for i in range(0, len(sampling_x), 256):
        batch = sampling_x[i : i + 256].to(dev)
        snapshots = sgld_purify_snapshots(
            model,
            sampler,
            batch,
            sampling_cfg,
            sampling_sweeps,
        )
        for sweep, snapshot in snapshots.items():
            clean_sampling_chunks[sweep].append(snapshot.cpu())
    for sweep, chunks in clean_sampling_chunks.items():
        purified = torch.cat(chunks)
        results[f"sgld_clean_purify_acc/{sweep}"] = float(
            (_pred_batches(model, purified, dev) == sampling_y).float().mean()
        )

    for eps in attack_eps:
        attack_cfg = PGDConfig(
            epsilon=eps, num_steps=40, random_start=True, restarts=1
        )
        adv = _attack_dataset(
            model, test_x, test_y, pgd_classification, attack_cfg, dev
        )
        adv_pred = _pred_batches(model, adv, dev)
        results[f"rob/{eps}"] = float((adv_pred == test_y).float().mean())
        results[f"uq_adv_acc/{eps}"] = results[f"rob/{eps}"]
        adv_score = _score_batches(model, adv, dev).numpy()
        wrong = (adv_pred != test_y).numpy()
        for q, threshold in thresholds.items():
            detected = adv_score < threshold
            results[f"uq_detection/{q}pct/{eps}"] = float(detected.mean())
            results[f"uq_det_err_detected/{q}pct/{eps}"] = (
                float(wrong[detected].mean()) if detected.any() else float("nan")
            )
            passed = ~detected
            results[f"uq_det_err_passed/{q}pct/{eps}"] = (
                float(wrong[passed].mean()) if passed.any() else float("nan")
            )

        adaptive = _attack_dataset(
            model,
            test_x,
            test_y,
            pgd_likelihood_aware,
            attack_cfg,
            dev,
            score_weight=adaptive_score_weight,
        )
        adaptive_pred = _pred_batches(model, adaptive, dev)
        results[f"adaptive_rob/{eps}"] = float(
            (adaptive_pred == test_y).float().mean()
        )
        results[f"uq_joint_adv_acc/{eps}"] = results[f"adaptive_rob/{eps}"]
        adaptive_score = _score_batches(model, adaptive, dev).numpy()
        adaptive_wrong = (adaptive_pred != test_y).numpy()
        for q, threshold in thresholds.items():
            detected = adaptive_score < threshold
            rate = float(detected.mean())
            results[f"adaptive_detection/{q}pct/{eps}"] = rate
            results[f"uq_joint_detection/{q}pct/{eps}"] = rate
            results[f"uq_joint_det_err_detected/{q}pct/{eps}"] = (
                float(adaptive_wrong[detected].mean())
                if detected.any()
                else float("nan")
            )
            passed = ~detected
            results[f"uq_joint_det_err_passed/{q}pct/{eps}"] = (
                float(adaptive_wrong[passed].mean())
                if passed.any()
                else float("nan")
            )

        # Purification uses fixed paired subsets shared across models and seeds.
        adv_sub = (adv if defense_idx is None else adv[defense_idx]).to(dev)
        adaptive_sub = (
            adaptive if defense_idx is None else adaptive[defense_idx]
        ).to(dev)
        adv_sub_pred = _pred_batches(model, adv_sub.cpu(), dev)
        adaptive_sub_pred = _pred_batches(model, adaptive_sub.cpu(), dev)
        for radius in radii:
            pur_cfg = PurificationConfig(radius=radius, num_steps=20)
            for attack_kind, attacked, before_pred in (
                ("standard", adv_sub, adv_sub_pred),
                ("joint", adaptive_sub, adaptive_sub_pred),
            ):
                grad_chunks = []
                for i in range(0, len(attacked), 256):
                    batch = attacked[i : i + 256]
                    grad_chunks.append(gradient_purify(model, batch, pur_cfg).cpu())
                grad_pur = torch.cat(grad_chunks)
                grad_pred = _pred_batches(model, grad_pur, dev)
                wrong_before = before_pred != defense_y
                denom = int(wrong_before.sum())
                grad_prefix = (
                    "uq_purify"
                    if attack_kind == "standard"
                    else "uq_joint_purify"
                )
                results[f"{grad_prefix}_acc/{eps}/{radius}"] = float(
                    (grad_pred == defense_y).float().mean()
                )
                results[f"{grad_prefix}_recovery/{eps}/{radius}"] = (
                    float((wrong_before & (grad_pred == defense_y)).sum() / denom)
                    if denom
                    else 1.0
                )
                if attack_kind == "joint":
                    # Backward-compatible aliases for early JEM outputs.
                    results[f"adaptive_uq_purify_acc/{eps}/{radius}"] = results[
                        f"{grad_prefix}_acc/{eps}/{radius}"
                    ]

        sampling_adv = (adv if sampling_idx is None else adv[sampling_idx]).to(dev)
        sampling_adaptive = (
            adaptive if sampling_idx is None else adaptive[sampling_idx]
        ).to(dev)
        for attack_kind, attacked in (
            ("standard", sampling_adv),
            ("joint", sampling_adaptive),
        ):
            before_pred = _pred_batches(model, attacked.cpu(), dev)
            wrong_before = before_pred != sampling_y
            denom = int(wrong_before.sum())
            chunks = {sweep: [] for sweep in sampling_sweeps}
            for i in range(0, len(attacked), 256):
                snapshots = sgld_purify_snapshots(
                    model,
                    sampler,
                    attacked[i : i + 256],
                    sampling_cfg,
                    sampling_sweeps,
                )
                for sweep, snapshot in snapshots.items():
                    chunks[sweep].append(snapshot.cpu())
            prefix = (
                "sgld_purify"
                if attack_kind == "standard"
                else "sgld_joint_purify"
            )
            for sweep, sweep_chunks in chunks.items():
                purified = torch.cat(sweep_chunks)
                purified_pred = _pred_batches(model, purified, dev)
                results[f"{prefix}_acc/{eps}/{sweep}"] = float(
                    (purified_pred == sampling_y).float().mean()
                )
                results[f"{prefix}_recovery/{eps}/{sweep}"] = (
                    float(
                        (
                            wrong_before
                            & (purified_pred == sampling_y)
                        ).sum()
                        / denom
                    )
                    if denom
                    else 1.0
                )
                if attack_kind == "joint":
                    results[f"adaptive_sgld_purify_acc/{eps}/{sweep}"] = results[
                        f"{prefix}_acc/{eps}/{sweep}"
                    ]

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--defense-subsample", type=int, default=None)
    parser.add_argument("--sampling-radius", type=float, default=0.2)
    parser.add_argument("--sampling-sweeps", type=int, nargs="+", default=(1, 3, 5))
    parser.add_argument("--sampling-steps-per-sweep", type=int, default=20)
    parser.add_argument("--sampling-step-size", type=float, default=0.01)
    parser.add_argument("--sampling-noise-std", type=float, default=0.005)
    parser.add_argument("--sampling-subsample", type=int, default=1000)
    parser.add_argument("--threshold-split", choices=("test", "valid"), default="test")
    parser.add_argument("--adaptive-score-weight", type=float, default=1.0)
    args = parser.parse_args()
    result = analyze_run(
        args.run_dir,
        device=args.device,
        defense_subsample=args.defense_subsample,
        sampling_radius=args.sampling_radius,
        sampling_sweeps=tuple(args.sampling_sweeps),
        sampling_steps_per_sweep=args.sampling_steps_per_sweep,
        sampling_step_size=args.sampling_step_size,
        sampling_noise_std=args.sampling_noise_std,
        sampling_subsample=args.sampling_subsample,
        threshold_split=args.threshold_split,
        adaptive_score_weight=args.adaptive_score_weight,
    )
    for key, value in sorted(result.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
