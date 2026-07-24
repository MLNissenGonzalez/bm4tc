"""Post-hoc analysis for one JEM run, aligned with the MPS metric schema."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.nn import functional as F

from src.datahandler import DataHandler
from src.utils.paths import data_root

from .attacks import PGDConfig, pgd_classification, pgd_likelihood_aware
from .device import resolve_device
from .model import JEMMLP
from .purification import PurificationConfig, gradient_purify, sgld_purify
from .sampler import ReplayBuffer, SGLDConfig, SGLDSampler
from .trainer import evaluate_classifier

logger = logging.getLogger(__name__)


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


def _msp_batches(model, data: torch.Tensor, device, batch_size=256) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            probs = model.class_probabilities(data[i : i + batch_size].to(device))
            out.append(probs.max(1).values.cpu())
    return torch.cat(out)


def _attack_dataset(model, data, labels, attack_fn, attack_cfg, device, **kwargs):
    chunks = []
    for i in range(0, len(data), 256):
        x = data[i : i + 256].to(device)
        y = labels[i : i + 256].to(device)
        chunks.append(attack_fn(model, x, y, attack_cfg, **kwargs).cpu())
    return torch.cat(chunks)


def _ood_metrics(id_score: np.ndarray, ood_score: np.ndarray) -> dict[str, float]:
    # Positive class is ID: larger JEM score should mean more in-distribution.
    labels = np.concatenate([np.ones(len(id_score)), np.zeros(len(ood_score))])
    scores = np.concatenate([id_score, ood_score])
    auroc = roc_auc_score(labels, scores)
    aupr_in = average_precision_score(labels, scores)
    aupr_out = average_precision_score(1 - labels, -scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    at_95 = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[at_95[0]]) if len(at_95) else 1.0
    return {
        "auroc": float(auroc),
        "aupr_in": float(aupr_in),
        "aupr_out": float(aupr_out),
        "fpr95": fpr95,
    }


def _load_mnist_like_ood(name: str, datahandler, resize: int = 12) -> torch.Tensor:
    """Load a torchvision grayscale OOD test set with the MNIST-r12 transform."""
    import torchvision.datasets as tv_datasets

    root = str(data_root() / ".datasets")
    cls = {"fashion_mnist": tv_datasets.FashionMNIST, "kmnist": tv_datasets.KMNIST}[name]
    dataset = cls(root=root, train=False, download=True)
    raw = dataset.data.float().unsqueeze(1) / 255.0
    resized = F.interpolate(
        raw, size=(resize, resize), mode="bilinear", align_corners=False
    )
    flat = resized.squeeze(1).numpy().reshape(-1, resize * resize).astype(np.float32)
    scaled = datahandler.scaler.transform(flat)
    return torch.from_numpy(np.asarray(scaled, dtype=np.float32))


def analyze_run(
    run_dir: str | Path,
    *,
    device: str = "auto",
    attack_eps: tuple[float, ...] = (0.1, 0.2, 0.3),
    percentiles: tuple[int, ...] = (1, 5, 10, 20),
    radii: tuple[float, ...] = (0.2, 0.3),
    defense_subsample: int | None = None,
    compute_ood: bool = True,
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
    alpha = float(cfg.trainer.get("alpha", 0.0))
    results["alpha"] = alpha
    results["parameters"] = float(model.count_parameters())

    test_x, test_y = _collect(dh.classification["test"])
    # Diagnostic counterpart of the MPS gen_loss. This is a contrastive
    # estimate using the saved replay buffer, not a normalized NLL.
    n_test = len(test_x)
    repeats = (n_test + len(sampler.buffer.data) - 1) // len(sampler.buffer.data)
    negatives = sampler.buffer.data.repeat(repeats, 1)[:n_test]
    with torch.no_grad():
        pos_joint, neg_score = [], []
        for i in range(0, n_test, 256):
            xb = test_x[i : i + 256].to(dev)
            yb = test_y[i : i + 256].to(dev)
            logits = model(xb)
            pos_joint.append(logits.gather(1, yb[:, None]).squeeze(1).cpu())
            neg_score.append(
                model.marginal_score(negatives[i : i + 256].to(dev)).cpu()
            )
    results["gen_loss"] = float(-torch.cat(pos_joint).mean() + torch.cat(neg_score).mean())
    results["mixed_loss"] = (
        (1.0 - alpha) * results["dis_loss"] + alpha * results["gen_loss"]
    )
    results["gen_loss_kind"] = "contrastive_replay_estimate"
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

    if defense_subsample is not None and defense_subsample < len(test_x):
        generator = torch.Generator().manual_seed(0)
        idx = torch.randperm(len(test_x), generator=generator)[:defense_subsample]
        defense_x, defense_y = test_x[idx], test_y[idx]
        defense_is_full = False
    else:
        defense_x, defense_y = test_x, test_y
        defense_is_full = True

    # Clean-purification sanity checks, matching the MPS UQ analysis.
    for radius in radii:
        pur_cfg = PurificationConfig(radius=radius, num_steps=20)
        grad_chunks, sgld_chunks = [], []
        for i in range(0, len(defense_x), 256):
            batch = defense_x[i : i + 256].to(dev)
            grad_chunks.append(gradient_purify(model, batch, pur_cfg).cpu())
            sgld_chunks.append(sgld_purify(model, sampler, batch, pur_cfg).cpu())
        grad_clean = torch.cat(grad_chunks)
        sgld_clean = torch.cat(sgld_chunks)
        results[f"uq_clean_purify_acc/{radius}"] = float(
            (_pred_batches(model, grad_clean, dev) == defense_y).float().mean()
        )
        results[f"sgld_clean_purify_acc/{radius}"] = float(
            (_pred_batches(model, sgld_clean, dev) == defense_y).float().mean()
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
        adaptive_score = _score_batches(model, adaptive, dev).numpy()
        for q, threshold in thresholds.items():
            results[f"adaptive_detection/{q}pct/{eps}"] = float(
                (adaptive_score < threshold).mean()
            )

        # Purification is intentionally estimated on one fixed paired subset.
        if defense_is_full:
            adv_sub = adv.to(dev)
            adaptive_sub = adaptive.to(dev)
        else:
            sub_cfg = PGDConfig(epsilon=eps, num_steps=40, random_start=True)
            adv_sub = _attack_dataset(
                model, defense_x, defense_y, pgd_classification, sub_cfg, dev
            ).to(dev)
            adaptive_sub = _attack_dataset(
                model,
                defense_x,
                defense_y,
                pgd_likelihood_aware,
                sub_cfg,
                dev,
                score_weight=adaptive_score_weight,
            ).to(dev)
        adv_sub_pred = _pred_batches(model, adv_sub.cpu(), dev)
        adaptive_sub_pred = _pred_batches(model, adaptive_sub.cpu(), dev)
        for radius in radii:
            pur_cfg = PurificationConfig(radius=radius, num_steps=20)
            for prefix, attacked, before_pred in (
                ("", adv_sub, adv_sub_pred),
                ("adaptive_", adaptive_sub, adaptive_sub_pred),
            ):
                grad_chunks, sgld_chunks = [], []
                for i in range(0, len(attacked), 256):
                    batch = attacked[i : i + 256]
                    grad_chunks.append(gradient_purify(model, batch, pur_cfg).cpu())
                    sgld_chunks.append(
                        sgld_purify(model, sampler, batch, pur_cfg).cpu()
                    )
                grad_pur = torch.cat(grad_chunks)
                sgld_pur = torch.cat(sgld_chunks)
                grad_pred = _pred_batches(model, grad_pur, dev)
                sgld_pred = _pred_batches(model, sgld_pur, dev)
                wrong_before = before_pred != defense_y
                denom = int(wrong_before.sum())
                results[f"{prefix}uq_purify_acc/{eps}/{radius}"] = float(
                    (grad_pred == defense_y).float().mean()
                )
                results[f"{prefix}sgld_purify_acc/{eps}/{radius}"] = float(
                    (sgld_pred == defense_y).float().mean()
                )
                results[f"{prefix}uq_purify_recovery/{eps}/{radius}"] = (
                    float((wrong_before & (grad_pred == defense_y)).sum() / denom)
                    if denom
                    else 1.0
                )
                results[f"{prefix}sgld_purify_recovery/{eps}/{radius}"] = (
                    float((wrong_before & (sgld_pred == defense_y)).sum() / denom)
                    if denom
                    else 1.0
                )

    if compute_ood:
        clean_msp = _msp_batches(model, test_x, dev).numpy()
        for name in ("fashion_mnist", "kmnist"):
            try:
                ood = _load_mnist_like_ood(name, dh)
                metrics = _ood_metrics(
                    clean_score, _score_batches(model, ood, dev).numpy()
                )
                for metric, value in metrics.items():
                    results[f"ood/{name}/{metric}"] = value
                msp_metrics = _ood_metrics(
                    clean_msp, _msp_batches(model, ood, dev).numpy()
                )
                for metric, value in msp_metrics.items():
                    results[f"ood_msp/{name}/{metric}"] = value
            except Exception as exc:
                logger.warning("Skipping OOD dataset %s: %s", name, exc)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--defense-subsample", type=int, default=None)
    parser.add_argument("--threshold-split", choices=("test", "valid"), default="test")
    parser.add_argument("--adaptive-score-weight", type=float, default=1.0)
    parser.add_argument("--no-ood", action="store_true")
    args = parser.parse_args()
    result = analyze_run(
        args.run_dir,
        device=args.device,
        defense_subsample=args.defense_subsample,
        compute_ood=not args.no_ood,
        threshold_split=args.threshold_split,
        adaptive_score_weight=args.adaptive_score_weight,
    )
    for key, value in sorted(result.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
