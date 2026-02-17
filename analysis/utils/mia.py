"""
Membership Inference Attack (MIA) evaluation for BornClassifier models.

This module provides tools to evaluate the privacy leakage of trained models
by attempting to distinguish training samples from test samples based on
model outputs (class probabilities).

The attack uses features derived from p(c|x) to train a binary classifier
that predicts whether a sample was in the training set.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
import logging

from src.utils.evasion.minimal import ProjectedGradientDescent
from src.utils.schemas import CriterionConfig

logger = logging.getLogger(__name__)


@dataclass
class MIAFeatureConfig:
    """Configuration for which features to extract from model outputs.

    Each feature captures a different aspect of model confidence that may
    reveal membership information (whether the sample was in training set).

    Attributes:
        max_prob: Use maximum class probability (models more confident on training data).
        entropy: Use prediction entropy (lower = more confident).
        correct_prob: Use probability assigned to the correct label.
        loss: Use negative log-likelihood loss (lower on training data).
        margin: Use difference between top and second probability.
        modified_entropy: Use normalized confidence (1 - entropy / log(num_classes)).
        use_true_labels: If True (default), use ground-truth labels for correct_prob
            and loss features (worst-case risk estimate). If False, use predicted
            labels (argmax of probs) to avoid label leakage.
    """
    max_prob: bool = True
    entropy: bool = True
    correct_prob: bool = True
    loss: bool = True
    margin: bool = True
    modified_entropy: bool = True
    use_true_labels: bool = True

    def enabled_features(self) -> List[str]:
        """Return list of enabled feature names."""
        features = []
        if self.max_prob:
            features.append("max_prob")
        if self.entropy:
            features.append("entropy")
        if self.correct_prob:
            features.append("correct_prob")
        if self.loss:
            features.append("loss")
        if self.margin:
            features.append("margin")
        if self.modified_entropy:
            features.append("modified_entropy")
        return features


class MIAFeatureExtractor:
    """Extract membership inference features from class probabilities.

    Features are computed from p(c|x) outputs and the true labels to capture
    different aspects of model confidence that may reveal membership.
    """

    def __init__(self, config: MIAFeatureConfig):
        """Initialize with feature configuration.

        Args:
            config: Specifies which features to extract.
        """
        self.config = config

    def extract(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract MIA features from class probabilities.

        Args:
            probs: Class probabilities of shape (batch_size, num_classes).
            labels: True labels of shape (batch_size,). Used for correct_prob
                and loss features when use_true_labels=True; otherwise predicted
                labels (argmax of probs) are used instead.

        Returns:
            Dictionary mapping feature names to tensors of shape (batch_size,).
        """
        features = {}
        eps = 1e-10  # For numerical stability
        num_classes = probs.shape[-1]

        # Choose labels for correct_prob and loss features
        if self.config.use_true_labels:
            reference_labels = labels
        else:
            reference_labels = probs.argmax(dim=-1)

        # Clamp probabilities to avoid log(0)
        probs_clamped = probs.clamp(min=eps, max=1.0 - eps)

        if self.config.max_prob:
            features["max_prob"] = probs.max(dim=-1)[0]

        if self.config.entropy:
            entropy = -(probs_clamped * torch.log(probs_clamped)).sum(dim=-1)
            features["entropy"] = entropy

        if self.config.correct_prob:
            batch_indices = torch.arange(probs.shape[0], device=probs.device)
            features["correct_prob"] = probs[batch_indices, reference_labels]

        if self.config.loss:
            batch_indices = torch.arange(probs.shape[0], device=probs.device)
            reference_probs = probs_clamped[batch_indices, reference_labels]
            features["loss"] = -torch.log(reference_probs)

        if self.config.margin:
            sorted_probs, _ = probs.sort(dim=-1, descending=True)
            features["margin"] = sorted_probs[:, 0] - sorted_probs[:, 1]

        if self.config.modified_entropy:
            entropy = -(probs_clamped * torch.log(probs_clamped)).sum(dim=-1)
            max_entropy = np.log(num_classes)
            features["modified_entropy"] = 1.0 - entropy / max_entropy

        return features

    def extract_batch(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor
    ) -> np.ndarray:
        """Extract features and concatenate into a single array.

        Args:
            probs: Class probabilities of shape (batch_size, num_classes).
            labels: True labels of shape (batch_size,).

        Returns:
            Feature array of shape (batch_size, num_features).
        """
        features = self.extract(probs, labels)
        enabled = self.config.enabled_features()

        # Stack features in consistent order
        feature_list = [features[name].cpu().numpy().reshape(-1, 1) for name in enabled]
        return np.hstack(feature_list)


@dataclass
class MIAResults:
    """Results from membership inference attack evaluation.

    Attributes:
        attack_accuracy: Classification accuracy of the attack model.
        auc_roc: Area under ROC curve (0.5 = random, 1.0 = perfect attack).
        precision_at_low_fpr: Precision at various false positive rates.
        feature_importance: Importance score for each feature (logistic regression coefficients).
        threshold_metrics: Per-feature threshold attack results.
        worst_case_threshold: Worst-case (oracle) threshold attack results per feature.
            Computed on the full dataset with the threshold tuned to maximize accuracy.
            This is a theoretical upper bound — a real attacker cannot select the
            optimal threshold without access to ground-truth membership labels.
        train_features: Feature array for training samples.
        test_features: Feature array for test samples.
        feature_names: Names of features used.
    """
    attack_accuracy: float
    auc_roc: float
    precision_at_low_fpr: Dict[float, float]
    feature_importance: Dict[str, float]
    threshold_metrics: Dict[str, Dict[str, float]]
    worst_case_threshold: Dict[str, Dict[str, float]]
    train_features: np.ndarray
    test_features: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    adversarial_worst_case_threshold: Optional[Dict[str, Dict[str, float]]] = None
    adversarial_strength: Optional[float] = None

    def privacy_assessment(self) -> str:
        """Return a privacy assessment based on AUC-ROC.

        Returns:
            String describing the privacy level based on attack success.
        """
        if self.auc_roc < 0.55:
            return "Excellent privacy preservation"
        elif self.auc_roc < 0.60:
            return "Good privacy"
        elif self.auc_roc < 0.70:
            return "Moderate leakage"
        else:
            return "Significant leakage"

    def summary(self) -> str:
        """Return a formatted summary of the MIA results."""
        lines = [
            "=" * 60,
            "Membership Inference Attack Results",
            "=" * 60,
            f"Attack Accuracy: {self.attack_accuracy:.4f}",
            f"AUC-ROC: {self.auc_roc:.4f}",
            f"Privacy Assessment: {self.privacy_assessment()}",
            "",
            "Precision at Low FPR:",
        ]
        for fpr, precision in sorted(self.precision_at_low_fpr.items()):
            lines.append(f"  FPR={fpr:.2%}: Precision={precision:.4f}")

        lines.extend(["", "Feature Importance (|coefficient|):"])
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for name, importance in sorted_importance:
            lines.append(f"  {name}: {importance:.4f}")

        lines.extend(["", "Per-Feature Threshold Attacks (AUC-ROC):"])
        sorted_thresh = sorted(
            self.threshold_metrics.items(),
            key=lambda x: x[1].get("auc_roc", 0),
            reverse=True
        )
        for name, metrics in sorted_thresh:
            lines.append(f"  {name}: {metrics.get('auc_roc', 0):.4f}")

        lines.extend([
            "",
            "Worst-Case Threshold Attack (oracle, full dataset):",
            "  (Threshold tuned to maximize accuracy — theoretical upper bound)",
        ])
        sorted_wc = sorted(
            self.worst_case_threshold.items(),
            key=lambda x: x[1].get("accuracy", 0),
            reverse=True
        )
        for name, metrics in sorted_wc:
            lines.append(
                f"  {name}: acc={metrics['accuracy']:.4f}  "
                f"threshold={metrics['threshold']:.6f}  "
                f"TPR={metrics['tpr']:.4f}  FPR={metrics['fpr']:.4f}"
            )

        if self.adversarial_worst_case_threshold is not None:
            lines.extend([
                "",
                f"Adversarial Worst-Case Threshold Attack (eps={self.adversarial_strength}):",
                "  (Features extracted from model outputs on PGD adversarial examples)",
            ])
            sorted_adv = sorted(
                self.adversarial_worst_case_threshold.items(),
                key=lambda x: x[1].get("accuracy", 0),
                reverse=True
            )
            for name, metrics in sorted_adv:
                lines.append(
                    f"  {name}: acc={metrics['accuracy']:.4f}  "
                    f"threshold={metrics['threshold']:.6f}  "
                    f"TPR={metrics['tpr']:.4f}  FPR={metrics['fpr']:.4f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


class MIAEvaluation:
    """Main class for running membership inference attack evaluation.

    This class computes MIA features from model outputs and trains an attack
    classifier to distinguish training samples from test samples.

    Example:
        >>> mia_eval = MIAEvaluation()
        >>> results = mia_eval.evaluate(bornmachine, train_loader, test_loader, device)
        >>> print(results.summary())
    """

    def __init__(
        self,
        feature_config: Optional[MIAFeatureConfig] = None,
        attack_model: str = "logistic",
        test_split: float = 0.3,
        random_state: int = 42,
        adversarial_strength: Optional[float] = None,
        adversarial_num_steps: int = 20,
        adversarial_step_size: Optional[float] = None,
        adversarial_norm: Union[str, int] = "inf",
    ):
        """Initialize MIA evaluation.

        Args:
            feature_config: Configuration for feature extraction. Uses defaults if None.
            attack_model: Type of attack classifier ("logistic" supported).
            test_split: Fraction of data to use for attack model evaluation.
            random_state: Random seed for reproducibility.
            adversarial_strength: Epsilon for PGD attack. None = skip adversarial MIA.
            adversarial_num_steps: Number of PGD steps.
            adversarial_step_size: PGD step size. None = auto (2.5 * eps / steps).
            adversarial_norm: Lp norm for PGD perturbation ball ("inf" or int >= 1).
        """
        self.feature_config = feature_config or MIAFeatureConfig()
        self.attack_model = attack_model
        self.test_split = test_split
        self.random_state = random_state
        self.extractor = MIAFeatureExtractor(self.feature_config)
        self.adversarial_strength = adversarial_strength
        self.adversarial_num_steps = adversarial_num_steps
        self.adversarial_step_size = adversarial_step_size
        self.adversarial_norm = adversarial_norm

    def _extract_all_features(
        self,
        model,
        data_loader: DataLoader,
        device: torch.device,
    ) -> np.ndarray:
        """Extract MIA features for all samples in a data loader.

        Args:
            model: BornMachine or model with class_probabilities method.
            data_loader: DataLoader yielding (data, labels) tuples.
            device: Device to run inference on.

        Returns:
            Feature array of shape (num_samples, num_features).
        """
        all_features = []

        model.to(device)

        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                # Get class probabilities
                probs = model.class_probabilities(batch_data)

                # Extract features
                features = self.extractor.extract_batch(probs, batch_labels)
                all_features.append(features)

        return np.vstack(all_features)

    def _extract_all_features_adversarial(
        self,
        model,
        data_loader: DataLoader,
        device: torch.device,
        pgd: ProjectedGradientDescent,
        strength: float,
    ) -> np.ndarray:
        """Extract MIA features from model outputs on adversarial examples.

        For each batch, generates untargeted PGD adversarial examples and
        extracts the same confidence features from the model's output on
        the perturbed inputs.

        Args:
            model: BornMachine or model with class_probabilities method.
            data_loader: DataLoader yielding (data, labels) tuples.
            device: Device to run inference on.
            pgd: ProjectedGradientDescent instance for generating adversarial examples.
            strength: Epsilon (attack radius) for PGD.

        Returns:
            Feature array of shape (num_samples, num_features).
        """
        all_features = []

        model.to(device)

        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # Generate adversarial examples (requires gradients)
            adv_examples = pgd.generate(
                model, batch_data, batch_labels, strength, device
            )

            # Extract features from model output on adversarial examples
            with torch.no_grad():
                probs = model.class_probabilities(adv_examples)
                features = self.extractor.extract_batch(probs, batch_labels)
                all_features.append(features)

        return np.vstack(all_features)

    def _compute_threshold_metrics(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-feature threshold attack metrics.

        For each feature, compute how well a simple threshold on that single
        feature can distinguish train from test samples.

        Args:
            train_features: Features for training samples.
            test_features: Features for test samples.
            feature_names: Names of the features.

        Returns:
            Dictionary mapping feature names to their threshold attack metrics.
        """
        results = {}

        # Labels: 1 for train (members), 0 for test (non-members)
        labels = np.concatenate([
            np.ones(len(train_features)),
            np.zeros(len(test_features))
        ])

        for i, name in enumerate(feature_names):
            feature_values = np.concatenate([
                train_features[:, i],
                test_features[:, i]
            ])

            # For loss and entropy, lower means more likely to be in training
            # For others, higher means more likely to be in training
            if name in ["loss", "entropy"]:
                scores = -feature_values  # Flip so higher = more likely training
            else:
                scores = feature_values

            try:
                auc = roc_auc_score(labels, scores)
            except ValueError:
                auc = 0.5  # If only one class present

            results[name] = {
                "auc_roc": auc,
                "train_mean": float(train_features[:, i].mean()),
                "train_std": float(train_features[:, i].std()),
                "test_mean": float(test_features[:, i].mean()),
                "test_std": float(test_features[:, i].std()),
            }

        return results

    def _compute_precision_at_fpr(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        target_fprs: List[float] = [0.01, 0.05, 0.10],
    ) -> Dict[float, float]:
        """Compute precision at specified false positive rates.

        Args:
            y_true: True binary labels.
            y_scores: Predicted scores/probabilities.
            target_fprs: Target false positive rates.

        Returns:
            Dictionary mapping FPR to precision at that FPR.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        results = {}
        n_neg = (y_true == 0).sum()
        n_pos = (y_true == 1).sum()

        for target_fpr in target_fprs:
            # Find threshold that gives approximately target FPR
            # FPR = FP / (FP + TN) = FP / N_neg
            # At each threshold, count how many negatives exceed it
            best_precision = 0.0
            for thresh in thresholds:
                fp = ((y_scores >= thresh) & (y_true == 0)).sum()
                tp = ((y_scores >= thresh) & (y_true == 1)).sum()
                fpr = fp / n_neg if n_neg > 0 else 0
                if fpr <= target_fpr + 0.01:  # Allow small tolerance
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    best_precision = max(best_precision, prec)

            results[target_fpr] = best_precision

        return results

    def _compute_worst_case_threshold(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute worst-case (oracle) threshold attack on the full dataset.

        For each feature, sweep all unique thresholds and pick the one that
        maximizes membership inference accuracy. This is a theoretical upper
        bound: a real adversary cannot select the optimal threshold without
        knowing ground-truth membership labels.

        The attack rule for each threshold t is:
            predict "member" if score >= t, "non-member" otherwise.
        For features where lower values indicate membership (loss, entropy),
        the sign is flipped before thresholding.

        Args:
            train_features: Features for training (member) samples, shape (N_train, F).
            test_features: Features for test (non-member) samples, shape (N_test, F).
            feature_names: Names of the features (length F).

        Returns:
            Dictionary mapping feature name to metrics dict with keys:
                accuracy, threshold, tpr, fpr, n_train, n_test.
        """
        n_train = len(train_features)
        n_test = len(test_features)
        n_total = n_train + n_test

        # Labels: 1 for train (members), 0 for test (non-members)
        labels = np.concatenate([np.ones(n_train), np.zeros(n_test)])

        results = {}

        for i, name in enumerate(feature_names):
            feature_values = np.concatenate([
                train_features[:, i],
                test_features[:, i]
            ])

            # For loss and entropy, lower means more likely member — flip sign
            if name in ["loss", "entropy"]:
                scores = -feature_values
            else:
                scores = feature_values

            # Sort unique thresholds (midpoints between consecutive unique values)
            sorted_unique = np.unique(scores)
            if len(sorted_unique) <= 1:
                results[name] = {
                    "accuracy": n_train / n_total,
                    "threshold": 0.0,
                    "tpr": 1.0,
                    "fpr": 1.0,
                    "n_train": n_train,
                    "n_test": n_test,
                }
                continue

            midpoints = (sorted_unique[:-1] + sorted_unique[1:]) / 2.0

            # Vectorised sweep: for each threshold, predict member if score >= t
            # preds[j, k] = 1 if scores[k] >= midpoints[j]
            # Use broadcasting: (T, 1) >= (1, N) -> (T, N)
            preds = (midpoints[:, None] <= scores[None, :]).astype(np.float64)

            # Accuracy for each threshold
            correct = (preds == labels[None, :]).sum(axis=1)
            accuracies = correct / n_total

            best_idx = np.argmax(accuracies)
            best_threshold = midpoints[best_idx]
            best_preds = preds[best_idx]

            tp = ((best_preds == 1) & (labels == 1)).sum()
            fp = ((best_preds == 1) & (labels == 0)).sum()
            tpr = tp / n_train if n_train > 0 else 0.0
            fpr = fp / n_test if n_test > 0 else 0.0

            # Map threshold back to original scale for interpretability
            if name in ["loss", "entropy"]:
                original_threshold = -best_threshold
            else:
                original_threshold = best_threshold

            results[name] = {
                "accuracy": float(accuracies[best_idx]),
                "threshold": float(original_threshold),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "n_train": n_train,
                "n_test": n_test,
            }

            logger.info(
                f"Worst-case threshold [{name}]: "
                f"acc={accuracies[best_idx]:.4f}, "
                f"threshold={original_threshold:.6f}, "
                f"TPR={tpr:.4f}, FPR={fpr:.4f}"
            )

        return results

    def evaluate(
        self,
        model,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ) -> MIAResults:
        """Run the membership inference attack evaluation.

        Args:
            model: BornMachine or model with class_probabilities method.
            train_loader: DataLoader for training data (members).
            test_loader: DataLoader for test data (non-members).
            device: Device to run inference on.

        Returns:
            MIAResults containing attack metrics and extracted features.
        """
        logger.info("Extracting features from training samples...")
        train_features = self._extract_all_features(model, train_loader, device)

        logger.info("Extracting features from test samples...")
        test_features = self._extract_all_features(model, test_loader, device)

        logger.info(f"Train features shape: {train_features.shape}")
        logger.info(f"Test features shape: {test_features.shape}")

        # Create binary classification problem
        # Label 1 = training sample (member), 0 = test sample (non-member)
        X = np.vstack([train_features, test_features])
        y = np.concatenate([
            np.ones(len(train_features)),
            np.zeros(len(test_features))
        ])

        # Split for attack model training and evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y,
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=y
        )

        # Train attack model
        logger.info("Training attack classifier...")
        if self.attack_model == "logistic":
            attack_clf = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight="balanced"
            )
        else:
            raise ValueError(f"Unknown attack model: {self.attack_model}")

        attack_clf.fit(X_train, y_train)

        # Evaluate attack
        y_pred = attack_clf.predict(X_eval)
        y_proba = attack_clf.predict_proba(X_eval)[:, 1]

        attack_accuracy = accuracy_score(y_eval, y_pred)
        auc_roc = roc_auc_score(y_eval, y_proba)

        logger.info(f"Attack accuracy: {attack_accuracy:.4f}")
        logger.info(f"Attack AUC-ROC: {auc_roc:.4f}")

        # Compute feature importance from logistic regression coefficients
        feature_names = self.feature_config.enabled_features()
        feature_importance = {
            name: float(coef)
            for name, coef in zip(feature_names, attack_clf.coef_[0])
        }

        # Compute per-feature threshold attacks
        threshold_metrics = self._compute_threshold_metrics(
            train_features, test_features, feature_names
        )

        # Compute worst-case (oracle) threshold attack on full dataset
        logger.info("Computing worst-case threshold attack (oracle)...")
        worst_case_threshold = self._compute_worst_case_threshold(
            train_features, test_features, feature_names
        )

        # Compute precision at low FPR
        precision_at_low_fpr = self._compute_precision_at_fpr(y_eval, y_proba)

        # Adversarial MIA pipeline
        adversarial_worst_case_threshold = None
        if self.adversarial_strength is not None:
            logger.info(
                f"Running adversarial MIA (eps={self.adversarial_strength}, "
                f"steps={self.adversarial_num_steps}, norm={self.adversarial_norm})..."
            )
            pgd = ProjectedGradientDescent(
                norm=self.adversarial_norm,
                criterion=CriterionConfig(name="nll", kwargs=None),
                num_steps=self.adversarial_num_steps,
                step_size=self.adversarial_step_size,
                random_start=True,
            )

            logger.info("Extracting adversarial features from training samples...")
            adv_train_features = self._extract_all_features_adversarial(
                model, train_loader, device, pgd, self.adversarial_strength
            )

            logger.info("Extracting adversarial features from test samples...")
            adv_test_features = self._extract_all_features_adversarial(
                model, test_loader, device, pgd, self.adversarial_strength
            )

            logger.info(f"Adversarial train features shape: {adv_train_features.shape}")
            logger.info(f"Adversarial test features shape: {adv_test_features.shape}")

            logger.info("Computing adversarial worst-case threshold attack (oracle)...")
            adversarial_worst_case_threshold = self._compute_worst_case_threshold(
                adv_train_features, adv_test_features, feature_names
            )

        return MIAResults(
            attack_accuracy=attack_accuracy,
            auc_roc=auc_roc,
            precision_at_low_fpr=precision_at_low_fpr,
            feature_importance=feature_importance,
            threshold_metrics=threshold_metrics,
            worst_case_threshold=worst_case_threshold,
            train_features=train_features,
            test_features=test_features,
            feature_names=feature_names,
            adversarial_worst_case_threshold=adversarial_worst_case_threshold,
            adversarial_strength=self.adversarial_strength,
        )
