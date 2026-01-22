import src.utils.schemas as schemas
import torch
from typing import *
import logging
from src.models.born import BornMachine
from src.data.handler import DataHandler
import src.utils.get as get

logger = logging.getLogger(__name__)

# TODO: Decide where to prepare classifier and generator views. Maybe do this in the trainer.
class BaseMetric:
    """
    Abstract base class for evaluation metrics.

    Provides shared infrastructure for computing metrics on BornMachine,
    including caching of predictions and synthetic samples in context dict.

    Attributes:
        freq: How often (in epochs) this metric should be evaluated.
    """

    def __init__(
            self,
            freq: int,
            cfg: schemas.Config,
            datahandler: DataHandler,
            device: torch.device
    ):
        self.freq = freq
        self.cfg = cfg
        self.datahandler = datahandler
        self.device = device
    
    def _labels_n_probs(
            self, 
            bornmachine: BornMachine,
            split: str,
            context: Dict[str, Any]
            ):
        """
        Method to precompute class probabilities if not already in context.
        """
        
        if "labels_n_probs" not in context:
            context["labels_n_probs"] = []
            with torch.no_grad():
                for data, labels in self.datahandler.classification[split]:
                    data = data.to(self.device)
                    probs = bornmachine.class_probabilities(data).cpu()
                    context["labels_n_probs"].append((labels, probs))


    def _generate(
            self,
            bornmachine: BornMachine,
            context: Dict[str, Any]
    ):
        """
        Method to generate from BornMachine if no synthetic examples in the context. For evaluation only. 
        """

        if "synths" not in context:
            with torch.no_grad():
                context["synths"] = bornmachine.sample(cfg=self.cfg.tracking.sampling)

    def evaluate(self, 
                 bornmachine: BornMachine, 
                 split: str, 
                 context: Dict[str, Any]):
        raise NotImplementedError


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#------Classification Loss and Accuracy-----------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# I want here the classification loss taking probabilities (batch_size, num_classes) and labels (batch_size,).

# TODO: cfg.trainer.classification might not be provided in a pure GAN-syle training or pure generative training. 
class ClassificationLossMetric(BaseMetric):
    """Compute classification loss (NLL) on the dataset."""

    def __init__(self, freq, cfg: schemas.Config, datahandler, device):
        super().__init__(freq, cfg, datahandler, device)
        if hasattr(cfg.trainer, "classification") and hasattr(cfg.trainer.classification, "criterion"):
            self.criterion = get.criterion(mode="classification", cfg=cfg.trainer.classification.criterion)
        else:
            crit_cfg = schemas.CriterionConfig(name="negative log-likelihood", kwargs={"eps": 1e-8})
            self.criterion = get.criterion(mode="classification", cfg=crit_cfg)
    def evaluate(self, bornmachine, split, context):
        bornmachine.classifier.eval()
        losses = []
        self._labels_n_probs(bornmachine, split, context)

        with torch.no_grad():
            for labels, probs in context["labels_n_probs"]:
                loss = self.criterion(probs, labels).mean().item()
                losses.append(loss)

        # Compute mean loss
        return sum(losses) / len(losses) if losses else float('nan')

class AccuracyMetric(BaseMetric):
    """Compute clean classification accuracy on the dataset."""

    def __init__(self, freq, cfg, datahandler, device):
        super().__init__(freq, cfg, datahandler, device)

    def evaluate(self, bornmachine: BornMachine, split, context):
        bornmachine.classifier.eval()
        correct, num_pred = 0, 0
        self._labels_n_probs(bornmachine, split, context)

        with torch.no_grad():
            for labels, probs in context["labels_n_probs"]:        
                predictions = torch.argmax(probs, dim=1) # (batch_size,)
                correct += (predictions == labels).sum().item()
                num_pred += labels.shape[0]
        
        return correct / num_pred if num_pred > 0 else float('nan')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------FID-like metric---------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
from .fid_like import FIDEvaluation


class FIDMetric(BaseMetric):
    """Compute FID-like score comparing synthetic samples to real data."""

    def __init__(self, freq, cfg, datahandler, device):
        super().__init__(freq, cfg, datahandler, device)
        self.fid = FIDEvaluation(cfg, datahandler, device)

    def evaluate(self, bornmachine: BornMachine, split, context):        
        self._generate(bornmachine, context)
        try:
            fid_score = self.fid.evaluate(context["synths"])
            return fid_score
        except ValueError as e:
            logger.warning(str(e))
            return float('nan')

    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------Visualisation------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
from .visualisation import visualise_samples


class VisualizationMetric(BaseMetric):
    """Generate visualization of synthetic samples for W&B logging."""

    def __init__(self, freq, cfg, datahandler, device):
        super().__init__(freq, cfg, datahandler, device)

    def evaluate(self, bornmachine: BornMachine, split, context):
        self._generate(bornmachine, context)
        ax = visualise_samples(context["synths"])
        return ax

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------Robustness---------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

from src.utils.evasion.minimal import RobustnessEvaluation


class RobustnessMetric(BaseMetric):
    """Compute adversarial robustness (accuracy under attack). Supports FGM and PGD."""

    def __init__(self, freq, cfg: schemas.Config, datahandler, device):
        super().__init__(freq, cfg, datahandler, device)
        # TODO: num_steps, step_size, random_start might not exist in cfg.tracking.evasion, 
        #       but there are default values in the RobustnessEvaluation.
        #       Use those defaults if not in struct/cfg.
        num_steps = cfg.tracking.evasion.num_steps if hasattr(cfg.tracking.evasion, 'num_steps') else None
        step_size = cfg.tracking.evasion.step_size if hasattr(cfg.tracking.evasion, 'step_size') else None
        random_start = cfg.tracking.evasion.random_start if hasattr(cfg.tracking.evasion, 'random_start') else None 
        self.evasion = RobustnessEvaluation(method=cfg.tracking.evasion.method,
                                            norm=cfg.tracking.evasion.norm,
                                            criterion=cfg.tracking.evasion.criterion,
                                            strengths=cfg.tracking.evasion.strengths,
                                            num_steps=num_steps,
                                            step_size=step_size,
                                            random_start=random_start)
        

    def evaluate(self, bornmachine: BornMachine, split, context):
        """
        Returns list of post attack accuracies, one for each strength provided.
        """
        robust_acc = self.evasion.evaluate(bornmachine, self.datahandler.classification[split], self.device)
        return robust_acc
        

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------NLL of full joint for GenerativeLossMetric----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

class GenerativeLossMetric(BaseMetric):
    """Compute generative loss (NLL of joint distribution) on the dataset."""

    def __init__(self, freq, cfg: schemas.Config, datahandler, device):
        super().__init__(freq, cfg, datahandler, device)
        if hasattr(cfg.trainer, "generative") and hasattr(cfg.trainer.generative, "criterion"):
            self.criterion = get.criterion(mode="generative", cfg=cfg.trainer.generative.criterion)
        else:
            crit_cfg = schemas.CriterionConfig(name="negative log-likelihood", kwargs={"eps": 1e-8})
            self.criterion = get.criterion(mode="generative", cfg=crit_cfg)

    def evaluate(self, bornmachine, split, context):
        """Compute mean generative NLL over the dataset."""
        losses = []
        with torch.no_grad():
            for data, labels in self.datahandler.classification[split]:
                data, labels = data.to(self.device), labels.to(self.device)
                loss = self.criterion(bornmachine, data, labels).item()
                losses.append(loss)
        return sum(losses) / len(losses) if losses else float('nan')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------Collecting metrics in a single class----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: Consider where to put bornmachine preparation
class MetricFactory:
    """Factory for creating metric instances by name."""

    @staticmethod
    def create(
            metric_name: str,
            freq: int,
            cfg: schemas.Config,
            datahandler: DataHandler,
            device: torch.device
    ) -> BaseMetric:
        """Create a metric instance by name (clsloss, genloss, acc, fid, rob, viz)."""
        mapping = {
            "clsloss": ClassificationLossMetric,
            "acc": AccuracyMetric,
            "fid": FIDMetric,
            "rob": RobustnessMetric,
            "viz": VisualizationMetric,
            "genloss": GenerativeLossMetric,
        }
        return mapping[metric_name](freq, cfg, datahandler, device)


class PerformanceEvaluator:
    """
    Unified evaluator that computes multiple metrics on a BornMachine.

    Metrics are configured via train_cfg.metrics dict mapping metric names
    to evaluation frequencies. Results are returned as a dict.
    """

    def __init__(
            self,
            cfg: schemas.Config,
            datahandler: DataHandler,
            train_cfg: schemas.ClassificationConfig | schemas.GANStyleConfig | schemas.AdversarialConfig,
            device: torch.device
    ):
        if getattr(datahandler, "classification") is None:
            datahandler.get_classification_loaders()

        self.metrics = {}
        self.update_freq = 1
        # Only metrics in the config get initialized.
        for metric_name, freq in train_cfg.metrics.items():
            metric : BaseMetric = MetricFactory.create(
                metric_name=metric_name, freq=freq,
                cfg=cfg, datahandler=datahandler, device=device
            )
            self.metrics[metric_name] = metric
            stop_crit = getattr(train_cfg, 'stop_crit', None)
            if stop_crit and metric_name == stop_crit: 
                metric.freq = 1
            self.update_freq = max(self.update_freq, freq)
            

    def should_evaluate(self, step: int, metric_name: str):
        freq = self.metrics[metric_name].freq
        return step % freq == 0

    def evaluate(self, 
                 bornmachine: BornMachine, 
                 split: str,
                 step: int):
        """
        Evaluates the bornmachine on a collection of metrics that are given in the tracking config of the experiment.
        Returns a dictionary of those metrics with keys in ['clsloss', 'genloss', 'acc', 'fid', 'rob', 'viz'].
        Metric carry `freq` attribute how often per `evaluate` call they are evaluated.
        """
        results, context = {}, {}
        
        for name, metric in self.metrics.items():
            # Only evaluated every freq step on validation.
            toEval = ((self.should_evaluate(step, name) and (split == "valid")) or (split == "test"))
            if toEval:
                if name == "rob":
                    rob_results = metric.evaluate(bornmachine, split, context)
                    for i, strength in enumerate(metric.evasion.strengths):
                        results[f"{name}/{strength}"] = rob_results[i]
                else:
                    results[name] = metric.evaluate(bornmachine, split, context)

        if step % self.update_freq == 0:
            for name, result in results.items():
                if result is None or name == "viz":
                    continue
                elif isinstance(result, (float, int)):
                    logger.info(f"Epoch {step+1} evaluation: {split} - {name}: {result:.3f}")
                else:
                    logger.warning(f"Skipping logging for metric '{name}' - unsupported type: {type(result)}")
        return results