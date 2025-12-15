import src.utils.schemas as schemas
import torch
from typing import *
import logging
from src.models.born import BornMachine
from src.data.handler import DataHandler
import src.utils.getters as get

logger = logging.getLogger(__name__)

# TODO: Decide where to prepare classifier and generator views. Maybe do this in the trainer.
class BaseMetric:
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
#------Loss and Accuracy-----------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# I want here the classification loss taking probabilities (batch_size, num_classes) and labels (batch_size,).


# classifier based 
class LossMetric(BaseMetric):
    def __init__(self, freq, cfg: schemas.Config, datahandler, device):
        super().__init__(freq, cfg, datahandler, device)
        self.criterion = get.criterion(mode="classification", cfg=cfg.trainer.classification.criterion)

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

# clean accuracy
class AccuracyMetric(BaseMetric):
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
# generator based
from fid_like import FIDEvaluation  

class FIDMetric(BaseMetric):
    def __init__(self, freq, cfg, datahandler, device):
        super().__init__(freq, cfg, datahandler, device)
        self.fid = FIDEvaluation(cfg, datahandler, device)

    def evaluate(self, bornmachine: BornMachine, split, context):        
        self._generate(bornmachine, context)
        try:
            fid_score = self.fid.evaluate(context["synths"])
            logger.info(f"FID score is {fid_score:.4f}.")
            return fid_score
        except ValueError as e:
            logger.warning(str(e))
            return float('nan')

    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------Visualisation------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
# generator based
from .visualisation import visualise_samples

class VisualizationMetric(BaseMetric):
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

# Classifier based
from src.utils.evasion.minimal import RobustnessEvaluation

class RobustnessMetric(BaseMetric):
    def __init__(self, freq, cfg: schemas.Config, datahandler, device):
        super().__init__(freq, cfg, datahandler, device)
        self.evasion = RobustnessEvaluation(method=cfg.tracking.evasion.method,
                                            norm=cfg.tracking.evasion.norm,
                                            criterion=cfg.tracking.evasion.criterion,
                                            strengths=cfg.tracking.evasion.strengths)
        

    def evaluate(self, bornmachine: BornMachine, split, context):
        """
        Returns list of post attack accuracies, one for each strength provided.
        """
        robust_acc = self.evasion.evaluate(bornmachine, self.datahandler.classification[split], self.device)
        logger.info(f"Robust accuracy at strength={self.evasion.strengths[0]} is {robust_acc[0]:.3f}.")
        return robust_acc
        

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------Collecting metrics in a single class----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------

# TODO: Consider where to put bornmachine preparation
# Metric Factory
class MetricFactory:
    @staticmethod
    def create(
        metric_name: str, freq: int, 
        cfg: schemas.Config, 
        datahandler: DataHandler, 
        device: torch.device
        ) -> BaseMetric:
        mapping = {
            "loss": LossMetric,
            "acc": AccuracyMetric,
            "fid": FIDMetric,
            "rob": RobustnessMetric,
            "viz": VisualizationMetric,
        }
        return mapping[metric_name](freq, cfg, datahandler, device)


class PerformanceEvaluator:
    def __init__(
            self, 
            cfg: schemas.Config, 
            datahandler: DataHandler,
            train_cfg: schemas.ClassificationConfig | schemas.GANStyleConfig,
            metrics: Dict[str, int],
            device: torch.device
            ):
        

        self.metrics = {}
        # Only metrics in the config get initialized.
        for metric_name, freq in metrics.items():
            metric : BaseMetric = MetricFactory.create(
                metric_name=metric_name, freq=freq,
                cfg=cfg, datahandler=datahandler, device=device
            )
            self.metrics[metric_name] = metric
            if metric_name == train_cfg.stop_crit: 
                metric.freq = 1
            

    def should_evaluate(self, step: int, metric_name: str):
        freq = self.metrics[metric_name].freq
        return step % freq == 0

    def evaluate(self, 
                 bornmachine: BornMachine, 
                 split: str,
                 step: int):
        """
        Evaluates the bornmachine on a collection of metrics that are given in the tracking config of the experiment.
        Returns a dictionary of those metrics with keys in ['loss', 'acc', 'fid', 'rob', 'viz'].
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
        return results