
from typing import List, Optional, Union
import torch

from .callback import CallBack

class _MetricCallack(CallBack):
    """A callback specially designed for metrics monitoring

    Metrics are averaged over a periode of iterations, defined in the __init__
    Then the average value is returned every periode.

    Constants:
        REQUIRED_ARGS: The list of arguments needed to compute the metric.

    Methods:
        __init__(self, return_rule: List[int, str], 
                 its_per_epochs: int) -> None:
            Initialises the Metric
        action(self, event: str, jump: int, 
               *args, **kwargs) -> Union[float, None]:
            Updates the current metric and returns it if the return_rule is
            fullfilled.
    """
    REQUIRED_ARGS = []
    def __init__(self, return_rule: List[int, str], 
                 its_per_epochs: int) -> None:
        """Initialises the Metric

        Args:
            return_rule (int, str): If int, is interpreted as the sampling
                frequency. If str as to be in ["iteration_end", "epoch_end"].
            its_per_epochs (int): number of iterations (in individuals) per
                epochs. Equivalent to the number of elements in the training
                set.
        """
        super(_MetricCallack, self).__init__(its_per_epochs)
        self._metric = 0
        self._count = 0
        self._iteration = 0
        self._return_rule = return_rule

    def action(self, event: str, jump: int, 
               *args, **kwargs) -> Union[float, None]:
        """Updates the current metric and returns it if the return_rule is
        fullfilled.

        Args:
            event (str): the type of envent.
            jump (str): The number of iteration performed since last call to
                action.
            *args, **kwargs: The args needed to compute the metric

        Returns:
            metric (float): If required returns the averaged metric, else None
        """
        self._count += jump
        self._iteration += jump
        self._update_metrics(*args, **kwargs)
        if self._require_return(event):
            return self._return_metric()

        return None

    def _return_metric(self):
        """When the return rule is fullfilled, returns the metric and re-inits
        count and metric

        Returns:
            metric (float)
        """
        to_return = self.metric/self._count
        self._metric = 0
        self._count = 0

        return to_return

    def _require_return(self, event: str) -> bool:
        """Checks if the return_rule is fullfilled

        Args:
            event (str): The type of event.

        Returns:
            (bool)
        """
        if isinstance(self._return_rule, str):
            return event==self._return_rule

        elif isinstance(self._return_rule, int):
            return self._count >= self._return_rule

        return False

    def _update_metrics(self, *args, **kwargs):
        pass

class AccCallBack(_CallBack):
    """Callbacks that returns accuracy averaged over periodes of iterations
    """
    def _update_metrics(self, pred: torch.FloatTensor, 
                        target: torch.IntTensor) -> None:
        """Updates acc value

        Args:
            pred (torch.FloatTensor): A tensor of predictions, in the format
                (N, n_classes).
            target (torch.IntTensor): Ground truth, in the format (N).
        """
        pred = output.argmax(dim=1, keepdim=True)
        self.metric += pred.eq(target.view_as(pred)).sum().item()

class LossCallBack(_CallBack):
    """Callbacks that returns loss averaged over periodes of iterations
    """
    REQUIRED_ARGS = ["loss"]
    def _update_metrics(self, loss: float) -> None:
        """updates the loss

        Args:
            loss (float): The value of the loss
        """
        # TODO Check if it makes more sens to average the loss inside this
        # function. Mainly check what is traditionally returned by pytorch
        # losses
        self.metric += loss


