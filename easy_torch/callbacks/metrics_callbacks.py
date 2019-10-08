
from typing import List, Optional, Union
import torch

from .callback import CallBack

class _MetricCallack(CallBack):
    """A callback specially designed for metrics monitoring



    """
    REQUIRED_ARGS = []
    def __init__(self, return_rule: List[int, str], 
                 its_per_epochs: int) -> None:
        """

        """
        super(_MetricCallack, self).__init__(its_per_epochs)
        self._metric = 0
        self._count = 0
        self._return_rule = return_rule

        return None

    def action(self, event: str, jump: int, 
               *args, **kwargs) -> Union[float, None]:
        self._count += jump
        self._iteration += jump
        self._update_metrics(*args)
        if self._require_return(event):
            return self._return_metric()

        return None

    def _return_metric(self):
        to_return = self.metric/self._count
        self.__init__()

        return to_return

    def _require_return(self, event: str) -> bool:
        if isinstance(self._return_rule, str):
            return event==self._return_rule
        
        if isinstance(self._return_rule, int):
            return self._count >= self._return_rule

        return False

    def _update_metrics(self):
        pass

class AccCallBack(_CallBack):
    REQUIRED_ARGS = ["target", "pred"]
    def __init__(self):
        super(AccCallBack, self).__init__()

    def _update_metrics(self, pred: torch.FloatTensor, 
                        target: torch.IntTensor) -> None:
        pred = output.argmax(dim=1, keepdim=True)
        self.metric += pred.eq(target.view_as(pred)).sum().item()

class LossCallBack(_CallBack):
    REQUIRED_ARGS = ["loss"]
    def __init__(self):
        super(LossCallBack, self).__init__()

    def _update_metrics(self, loss: torch.FloatTensor) -> None:
        self.metric += loss


