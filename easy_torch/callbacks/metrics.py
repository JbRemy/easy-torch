
from typing import List, Optional, Union
import torch

from ._callback import _CallBack


# TODO: Add multiclass, weighted metrics

class _MetricCallBack(_CallBack):
    """A callback specially designed for metrics monitoring

    metrics are averaged over a periode of iterations, defined in the __init__
    then the average value is returned every periode.

    constants:
        required_args: the list of arguments needed to compute the metric.
        CALL_AT (str): when to call the callback

    methods:
        __init__(self, return_rule: list[int, str], 
                 its_per_epochs: int) -> none:
            initialises the metric
        action(self, event: str, jump: int, 
               *args, **kwargs) -> union[float, none]:
            updates the current metric and returns it if the return_rule is
            fullfilled.
        testify(cls):
            returns another class that require test metrics.
    """
    CALL_AT = "train"
    REQUIRED_ARGS = []
    def __init__(self, return_rule: List[Union[int, str]], 
                 *args, **kwargs) -> None:
        """Initialises the Metric

        Args:
            return_rule (int, str): If int, is interpreted as the sampling
                frequency. If str as to be in ["iteration_end", "epoch_end"].
            *args, **kwargs: see Callback.__init__
        """
        super(_MetricCallBack, self).__init__(*args, **kwargs)
        self._metric = 0
        self._count = 0
        self._iteration = 0
        self._return_rule = return_rule

    def _action(self, event: str, jump: int, 
               *args, **kwargs) -> Union[float, None]:
        """Updates the current metric and returns it if the return_rule is
        fullfilled.

        Args:
            event (str): the type of envent.
            jump (str): The number of iteration performed since last call to
                action.
            *args, **kwargs: The args needed to compute the metric REQUIRED_ARGS

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
        to_return = self._metric/self._count
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

    # TODO @Simon: What do you thnik ? Is this the proper way do do it ?
    # Also does it work for the subclasses of this meta class
    @classmethod
    def testify(cls):
        """Returns another class that produces the test metric
        """
        class TestClass(cls):
            """A test version of %(class_name)s.

            The difference with the parent class is that the REQUIRED_ARGS are
            tests version of those args, and the return rule becaomes
            "epoch_end".

            see: %(class_name)s
            """ 
            def __init__(self, *args, **kwargs):
                super().__init__("epoch_end", *args, **kwargs)

        TestClass.REQUIRED_ARGS = ["_test%s" % _ for _ in cls.REQUIRED_ARGS]
        TestClass.CALL_AT = "test"
        TestClass.__name__ = "Test%s" % cls.__name__
        TestClass.__doc__ %= {"class_name": cls.__name__}

        return TestClass

class Acc(_MetricCallBack):
    """Callbacks that returns accuracy averaged over periodes of iterations
    """
    REQUIRED_ARGS = ["_output", "_target"]
    def _update_metrics(self, output: torch.FloatTensor, 
                        target: torch.IntTensor) -> None:
        """Updates acc value

        Args:
            pred (torch.FloatTensor): A tensor of predictions, in the format
                (N, n_classes).
            target (torch.IntTensor): Ground truth, in the format (N).
        """
        pred = output.argmax(dim=1, keepdim=True)
        self._metric += pred.eq(target.view_as(pred)).sum().item()

TestAcc = Acc.testify()

class Loss(_MetricCallBack):
    """Callbacks that returns loss averaged over periodes of iterations
    """
    REQUIRED_ARGS = ["_loss"]
    def _update_metrics(self, loss: float) -> None:
        """updates the loss

        Args:
            loss (float): The value of the loss
        """
        # TODO Check if it makes more sens to average the loss inside this
        # function. Mainly check what is traditionally returned by pytorch
        # losses
        self._metric += loss

TestLoss = Loss.testify()

