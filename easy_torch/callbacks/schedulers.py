
from typing import List, Optional, Union

import torch
from torch import nn

from ._callback import _CallBack

# TODO: Documentation

class _Scheduler(_CallBack):
    """A super class of schedulers for optimizer hyperparameters.

    constants:
        CALL_AT (str): when to call the callback


    """
    CALL_AT = "train"
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 *args, **kwargs) -> None:
        """Initializes the scheduler

        Args:
            optimizer (nn.optim.optimizer): The optimizer to update.
            *args, **kwargs: see Callback.__init__
        """
        super(_Scheduler, self).__init__(*args, **kwargs)
        self._optimizer = optimizer

    def _modif_optimizer(self, key: str, increment: Optional[float]=None,
                         decay: Optional[float]=None) -> None:
        """Updates the optimizer hyperparamters with respect to the provided
        values.

        Only one of decay and increment must be provided.

        Args:
            key (str): The name of the hyperparameter to decay
            increment (float): param +=  increment
            decay (float): param *= decay
        """
        for param_group in self._optimizer.param_groups:
            if decay:
                param_group[key] *=  decay

            if increment:
                param_group[key] += increment

class StepDecay(_Scheduler):
    """A Step wise decay for learing rate. 

    At the begining of every provided epochs, the learning rate is decayed by rate.

    Attributes:
        updated (bool): True if the decay was perform at least one.
    """
    def __init__(self, steps: Union[List[int], int], rate: float,
                 *args, **kwargs) -> None:
        """Initializes the StepDecay scheduler

        Args:
            steps (list, int): The steps at which learning rate decay should be
                performed.
            rate (float): The decay rate lr*=rate.
        """
        super(StepDecay, self).__init__(*args, **kwargs)
        self._steps = steps if isinstance(steps, list) else [steps]
        self._rate = rate
        self.updated = False

    def _action(self, event: str, jump: int) -> None:
        """If a provided epoch starts, learning rate gets decayed.

        Ars:
            event (str): the type of envent.
            jump (str): The number of iteration performed since last call to
                action.
        """
        if event == "epoch_end":
            if self._epoch in self.steps:
                self._modif_optimizer('lr', decay=self._rate)
                self.updated = True
    
class TriangularSchedule(_Scheduler):
    def __init__(self, half_cycle, lr_bounds: [float],
                 momentum_bounds: [float]=[0.85, 0.9], 
                 increase_first: bool=True, *args, **kwargs) -> None:
        super(StepDecay, self).__init__(*args, **kwargs)
        self._slope_sign = 1 if increase_first else -1
        self._lr_slope = self.get_slope(lr_bounds, half_cycle)
        self._momentum_slope = self.get_slope(momentum_bounds,
                                              half_cycle)
        self._current_half_cycle = 0

    def _action(self, event: str, jump: int) -> None:
        """If a provided epoch starts, learning rate gets decayed.

        Ars:
            event (str): the type of envent.
            jump (str): The number of iteration performed since last call to
                action.
        """
        if event == "iteration":
            lr_increment=self._get_increment(jump, "lr")
            self._modif_optimizer('lr', increment=lr_increment)
            momentum_increment=self._get_increment(jump, "momentum")
            self._modif_optimizer('momentum', increment=momentum_increment)

        if event == "epoch":
            if self._epoch // self._half_cycle != self._current_half_cycle:
                self._current_half_cycle += 1
                self._slope_sign = -self._slope_sign

        return None

    def _get_increment(self, jump, param):
        return getattr(self, "_%s_slope" % param)*self._slope_sign*self._iteration*jump

    def _get_slope(self, bounds: [float], half_cycle: int) -> float:
        slope = max(bounds) - min(bounds)
        slope /= (self._its_per_epochs*half_cycle)
        
        return slope

