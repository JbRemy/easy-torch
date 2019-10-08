
from typing import List, Optional, Union
import torch

class CallBack(object):
    """A CallBack is called every time an iteration ends, it can monitor
    metrics or modify parameters of the network.

    Methods:
        iteration_end(self, jump: int, *args, **kwargs) -> Union[Tuple, None]:
            When an iteration ends, some action is to be performed by the
            callback. If it is also an epoch end, some other action can be
            performed.
        action(self, event, jump, *args, **kwargs):
            The action to perform, depends on the child class.
    """
    def __init__(self, its_per_epochs: int) -> None:
        """Initilises the class

        Args:
            its_per_epochs (int): number of iterations (in individuals) per
                epochs. Equivalent to the number of elements in the training
                set.
        """
        self._iteration = 0
        self._epoch = 0
        self._its_per_epochs = its_per_epochs

    def iteration_end(self, jump: int, *args, **kwargs)\
            -> Union[Tuple, None]:
        """When an iteration ends, performs the required action

        Args:
            jump (int): The number of iterations passed since last
                iteration_end call. Should be batch_size.
            *args, **kwargs: The arguments for the action function.
        """
        it_end_res = self.action("iteration_end", jump, *args, **kwargs)
        if self._iteration // self._its_per_epochs != self._epoch:
            self._epoch += 1
            epoch_end_res = self.action("epoch_end", jump, *args, **kwargs)

        else:
            epoch_end_res = None

        self._iteration += jump

        return it_end_res, epoch_end_res

    def action(self, event, jump, *args, **kwargs):
        pass