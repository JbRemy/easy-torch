
from typing import List, Optional, Union
import torch

class _CallBack(object):
    """A CallBack is called every time an iteration ends, it can monitor
    metrics or modify parameters of the network.

    Constants:
        CALL_AT (str): When to call the callback

    Methods:
        iteration_end(self, jump: int, *args, **kwargs) -> Union[Tuple, None]:
            When an iteration ends, some action is to be performed by the
            callback. If it is also an epoch end, some other action can be
            performed.
        action(self, event, jump, *args, **kwargs):
            The action to perform, depends on the child class.
    """
    CALL_AT = ""
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

        Returns:
            it_end_res, epoch_end_res:
                Depending on the type of callbacks something may be returned.
                otherwise both are Nones.
        """
        self._iteration += jump
        it_end_res = self._action("iteration_end", jump, *args, **kwargs)
        if self._iteration // self._its_per_epochs != self._epoch:
            self._epoch += 1
            epoch_end_res = self._action("epoch_end", 0, *args, **kwargs)

        else:
            epoch_end_res = None

        return it_end_res, epoch_end_res

    def _action(self, event: str, jump: int, *args, **kwargs):
        """The callback's action

        Ars:
            event (str): the type of envent.
            jump (str): The number of iteration performed since last call to
                action.
            *args, **kwargs
        """
        pass

