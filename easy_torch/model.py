from typing import List, Optional, Union, Sequence

import os
from copy import copy
import warning

import torch 
import torch.nn as nn

from tqdm import tqdm 
import numpy as np

from .network import Network
from .callbacks import schedulers, _callback
from .helpers.torch import get_device, set_seed, send_to

# TODO: We need to define the syntax for network definition somewhere.
# TODO: Probably define a more proper way to store the metrics

class Model(object):
    """The overlord class to manage and train neural networks.

    The public attributes are the ones that need to be saved when saving the
    model.
    
    Attributes:
        layers (list): A list of layers following the correct syntax.
        device (torch.device)
        criteron (str)
        optimizer_name (str)
        optimizer_kwargs (dict)
        training_seed (int)
        initialization_seed (int)
    Methods:
        __init__(self, layers: List[str], device: str="auto", 
                 seed: Optional[int]=None) -> None:
           Defines architectural parameters of the network.
        compile(self, optimizer: str, criteron: str, 
                optimizer_kwargs: Optional[dict]=None, 
                criteron_kwargs: Optional[dict]=None) -> None:
            Defines the training aspects of the model.

    """
    def __init__(self, layers: List[str], device: str="auto", 
                 seed: Optional[int]=None) -> None:
        """Defines architectural parameters of the network.

        The network is not build at this point because we need to see the
        data shape for that.

        Args:
            layers (list): A list of layers following the correct syntax.
            device (str): On which device sould the model be send. "cpu",
                "gpu", "auto".
                (default: "auto")
            seed (int): The seed for initialisation of the parameters.
        """
        self.layers = layers
        self.device = get_device(device)
        self.initialization_seed = seed if seed else np.random.randint()
        self.criteron = None
        self.optimizer_name = None
        self.optimizer_kwargs = None
        self.training_seed = None
        self.record = {}
        # TODO @Simon: Properly manage warnings ? 
        if seed and self.device == torch.device("cuda"):
            if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
                warning.warn(
                    """
                    You specified a seed but don't meet the conditions for
                    reproducibility when using the CDNN backend of pytorch. You
                    may want to add : "torch.backends.cudnn.deterministic =
                    True;  torch.backends.cudnn.benchmark = False" to the
                    begining of your script.
                    """,
                    Warning
                )

        self._optimizer = None
        self._current_info = {}
        self._train_data = None
        self._train_target = None
        self._train_output = None
        self._train_loss = None
        self._test_data = None
        self._test_target = None
        self._test_output = None
        self._test_loss = None
        self._test_callbacks = None
        self._train_callbacks = None

    def compile(self, optimizer: str, criteron: str, 
                optimizer_kwargs: Optional[dict]=None, 
                criteron_kwargs: Optional[dict]=None) -> None:
        """Defines the training aspects of the model.

        Once a model is compiled, we know what architecture to train on which
        criteron with which algorithm.
        The optimizer is not instanciated at this point becaus we need the
        network parameters to be initialized.

        Args:
            optimizer (str): The name of a torch.optim.optimizer.
                https://pytorch.org/docs/stable/optim.html
            criteron (str): The name of a troch.nn Loss function.
                https://pytorch.org/docs/stable/nn.html
            optimizer_kwargs (dict): arguments for said optimizer.
            criteron_kwargs (dict): arguments for said criteron.
        """
        self.criteron = getattr(nn, criteron)(**criteron_kwargs)
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs

    def train(self, train_loader: torch.utils.data.DataLoader, 
              epochs: int,
              log_folder: Optional[str]=None, 
              log_freq: Optional[int]=None, 
              test_loader: Optional[torch.utils.data.DataLoader]=None, 
              test_freq: Optional[int]=None,
              callbacks: Optional[list[Union[str, _callback._CallBack]]]=None) -> None:
        # TODO: Find a way to pass strings callbacks and initializes them
        # automatically, in a User friendly way. For now you need to init them
        # before calling train.
        """Trains the network

        Args:
            train_loader (torch.utils.data.DataLoader): A data loader for
                training data.
            epochs (int): The total number of epochs.
            log_folder (str): Where to log training data, and save the model if
                needed.
            log_freq (int): Frequency of logging results (in iterations).
            test_loader (torch.utils.data.DataLoader): A data loader for
                testing data.
            test_freq (int): Test frequency.
            callbacks (list): A list of callbacks to apply, already
                instanciated.
        """
        self._initialize_weights_and_optimizer(train_loader_SHAPE) # TODO train_loader_SHAPE
        self._initialize_callbacks(callbacks)

        for epoch in epochs:
            _pbar = self._initialize_pbar(epoch, epochs)
            for i, (data, target) in enumerate(train_loader):
                if i & test_freq == 0:
                    self._test(test_loader)

                jump = data.size(0)
                self._data, self._target = send_to([data, target], self.device)
                self._train_data, self._train_target = send_to([test_data, test_target], self.device)
                self._train_output = self.network(self._train_data)
                self._train_loss = criteron(self._train_output, self._train_target)
                self._train_loss.backward()
                self._optimizer.step()
                self._call_callbacks("train", jump)

                pbar.update(batch_size)
                pbar.set_postfix(self._current_infos)

                if self._check_terminate():
                    return None

        return None

    def _test(self, loader) -> None:
        """One pass on the test dataset

        Args:
            loader (torch.utils.data.DataLoader): A data loader for testing data.
        """
        for i, (test_data, test_target) in enumerate(loader):
            test_jump = data.size(0)
            self._test_data, self._test_target = send_to([test_data, test_target], self.device)
            self._test_output = self.network(self._test_data)
            self._test_loss = criteron(self._test_output, self._test_target)
            self._call_callbacks("test", test_jump)

    def _call_callbacks(self, time: "str"):
        """Calls all reqiured callbacks

        Args:
            time (str): "test" or "train"
        """
        for callback in getattr(self, "_%s_callbacks" % time):
            args = [getattr(self, _) for _ in callbacks.REQUIRED_ARGS]
            self._store_res(callback.__name__, callbacks.iteration_end(jump, args))

    def _store_res(self, name: str, res: Sequence):
        """Stores results in self.record if needed, at the iteration address

        Also updates the _current_info dict, storing current values of metrics.

        Args:
            name (str): the name of the result
            res (iterable): the metrics to store
        """
        # TODO @Simon: What do you think about this ? The point is that you dont
        # check if name exists.
        for res_ in res:
            if res_:
                try:
                    self.record[name][iterations] = res_

                except KeyError:
                    self.record[name] = {}
                    self.record[name][iterations] = res_

            self._current_info.update({name: res_})

    def _initialize_weights_and_optimizer(self, input_shape: Sequence[int]) -> None:
        """Builds the Network, initializes it's weights, and creates the
        optimizer.

        Args:
            input_shape (list, tuple): The input shape of the data.
        """
        set_seed(self.initialisation_seed)
        self._network = Network(self.layers, self.device, input_shape) 
        self._optimizer = getattr(torch.optim, self.optimizer_name)(
            self.network.parameters(),
            **self.optimizer_kwargs
        )

    def _initialize_callbacks(self, callbacks: List[_CallBack]) -> None:
        """Build all the required callbacks

        Args:
            callbacks: Optional[list[Union[str, _callback._CallBack]]]
        """
        self._test_callbacks = [callback for callback in callbacks if callback.CALL_AT=="test"] 
        self._train_callbacks = [callback for callback in callbacks if callback.CALL_AT=="train"] 

    def _initialize_pbar(self, epoch: int, epochs: int) -> tqdm:
        """Initializes a progress bar for the current epoch

        Args:
            epoch (int): current epoch
            epochs (int): total number of epochs.

        Returns:
            the progess bar (tqdm.tqdm)
        """
        return tqdm(total=its_per_epochs, unit_scale=True,
                    desc="Epoch %i/%i" % (epoch+1, epochs),
                    postfix={})

    def _check_terminate(self) -> bool:
        """Cheks if a KILL order as been given

        Return:
            (bool)
        """
        return "KILL" in self._current_info.values()




















