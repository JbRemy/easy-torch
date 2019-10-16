from typing import List, Optional, Union, Sequence, Tuple

import os
from copy import copy
import warnings

import torch 
import torch.nn as nn

from tqdm import tqdm 
import numpy as np

from .network import Network
from .callbacks import schedulers, _callback
from .helpers.torch import get_device, set_seed, send_to
from .helpers.general import save_json, exists_or_create_dir

# TODO: We need to define the syntax for network definition somewhere.
# TODO: Probably define a more proper way to store the metrics
# TODO: save callbacks

class Model(object):
    """The overlord class to manage and train neural networks.

    The public attributes are the ones that need to be saved when saving the
    model.
    
    Attributes:
        layers (list): A list of layers following the correct syntax.
        device (torch.device)
        criterion (str)
        optimizer_name (str)
        optimizer_kwargs (dict)
        training_seed (int)
        initialization_seed (int)
    Methods:
        __init__(self, layers: List[str], device: str="auto", 
                 seed: Optional[int]=None) -> None:
           Defines architectural parameters of the network.
        compile(self, optimizer: str, criterion: str, 
                optimizer_kwargs: Optional[dict]=None, 
                criterion_kwargs: Optional[dict]=None) -> None:
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
        self.device = device
        self.initialization_seed = seed if seed else np.random.randint()
        self.criterion = None
        self.criterion_kwargs = None
        self.optimizer_name = None
        self.optimizer_kwargs = None
        self.training_seed = None
        self.record = {}
        self.iteration = 0
        # TODO @Simon: Properly manage warnings ? 
        if seed and self.device == "cuda":
            if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
                warnings.warn(
                    """
                    You specified a seed but don't meet the conditions for
                    reproducibility when using the CDNN backend of pytorch. You
                    may want to add : "torch.backends.cudnn.deterministic =
                    True;  torch.backends.cudnn.benchmark = False" to the
                    begining of your script.
                    """,
                    Warning
                )

        self._device = get_device(device)
        self._optimizer = None
        self._current_info = {}
        self._data = None
        self._target = None
        self._output = None
        self._loss = None
        self._test_data = None
        self._test_target = None
        self._test_output = None
        self._test_loss = None
        self._test_callbacks = None
        self._train_callbacks = None

    def compile(self, optimizer: str, criterion: str, 
                optimizer_kwargs: Optional[dict]=None, 
                criterion_kwargs: Optional[dict]=None) -> None:
        """Defines the training aspects of the model.

        Once a model is compiled, we know what architecture to train on which
        criterion with which algorithm.
        The optimizer is not instanciated at this point becaus we need the
        network parameters to be initialized.

        Args:
            optimizer (str): The name of a torch.optim.optimizer.
                https://pytorch.org/docs/stable/optim.html
            criterion (str): The name of a troch.nn Loss function.
                https://pytorch.org/docs/stable/nn.html
            optimizer_kwargs (dict): arguments for said optimizer.
            criterion_kwargs (dict): arguments for said criterion.
        """
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs
        self._criterion = getattr(nn, criterion)(**criterion_kwargs)
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs

    def train(self, train_loader: torch.utils.data.DataLoader, 
              epochs: int,
              log_folder: Optional[str]=None, 
              log_freq: Optional[int]=None, 
              test_loader: Optional[torch.utils.data.DataLoader]=None, 
              test_freq: Optional[int]=None,
              callbacks: Optional[List[Union[str, _callback._CallBack]]]=None) -> None:
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
        size, its_per_epochs = self._get_size_and_its_per_epochs(train_loader)

        self._initialize_weights_and_optimizer(size) # TODO train_loader_SHAPE
        self._initialize_callbacks(callbacks)

        for epoch in range(epochs):
            _pbar = self._initialize_pbar(epoch, epochs, its_per_epochs)
            for i, (data, target) in enumerate(train_loader):
                jump = data.size(0)
                if self._check_crossed(test_freq, jump):
                    self._test(test_loader)

                self._data, self._target = send_to([data, target], self._device)
                self._output = self._network(self._data)
                self._loss = self._criterion(self._output, self._target)
                self._optimizer.zero_grad()
                self._loss.backward()
                self._optimizer.step()
                self._call_callbacks("train", jump)

                _pbar.update(jump)
                _pbar.set_postfix(self._current_info)

                if self._check_save() or self._check_crossed(log_freq, jump):
                    self.save_model(log_folder)

                if self._check_terminate():
                    return None # TODO @Simon: Is this the correct way to kill the function ?

                self.iteration += jump

        self.save_record(log_folder)

    def save_model(self, folder: Optional[str]=None, 
             fname: Optional[str]=None) -> None:
        """Saves the model 

        This saves all public attributes + the dict and optimizer state dicts.
        This allows to reload the model when needed.

        Args:
            folder (str): The directory where to save
            fname (str): The prefix for every file inside the folder.
                If None fname is the callback that triggered the SAVE order.
        """
        exists_or_create_dir(folder)
        to_save = {}
        for attr in [attr for attr in dir(self) if not attr.startswith('_') and
                    not callable(getattr(self, attr)) and not attr == "record"]:
            to_save.update({attr: getattr(self, attr)})

        fname = fname if fname else "".join([k for k,v in self._current_info.items() if v=="SAVE" ])
        fname += "_%i.pt" % self.iteration
        to_save.update({
            'model_state_dict': self._network.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
        })
        torch.save(to_save, os.path.join(folder, fname))

    def save_record(self, folder: Optional[str]=None, 
             fname: str="record") -> None:
        """Saves the record 

        Args:
            folder (str): The directory where to save
            fname (str): The prefix for every file inside the folder.
                If None fname is the callback that triggered the SAVE order.
        """
        save_json(self.record, os.path.join(folder, fname))

    def _test(self, loader) -> None:
        """One pass on the test dataset

        Args:
            loader (torch.utils.data.DataLoader): A data loader for testing data.
        """
        if len(self._test_callbacks) > 0:
            for i, (test_data, test_target) in enumerate(loader):
                test_jump = test_data.size(0)
                self._test_data, self._test_target = send_to([test_data, test_target], self._device)
                self._test_output = self._network(self._test_data)
                self._test_loss = self._criterion(self._test_output, self._test_target)
                self._call_callbacks("test", test_jump)

    def _call_callbacks(self, time: "str", jump: int):
        """Calls all reqiured callbacks

        Args:
            time (str): "test" or "train"
            jump (int): number of passed iteration
        """
        for callback in getattr(self, "_%s_callbacks" % time):
            args = [getattr(self, _) for _ in callback.REQUIRED_ARGS]
            self._store_res(callback.__class__.__name__, callback.iteration_end(jump, *args))

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
                    self.record[name][str(self.iteration)] = res_

                except KeyError:
                    self.record[name] = {}
                    self.record[name][str(self.iteration)] = res_

                self._current_info.update({name: res_})


    def _initialize_weights_and_optimizer(self, input_shape: Sequence[int]) -> None:
        """Builds the Network, initializes it's weights, and creates the
        optimizer.

        Args:
            input_shape (list, tuple): The input shape of the data.
        """
        set_seed(self.initialization_seed)
        self._network = Network(self.layers, self._device, input_shape) 
        self._optimizer = getattr(torch.optim, self.optimizer_name)(
            self._network.parameters(),
            **self.optimizer_kwargs
        )

    def _initialize_callbacks(self, callbacks: Optional[List[_callback._CallBack]]=None) -> None:
        """Build all the required callbacks

        Args:
            callbacks: Optional[list[Union[str, _callback._CallBack]]]
        """
        if callbacks:
            self._test_callbacks = [callback for callback in callbacks if callback.CALL_AT=="test"] 
            self._train_callbacks = [callback for callback in callbacks if callback.CALL_AT=="train"] 
        
        else:
            self._test_callbacks, self._train_callbacks = [], []


    def _initialize_pbar(self, epoch: int, epochs: int, its_per_epochs: int) -> tqdm:
        """Initializes a progress bar for the current epoch

        Args:
            epoch (int): current epoch
            epochs (int): total number of epochs.
            its_per_epochs (int)

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

    def _check_save(self) -> bool:
        # TODO @Simon: you can't save if you didnt specify a log_folder, but
        # nothing prevents it now, should we hard code it here. Like:
        #   return "SAVE" in self._current_info.values() and log_folder
        # Or add a test in the begining of train ?
        """Cheks if a SAVE order as been given

        Return:
            (bool)
        """
        return "SAVE" in self._current_info.values()

    def _check_crossed(self, freq: int, jump: int) -> bool:
        """Checks if iterations just cross a frequency point
        
        Args:
            freq (int)
            jump (int): The update before last check

        Return: 
            (bool)
        """
        return self.iteration // freq > (self.iteration-jump) // freq 

    # TODO @Simon: When should @satticmethod be specified ?
    @staticmethod
    def _get_size_and_its_per_epochs(loader: torch.utils.data.DataLoader)\
            -> Tuple[Tuple[int], int]:
        """Returns batch_size and number of iterations per epochs

        Args:
            loader (torch.utils.data.DataLoader)

        Returns:
            batchsize (tuple)
            its_per_epochs (int)
        """
        for i, (data, target) in enumerate(loader):
            size = tuple(data.size())
            break

        return size, len(loader.dataset)



















