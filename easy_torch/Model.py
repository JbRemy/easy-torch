from __future__ import annotations
import os

from typing import List, Optional, Union

import torch 
from torchvision import datasets, transforms
import torch.nn as nn

from tqdm import tqdm 

import numpy as np

from .Network import Network
from .DataLogger import DataLogger

class Model(object):
    """A class used to define and train a Neural Network.

    Attributes:
        network: Instance of the Network class.
        optimizer: A torch.nn optimizer.

    Methodds:
        train(train_loader: torch.utils.data.DataLoader, 
              test_loader: torch.utils.data.DataLoader, 
              epochs: int, 
              criteron: str, 
              optimizer: str,
              criteron_kwargs: dict={}, 
              optimizer_kwargs: dict={"lr": 0.1}, 
              log_folder: Optional[str]=None, 
              log_freq: int=100, 
              test_freq: int=1000,
              required_checkpoints: Optional[Union[int, List[int]]]=None, 
              save_best_model: Optional[str]=None, 
              patience: Optional[int]=None) -> None
            Trains the model with the specified hyperparameters. 
            If a log_folder is specified, can log the results and save
            checkpoints of the model

        save(file_name: str) -> None
            Saves the model in the form of a checkpoint.

        build(self,  optimizer: str, input_shape: list,
              optimizer_kwargs: dict={}) -> None
            Builds the model, usefull only to initialize the model on
            predefined weights before training.

        load(cls, file_name: str, modif: dict={}, 
             resume: bool=True) -> Model
            (classmethod) Loads a model from a checkpoint file.

    See __main__ for an example.
    """
    def __init__(self, layers: List[str], device: str="auto", 
                 seed: Optional[int]=None) -> None:
        """Initializes the Model class with its architecture and device.
        
        Args:
            layers (list): A list of string describing layers, see Network.
            device (str): A string defining the device to run the network.
                (default: "auto")
            seed (int): Seed for training and initialization

        Returns:
            None
        """
        self.network = None
        self.optimizer = None
        self._dict = {"layers": layers, "device": device, "seed": seed}
        self._iteration = 0
        self._logger = None
        self._best_metric = None
        self._seed = seed

        return None

    def train(self, train_loader: torch.utils.data.DataLoader, 
              test_loader: torch.utils.data.DataLoader, 
              epochs: int, criteron: Optional[str]=None, 
              optimizer: Optional[str]=None,
              criteron_kwargs: dict={}, optimizer_kwargs: dict={"lr": 0.1}, 
              log_folder: Optional[str]=None, log_freq: int=100, 
              test_freq: int=1000,
              required_checkpoints: Optional[Union[int, List[int]]]=None, 
              save_best_model: Optional[str]=None, 
              patience: Optional[int]=None) -> None:
        """Trains the network on the train_loader, evaluates it on the test_loader. 

        Starts by building the network with respect ot the input data shape,
        creates the optimizer and criteron. Then trains the network.
        If the model is already build this stage is skipped. This allows to
        force the weigths of the netowrk.
        It is important to understand that everything is counted in terms of
        iterations. then if log_freq is 100 and batch_size is 64, the first log
        happens on the second batch.
        If save_best_model is defined, the best_model is evaluated based on
        testing frequency. So is the metric for early stopping.
        If self.iteration is already greater than zero, for exemple after a
        load from a checkpoint, training jumps directly to this iteration.

        >>>  model.train(train_loader, test_loader, log_folder="./test", 
        >>>              required_checkpoints = [1000, 2000],
        >>>              save_best_model="test_acc", patience=1000)
        test/
        | 
        |- res.json
        |- Best_Model.pt
        |- CheckPoint_1000.pt
        |- CheckPoint_2000.pt

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for
                training.
            test_loader (torch.utils.data.DataLoader): DataLoader for testing.
            epochs (int): An integer that sets the maximum number of ecpochs
            criteron (str): A string representing the torch.nn criteron to use.
                To get a list of possibilities checkout torch.nn documentation.
                (default: None)
            optimizer (str): A string representing the torch.optim optimizer to
                use. To get a list of possibilities checkout torch.optim 
                documentation.
                (default: None)
            criteron_kwargs (dict): A dict containing the arguments to pass to
                the criteron.
                (default: {})
            optimizer_kwargs (dict): A dict containing the arguments to pass to
                the optimizer.
                (default: {"lr": 0.1})
            log_folder (str): Directory to wich logs and model should be saved.
                (default: None)
            log_freq (int): At wich frequency should training metrics be
                logged.
                (defaul: 100)
            test_freq (int): At wich frequency should testing metrics be
                logged.
                (defaul: 1000)
            requiered_checkpoints (list, int): A list, or only integer, to
                force some checkpoints at some iterations.
                (default: None)
            save_best_model str: A string to define which metric to monitor to 
                define and save the best model. If None, no model is saved.
                (default: None)
            patience (int): If define, how many iterations to wait before
                terminating training if no improvement is seen, early stopping.

        Returns:
            None
        """
        if not self._build:
            self.build(optimizer, np.shape(train_loader[0]), optimizer_kwargs)

        self._logger = DataLogger(log_folder, log_freq, test_freq)
        if self.iteration > 0:
            wait_to_start = self._iteration

        wait_to_start = 0
        patience_count = 0
        criteron = getattr(nn, criteron)(**criteron_kwargs)
        done = False
        if not self.seed is None:
            self._set_ssed(self.seed)

        for epoch in range(epochs):
            with tqdm(total=len(train_loader)*train_loader.batch_size, unit_scale=True,
                     desc="Epoch %i/%i" % (epoch+1, epochs),
                      ncols=150, postfix={"train_acc": 0.0, "test_acc": 0.0}) as pbar:
                for i, (data, target) in enumerate(train_loader):
                    if self._iteration >= wait_to_start:
                        self.network.train()
                        data  = data.to(self.network.device)
                        target = target.to(self.network.device)
                        output = self.network(data)
                        self.optimizer.zero_grad()
                        loss = criteron(output, target)
                        loss.backward()
                        self.optimizer.step()

                        self._logger.update("train", self._iteration,
                                           loss, output, target)
                        
                        if self._logger.require_log(self._iteration, "test"):
                            for test_i, (test_data, test_target) in enumerate(test_loader):
                                self.network.eval()
                                test_data  = test_data.to(self.network.device)
                                test_target = test_target.to(self.network.device)
                                test_output = self.network(test_data)
                                test_loss = criteron(test_output, test_target)
                                self._logger.update("test", self._iteration, test_loss, 
                                                   test_output, test_target, 
                                                   force_log=(test_i+1)==len(test_loader))

                            if (not save_best_model is None) or (not patience is None):
                                if self._monitor(getattr(self._logger, save_best_model),
                                                save_best_model) and not patience is None:
                                    patience_count = self._iteration

                                if self._iteration - patience_count > patience:
                                    done=True
                                    break

                    self._logger.previous_iteration = self._iteration
                    self._iteration += data.size(0)
                    pbar.update(data.size(0))
                    pbar.set_postfix({"train_acc": self._logger.train_acc,
                                      "test_acc": self._logger.test_acc})
                    self._logger.save()
                
                if done :
                    break
        return None

    def save(self, file_name: str) -> None:
        """Saves the model to file_name.

        Saves a dict with all the necessary informations to reload the model.
        
        Args:
            file_name (str): The path to save the model.

        Returns:
            None
        """
        save_dict = {
            "iteration": self._iteration,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        save_dict.update(self._dict)
        torch.save(save_dict, file_name)

        return None

    def  build(self,  optimizer: str, input_shape: list,
               optimizer_kwargs: dict={}):
        """Builds the model and the opotimizer.

        First instanciates the Netowrk class based on the mentionned
        architechture. 
        Then initializes the optimizer based on the passed arguments.

        Args: 
            optimizer (str): A string representing the torch.optim optimizer to
                use. To get a list of possibilities checkout torch.optim 
                documentation.
            input_shape (list): The shape of data when entering the network.
            optimizer_kwargs (dict): A dict containing the arguments to pass to
                the optimizer.
                (default: {"lr": 0.1})

        Returns:
            None
        """
        if not self.seed is None:
            self._set_seed(seed)

        self.network = Network(self._dict["layers"], 
                               device=self._dict["device"], 
                               input_shape=input_shape)
        self._dict.update({"optimizer": optimizer, 
                          "optimizer_kwargs": optimizer_kwargs,
                          "input_shape": input_shape})
        self.optimizer = getattr(torch.optim, optimizer)(self.network.parameters(), 
                                                **optimizer_kwargs)

        return None
        
    def _monitor(self, res: float, save_best_model: Optional[str]) -> bool:
        """Registers the metric and saves the best model if needed.

        If performance is better registers the new metric and, if
        save_best_model is defined, saves the model.

        Returns:
            Bool: True if the new metric is an improovement, False otherwise.
        """

        if self._best_metric is None:
            self._register(res, save_best_model)
            return True

        elif self._best_metric < res:
            self._register(res, save_best_model)
            if not save_best_model is None:
                self.save(os.path.join(self._logger.folder_name, "Best_Model.pt"))

            return True
        
        return False
                    
    def _register(self, res: float, save_best_model: str) -> None:
        """Saves result

        Args:
            res (float): The value to store
            save_best_model (str): The type of metric.

        Returns:
            None
        """
        if "loss" in save_best_model:
            self._best_metric = -res

        elif "acc" in save_best_model:
            self._best_metric = res

        return None

    @classmethod
    def load(cls, file_name: str, modif: dict={}, resume: bool=True) -> Model:
        """Loads a model from a checkpoint.
    
        First loads the checkpoint, then modifies the dictionary based on the
        midif dict. Usefull to modify the  optimizer args for example.
        Then inits the new Model and builds it.
        If resume is True, restores tranining elements to their states.

        Args:
            file_name (str): The name of the checkpoint file
            modif (dict): A modification dictionary for the settings of the
                Network. For example to modify the learing rate you can pass 
                {"optimizer_kwargs": {"lr":0.1}}
                (default: {})
            resume (bool): If True restores traning to where it stopped.
        """
        checkpoint = torch.load(file_name)
        for k,v in modif.items():
            checkpoint[k] = v
        new = cls(layers=checkpoint["layers"], device=checkpoint["device"]) 
        new._build(optimizer=checkpoint["optimizer"], 
                   optimizer_kwargs=checkpoint["optimizer_kwargs"], 
                   input_shape=checkpoint["input_shape"])
        new.network.load_state_dict(checkpoint["model_state_dict"])
        if resume:
            new.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            new.iteration = checkpoint["iteration"]
        
        return new

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Sets the seed for torch, cuda, and numpy.

        Args:
            seed (int): The seed

        Returns:
            None
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        return None


if __name__=="__main__":

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data.mnist', train=True, download=True, 
                         transform=transforms.ToTensor()),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data.mnist', train=False, transform=transforms.ToTensor()),
        batch_size=64, shuffle=True)

    layers = [
        "Linear-300-ReLU",
        "Linear-100-ReLU",
        "Linear-100"
    ]

    model = Model(layers, "SGD", {"lr":1}, device="auto",
                  input_shape=[None,1,28,28])

    model.train(train_loader, test_loader, epochs=2, 
                criteron="CrossEntropyLoss", criteron_kwargs={}, 
                log_folder="./test", log_freq=172, test_freq=500, 
                from_checkpoint=None,
                required_checkpoints=None, save_best_model="test_acc",
                patience=1000)

    model = Model.Load("test/Best_Model.pt", modif={"optimizer_kwargs":
                                                    {"lr":0.1}})
    model.train(train_loader, test_loader, epochs=2, 
                criteron="CrossEntropyLoss", criteron_kwargs={}, 
                log_folder="./test", log_freq=172, test_freq=500, 
                from_checkpoint=None,
                required_checkpoints=None, save_best_model="test_acc",
                patience=1000)







