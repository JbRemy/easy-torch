import sys
sys.path.append("../")

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn

from easy_torch.model import Model
from easy_torch.callbacks.metrics import Acc, TestAcc


# Definition initialisation parameters of the network
layers = ["Linear-300", "Linear-100", "Linear-10"]
device = "auto"
seed = 42

# Definition of optimization parameters
optimizer = "SGD"
optimizer_kwargs = {"lr": 0.1}
criterion = "CrossEntropyLoss"
criterion_kwargs = {}

# Training parameters
train_loader = DataLoader(
    datasets.MNIST('./data/mnist', train=True, download=True, 
                   transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('./data/mnist', train=False, 
                   transform=transforms.ToTensor()),
    batch_size=64, shuffle=False
)

epochs = 10
log_folder = "./lenet_results"
log_freq = 20000
test_freq = 60000

callbacks = [
    Acc("epoch_end", 60000),
    TestAcc(10000)
]

# Definition and training of the network
lenet = Model(layers=layers, device=device, seed=seed)
lenet.compile(optimizer, criterion, optimizer_kwargs, criterion_kwargs)
lenet.train(train_loader, epochs, log_folder, log_freq, test_loader, test_freq,
            callbacks) 



