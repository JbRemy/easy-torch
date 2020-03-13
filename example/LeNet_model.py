import sys
sys.path.append("../")

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

from easy_torch.model import Model
from easy_torch.callbacks.metrics import Acc, TestAcc


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Definition initialisation parameters of the network
model = LeNet()
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
log_folder = "./lenet_model_results"
log_freq = 20000
test_freq = 60000

callbacks = [
    Acc("epoch_end", 60000),
    TestAcc(10000)
]

# Definition and training of the network
lenet = Model(model=model, device=device, seed=seed)
lenet.compile(optimizer, criterion, optimizer_kwargs, criterion_kwargs)
lenet.train(train_loader, epochs, log_folder, log_freq, test_loader, test_freq,
            callbacks) 



