import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

def get_network():
    return PytorchCIFAR10Default

class PytorchCIFAR10Default(nn.Module):
    def __init__(self):
        super(PytorchCIFAR10Default, self).__init__()
        self.name = "pytorchcifar10default"
        self.save_d = "cifar10/pytorchcifar10default/"
        self.ftr_name = "fc1"
        self.output_name = "fc2"

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 5, 1)
        self.conv4_bn = nn.BatchNorm2d(256)

        self.dropout0 = nn.Dropout2d(0.25)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc0 = nn.Linear(4096, 1024)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)


        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = torch.flatten(x, 1)

        x = self.dropout0(x)
        x = self.fc0(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)


        output = F.log_softmax(x, dim=1)
        return output
