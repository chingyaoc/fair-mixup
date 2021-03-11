import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ResNet18_Encoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.resnet.avgpool = Identity()

    def forward(self, x):
        outputs = self.resnet(x)
        return outputs.view(-1, 512, 8, 8)

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.avg(x).view(-1, 512)
        x = self.fc1(x)
        x = self.relu(x)
        outputs = self.fc2(x)
        return torch.sigmoid(outputs)
