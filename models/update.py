import torch
import torch.nn as nn
from .utils.utils import *

IN_DIM = 164 # 162 + 2

class CNN(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN, self).__init__()

        outputdim = input_dim
        # 164 = 81*2+2 = 9*9*2+2
        self.layer1 = nn.Sequential(nn.Conv2d(IN_DIM, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim_final = outputdim

        ### global motion
        self.layer10 = nn.Sequential(nn.Conv2d(input_dim, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)

        return x

class CNN16(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN16, self).__init__()

        outputdim = input_dim
        # 164 = 81*2+2 = 9*9*2+2
        self.layer1 = nn.Sequential(nn.Conv2d(IN_DIM, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), 
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim = input_dim
        # self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
        #                             nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        # input_dim = outputdim
        # outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim_final = outputdim

        ### global motion
        self.layer10 = nn.Sequential(nn.Conv2d(input_dim, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))


    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)

        return x

class GMA(nn.Module):
    def __init__(self, args, sz, in_dim = 164):
        super().__init__()
        self.args = args
        global IN_DIM
        IN_DIM = in_dim

        if sz==32:
            self.cnn = CNN(128)
        if sz==16:
            self.cnn = CNN16(128)
            
    def forward(self, corr, flow):      
        delta_flow = self.cnn(torch.cat((corr, flow), dim=1))
        return delta_flow



