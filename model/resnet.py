import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock2D(nn.Module):
    def __init__(self, channels,kernel_size, padding, dropout, stride=1, dilation=1):

        super(BasicBlock2D, self).__init__()

        padding = padding * dilation

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding,dilation)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding,dilation)
        self.bn2 = nn.InstanceNorm2d(channels, affine=True)
        self.elu2 = nn.ELU()

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)

        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.elu2(out)

        return out

class BasicBlock1D(nn.Module):
    def __init__(self, angle_channel):

        super(BasicBlock1D, self).__init__()
        self.angle_channel = angle_channel
        self.conv1 = nn.Sequential(
                        nn.Linear(self.angle_channel, self.angle_channel),
                        nn.ReLU()
                    )
        self.conv2 = nn.Sequential(
                        nn.Linear(self.angle_channel, self.angle_channel),
                        nn.ReLU()
                    )
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, channels, num_blocks, dropout, dilation_list, kernel_size, padding):

        super(ResNet, self).__init__()

        dilations = dilation_list

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.elu = nn.ELU()

        self.residual_blocks = nn.ModuleList([])
        self.residual_blocks.append(BasicBlock2D(channels,kernel_size = 5, padding = 2, dilation =1, dropout=dropout))
        self.residual_blocks.append(BasicBlock2D(channels,kernel_size = 5, padding = 2, dilation =2, dropout=dropout))
        self.residual_blocks.append(BasicBlock2D(channels,kernel_size = 5, padding = 2, dilation =4, dropout=dropout))

        d_idx = 0
        if num_blocks > 4:
            for i in range(num_blocks - 4):
                self.residual_blocks.append(BasicBlock2D(channels,kernel_size, padding, dilation =dilations[d_idx], dropout=dropout))
                d_idx = (d_idx + 1) % len(dilations)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        
        for layer in self.residual_blocks:
            x = layer(x)

        return x