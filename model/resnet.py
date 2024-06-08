'''ResNet18/34/50/101/152 in PyTorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, n_channels, n_classes=10, cifar_stem=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if cifar_stem:
            self.stem = nn.Sequential()
            self.stem.add_module('conv0', nn.Conv2d(n_channels, self.in_planes, kernel_size=3, stride=1, padding=1,
                                                    bias=False))
            self.stem.add_module('BN1', nn.BatchNorm2d(self.in_planes))
            self.stem.add_module('ReLU1', nn.ReLU(inplace=True))
        else:  # e.g. ImageNet
            self.stem = nn.Sequential()
            self.stem.add_module('conv0', nn.Conv2d(n_channels, self.in_planes, kernel_size=7, stride=2, padding=3,
                                                    bias=False))
            self.stem.add_module('BN1', nn.BatchNorm2d(self.in_planes))
            self.stem.add_module('ReLU1', nn.ReLU(inplace=True))
            self.stem.add_module('MaxPool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, n_classes)

        self.embdim = 512 * block.expansion

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        emb = torch.flatten(out, 1)
        out = self.linear(emb)
        return out, emb

    def get_embedding_dim(self):
        return self.embdim


def ResNet18(n_channels, n_classes=10, cifar_stem=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], n_channels, n_classes, cifar_stem)


def ResNet34(n_channels, n_classes=10, cifar_stem=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], n_channels, n_classes, cifar_stem)


def ResNet50(n_channels, n_classes=10, cifar_stem=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], n_channels, n_classes, cifar_stem)


def ResNet101(n_channels, n_classes=10, cifar_stem=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], n_channels, n_classes, cifar_stem)


def ResNet152(n_channels, n_classes=10, cifar_stem=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], n_channels, n_classes, cifar_stem)
