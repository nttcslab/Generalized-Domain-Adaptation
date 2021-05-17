import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self, input_dim=1, out_dim=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # projection MLP
        self.l1 = nn.Linear(64, 64)
        self.fc_bn1 = nn.BatchNorm1d(64)
        self.l2 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, 2, 2)

        h = torch.mean(x, dim=[2, 3])

        x = self.l1(h)
        x = F.relu(x)
        x = self.fc_bn1(x)
        x = self.l2(x)

        return h, x
