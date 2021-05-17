import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AlexSimCLR(nn.Module):
    def __init__(self, out_dim):
        super(AlexSimCLR, self).__init__()
        self.alexnet = models.alexnet(pretrained=False, num_classes=out_dim)

    def forward(self, x):
        h = self.alexnet.features(x)
        h = self.alexnet.avgpool(h)
        h = torch.flatten(h, 1)

        x = self.alexnet.classifier(h)
        return h, x