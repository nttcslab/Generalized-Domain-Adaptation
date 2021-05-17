import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from .switchable_norm import SwitchNorm1d, SwitchNorm2d

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = torch.nn.LeakyReLU()
        self.bn = SwitchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Dense_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()
        self.bn = SwitchNorm1d(out_features)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class SVHN_Extractor(nn.Module):
    def __init__(self):
        super(SVHN_Extractor, self).__init__()
        self.in1 = nn.InstanceNorm2d(3)
        self.conv1 = Conv_Block(3, 64, kernel_size=5)    
        self.conv2 = Conv_Block(64, 64, kernel_size=5)
        self.conv3 = Conv_Block(64, 128, kernel_size=3, stride=2)
        self.conv4 = Conv_Block(128, 128, kernel_size=3, stride=2)
        self.fc1 = Dense_Block(3200, 100)
        self.fc2 = Dense_Block(100, 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, SwitchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, SwitchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.in1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.dropout2d(x)
        x = x.view(x.size(0), -1)
        return x

    def conv_features(self, x):
        feats = []
        x = self.in1(x)
        x = self.conv1(x)
        feats.append(x)
        x = self.conv2(x)
        feats.append(x)
        return feats

class SVHN_Class_classifier(nn.Module):
    def __init__(self, num_classes):
        super(SVHN_Class_classifier, self).__init__()
        self.fc1 = Dense_Block(3200, 100)
        self.fc2 = Dense_Block(100, 100)
        self.fc3 = nn.Linear(100, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, SwitchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, SwitchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, constant = 1, reverse = False):
        x = self.fc1(x)
        x = F.dropout(x)
        x = self.fc2(x)
        if reverse == True:
            x = GradReverse.grad_reverse(x, constant)
        x = self.fc3(x)
        return x

class SVHN_Domain_classifier(nn.Module):

    def __init__(self, num_domains=2):
        super(SVHN_Domain_classifier, self).__init__()
        self.fc1 = Dense_Block(3200, 100)
        self.fc2 = Dense_Block(100, 100)
        self.fc3 = nn.Linear(100, num_domains)

    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        x = self.fc1(x)
        x = F.dropout(x)
        x = self.fc2(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return x