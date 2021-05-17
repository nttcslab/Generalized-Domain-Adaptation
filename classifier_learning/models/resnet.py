from torchvision.models import resnet50
from models.Discriminator import Discriminator
import torch
import torch.nn as nn
import torch.nn.init as init

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, reverse=True):
        ctx.lambd = lambd
        ctx.reverse=reverse
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            return (grad_output * -ctx.lambd), None, None
        else:
            return (grad_output * ctx.lambd), None, None

def grad_reverse(x, lambd=1.0, reverse=True):
    return GradReverse.apply(x, lambd, reverse)

def resnet(num_classes, pretrained=True):
    model = resnet50(pretrained=pretrained)
    return model

def get_models(num_classes, num_domains, pretrained=True):
    base_model = resnet(num_classes, pretrained=pretrained)
    discriminator = Discriminator([256, 1024, 1024, num_domains], grl=True, reverse=True)
    classifier = nn.Linear(256, num_classes)
    nn.init.xavier_uniform_(classifier.weight, .1)
    nn.init.constant_(classifier.bias, 0.)  

    extractor = Resnet_Extractor(base_model)
    domain_classifier = Resnet_Domain_classifier(discriminator)
    class_classifier = Resnet_Class_classifier(classifier)

    return extractor, class_classifier, domain_classifier

class Resnet_Extractor(nn.Module):
    def __init__(self, base_model):
        super(Resnet_Extractor, self).__init__()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.bottleneck = nn.Linear(2048, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x

    def conv_features(self, x) :
        results = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # results.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        results.append(x)
        x = self.layer2(x)
        results.append(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # results.append(x)
        return results

class Resnet_Class_classifier(nn.Module):
    def __init__(self, classifier):
        super(Resnet_Class_classifier, self).__init__()
        self.classifier = classifier

    def forward(self, x, T=1.0, reverse=False, constant=1.0):
        if reverse == True:
            x = grad_reverse(x, constant)
        x = self.classifier(x)
        return x

class Resnet_Domain_classifier(nn.Module):
    def __init__(self, discriminator):
        super(Resnet_Domain_classifier, self).__init__()
        self.discriminator = discriminator

    def forward(self, x, constant):
        x = self.discriminator(x, constant)
        return x

if __name__ == "__main__":
    extractor, classifier, discriminator = get_models(63, 2, True)
    extractor.eval()
    discriminator.eval()
    classifier.eval()

    with torch.no_grad():
        samples = torch.zeros(8, 3, 224, 224)
        print(samples.size())
        feature = extractor(samples)
        print(feature.size())
        dis_result = discriminator(feature, 1.0)
        print(dis_result.size())
        cls_result = classifier(feature)
        print(cls_result.size())