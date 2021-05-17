import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from models.Discriminator import Discriminator
from torchvision.models import AlexNet

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

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

def alexnet(num_classes, pretrained=True):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        print('Load pre trained model')
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.classifier[-1].weight, .1)
    nn.init.constant_(model.classifier[-1].bias, 0.)
    return model

def get_models(num_classes, num_domains, pretrained=True):
    base_model = alexnet(num_classes, pretrained=pretrained)
    discriminator = Discriminator([4096, 1024, 1024, num_domains], grl=True, reverse=True)
    feature_layers = nn.Sequential(*list(base_model.classifier.children())[:-1])
    fc = list(base_model.classifier.children())[-1]

    extractor = Alex_Extractor(base_model.features, feature_layers)
    domain_classifier = Alex_Domain_classifier(discriminator)
    class_classifier = Alex_Class_classifier(fc)

    return extractor, class_classifier, domain_classifier

class Alex_Extractor(nn.Module):
    def __init__(self, base_features, class_features):
        super(Alex_Extractor, self).__init__()
        self.base_features = base_features
        self.class_features = class_features

    def forward(self, x):
        x = self.base_features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.class_features(x)
        return x

    def conv_features(self, x):
        results = []
        for i, model in enumerate(self.base_features):
            x = model(x)
            if i in {4, 7}:
                results.append(x)
        return results

class Alex_Class_classifier(nn.Module):
    def __init__(self, classifier):
        super(Alex_Class_classifier, self).__init__()
        self.classifier = classifier

    def forward(self, x, T=1.0, reverse=False, constant=1.0):
        if reverse == True:
            x = grad_reverse(x, constant)
        x = self.classifier(x)
        return x

class Alex_Domain_classifier(nn.Module):
    def __init__(self, discriminator):
        super(Alex_Domain_classifier, self).__init__()
        self.discriminator = discriminator

    def forward(self, x, constant):
        x = self.discriminator(x, constant)
        return x

if __name__ == "__main__":
    extractor, discriminator, classifier = get_models(10, 2, True)
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