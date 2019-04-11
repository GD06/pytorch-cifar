'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable 


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class SubVGG(nn.Module):
    def __init__(self, vgg_name):
        super(SubVGG, self).__init__()

        self._layers = self._make_layers(cfg[vgg_name])
        self.layer1 = nn.Sequential(*(self._layers[:5]))
        self.layer2 = nn.Sequential(*(self._layers[5:10]))
        self.layer3 = nn.Sequential(*(self._layers[10:15]))
        self.layer4 = nn.Sequential(*(self._layers[15:]))
        self.classifier = nn.Linear(512, 10)

        self._inter_var = []

    def forward(self, x):
        self._inter_var.clear() 
        self._inter_var.append(Variable(x.data, requires_grad=True))
        out = self.layer1(x)

        self._inter_var.append(Variable(out.data, requires_grad=True))
        out = self.layer2(out)

        self._inter_var.append(Variable(out.data, requires_grad=True))
        out = self.layer3(out)

        self._inter_var.append(Variable(out.data, requires_grad=True))
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        self._inter_var.append(Variable(out.data, requires_grad=True))
        return out

    def sub_backward(self, targets):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(self._inter_var[-1], targets)
        loss.backward()

        back_var = self.layer4(self._inter_var[-2])
        back_var = back_var.view(back_var.size(0), -1)
        back_var = self.classifier(back_var)
        back_var.backward(self._inter_var[-1].grad)

        back_var = self.layer3(self._inter_var[-3])
        back_var.backward(self._inter_var[-2].grad)

        back_var = self.layer2(self._inter_var[-4])
        back_var.backward(self._inter_var[-3].grad)

        back_var = self.layer1(self._inter_var[-5])
        back_var.backward(self._inter_var[-4].grad)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=False)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return layers 


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print((y.size()))

# test()
