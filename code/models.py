
# Network from https://github.com/davidcpage/cifar10-fast
# Adapted to python < 3.6

import torch
import torch.nn as nn


################################################################################
#  Code below taken from : https://github.com/davidcpage/cifar10-fast.git
################################################################################

from collections import namedtuple, OrderedDict
import time
import torch
from torch import nn
import numpy as np


class Identity(nn.Module):
    def forward(self, x): return x


class Mul(nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x*self.weight


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))


class Add(nn.Module):
    def forward(self, x, y): return x + y


class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)


class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target


def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False,
               bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m

#####################
## dict utils
#####################


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)

#####################
## graph building
#####################

sep='_'
RelativePath = namedtuple('RelativePath', ('parts'))
rel_path = lambda *parts: RelativePath(parts)


def build_graph(net):
    net = OrderedDict(path_iter(net))
    default_inputs = [[('input',)]]+[[k] for k in net.keys()]
    with_default_inputs = lambda vals: (val if isinstance(val, tuple) else (val, default_inputs[idx]) for idx,val in enumerate(vals))
    parts = lambda path, pfx: tuple(pfx) + path.parts if isinstance(path, RelativePath) else (path,) if isinstance(path, str) else path
    return OrderedDict([
        (sep.join((*pfx, name)), (val, [sep.join(parts(x, pfx)) for x in inputs])) for (*pfx, name), (val, inputs) in zip(net.keys(), with_default_inputs(net.values()))
    ])


class TorchGraph(nn.Module):
    def __init__(self, net):
        self.graph = build_graph(net)
        super().__init__()
        for n, (v, _) in self.graph.items():
            setattr(self, n, v)

    def forward(self, inputs):
        self.cache = OrderedDict(inputs)
        for n, (_, i) in self.graph.items():
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache

    def half(self):
        for module in self.children():
            if type(module) is not nn.BatchNorm2d:
                module.half()
        return self


def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return OrderedDict([
        ('conv', nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)),
        ('bn', batch_norm(c_out, bn_weight_init=bn_weight_init, **kw)),
        ('relu', nn.ReLU(True))
    ])

def residual(c, **kw):
    return OrderedDict([
        ('in', Identity()),
        ('res1', conv_bn(c, c, **kw)),
        ('res2', conv_bn(c, c, **kw)),
        ('add', (Add(), [rel_path('in'), rel_path('res2', 'relu')])),
    ])

def basic_net(channels, weight,  pool, **kw):
    return OrderedDict([
        ('prep', conv_bn(3, channels['prep'], **kw)),
        ('layer1', OrderedDict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool)),
        ('layer2', OrderedDict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool)),
        ('layer3', OrderedDict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool)),
        ('classifier', OrderedDict([
            ('pool', nn.MaxPool2d(4)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(channels['layer3'], 10, bias=False)),
            ('logits', Mul(weight)),
        ]))
    ])


def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2),
        extra_layers=(),
        res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)
    return n


class OriginalFastResnet(TorchGraph):

    def __init__(self):
        super(OriginalFastResnet, self).__init__(net())
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.graph.update({
            'loss': (self.loss, ['classifier_logits', 'target'])
        })

################################################################################
#  END of Code below taken from : https://github.com/davidcpage/cifar10-fast.git
################################################################################


def seq_conv_bn(in_channels, out_channels, conv_kwargs, bn_kwargs):
    if "padding" not in conv_kwargs:
        conv_kwargs["padding"] = 1
    if "stride" not in conv_kwargs:
        conv_kwargs["stride"] = 1
    if "bias" not in conv_kwargs:
        conv_kwargs["bias"] = False
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, **conv_kwargs),
        batch_norm(out_channels, **bn_kwargs),
        nn.ReLU(inplace=True)
    )


def conv_bn_elu(in_channels, out_channels, conv_kwargs, bn_kwargs, alpha=1.0):
    if "padding" not in conv_kwargs:
        conv_kwargs["padding"] = 1
    if "stride" not in conv_kwargs:
        conv_kwargs["stride"] = 1
    if "bias" not in conv_kwargs:
        conv_kwargs["bias"] = False
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, **conv_kwargs),
        batch_norm(out_channels, **bn_kwargs),
        nn.ELU(alpha=alpha, inplace=True)
    )


class FastResnet(nn.Module):

    def __init__(self, conv_kwargs=None, bn_kwargs=None,
                 conv_bn_fn=seq_conv_bn,
                 final_weight=0.125):
        super(FastResnet, self).__init__()

        conv_kwargs = {} if conv_kwargs is None else conv_kwargs
        bn_kwargs = {} if bn_kwargs is None else bn_kwargs

        self.prep = conv_bn_fn(3, 64, conv_kwargs, bn_kwargs)

        self.layer1 = nn.Sequential(
            conv_bn_fn(64, 128, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2),
            IdentityResidualBlock(128, 128, conv_kwargs, bn_kwargs, conv_bn_fn=conv_bn_fn)
        )

        self.layer2 = nn.Sequential(
            conv_bn_fn(128, 256, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            conv_bn_fn(256, 512, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2),
            IdentityResidualBlock(512, 512, conv_kwargs, bn_kwargs, conv_bn_fn=conv_bn_fn)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.Linear(512, 10, bias=False)
        )

        if final_weight == "auto":
            self.final_weight = torch.nn.Parameter(torch.Tensor([0.125]))
        else:
            self.final_weight = final_weight

    def forward(self, x):

        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        x = x * self.final_weight
        return x


class FastResRecNet(nn.Module):

    def __init__(self, conv_kwargs=None, bn_kwargs=None,
                 conv_bn_fn=seq_conv_bn,
                 final_weight=0.125):
        super(FastResRecNet, self).__init__()

        conv_kwargs = {} if conv_kwargs is None else conv_kwargs
        bn_kwargs = {} if bn_kwargs is None else bn_kwargs

        self.prep = conv_bn_fn(3, 64, conv_kwargs, bn_kwargs)

        self.layer1 = nn.Sequential(
            conv_bn_fn(64, 128, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2),
            IdentityResidualRecurrentBlock(128, conv_kwargs, bn_kwargs, conv_bn_fn=conv_bn_fn),
        )

        self.layer2 = nn.Sequential(
            conv_bn_fn(128, 256, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2),
            IdentityResidualRecurrentBlock(256, conv_kwargs, bn_kwargs, conv_bn_fn=conv_bn_fn),
        )

        self.layer3 = nn.Sequential(
            conv_bn_fn(256, 512, conv_kwargs, bn_kwargs),
            nn.MaxPool2d(kernel_size=2),
            IdentityResidualRecurrentBlock(512, conv_kwargs, bn_kwargs, conv_bn_fn=conv_bn_fn),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.Linear(512, 10, bias=False)
        )

        if final_weight == "auto":
            self.final_weight = torch.nn.Parameter(torch.Tensor([0.125]))
        else:
            self.final_weight = final_weight

    def forward(self, x):

        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        x = x * self.final_weight
        return x
        

class FastWideResNet(nn.Module):
    
    def __init__(self, final_weight=1.0, dropout_p=None, **kwargs):
        super(FastWideResNet, self).__init__()
        self.width = 4
        self.depth = 10
        self.final_weight = final_weight
        self.dropout_p = dropout_p

        n = (self.depth - 4) // 6
        widths = [int(v * self.width) for v in (16, 32, 64)]

        self.prep = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        
        self.group0 = self._make_group(16, widths[0], n=n, stride=1)
        self.group1 = self._make_group(widths[0], widths[1], n=n, stride=2)
        self.group2 = self._make_group(widths[1], widths[2], n=n, stride=2)
        
        self.bn = nn.BatchNorm2d(widths[2])
        self.relu = nn.ReLU(True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(widths[2], 10)        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_group(self, inplanes, outplanes, n, stride):
        downsample = None
        if stride != 1 or inplanes != outplanes:
            downsample = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)

        layers = [FastWRNBlock(inplanes, outplanes, stride=stride, downsample=downsample, dropout_p=self.dropout_p)]
        for i in range(1, n):
            layers.append(FastWRNBlock(outplanes, outplanes, stride=1, dropout_p=self.dropout_p))

        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.prep(x)
        x = self.group0(x)
        x = self.group1(x)
        x = self.group2(x)                
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y * self.final_weight


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, conv_kwargs, bn_kwargs,
                 conv_bn_fn=seq_conv_bn):
        super(IdentityResidualBlock, self).__init__()
        self.conv1 = conv_bn_fn(in_channels, out_channels, conv_kwargs, bn_kwargs)
        self.conv2 = conv_bn_fn(out_channels, out_channels, conv_kwargs, bn_kwargs)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class IdentityResidualRecurrentBlock(nn.Module):

    def __init__(self, n_channels, conv_kwargs, bn_kwargs,
                 conv_bn_fn=seq_conv_bn):
        super(IdentityResidualRecurrentBlock, self).__init__()
        self.conv_rec = conv_bn_fn(n_channels, n_channels, conv_kwargs, bn_kwargs)        

    def forward(self, x):
        residual = x
        x = self.conv_rec(x)
        x = self.conv_rec(x)
        return x + residual


class FastWRNBlock(nn.Module):
    
    def __init__(self, inplanes, outplanes, stride=1, downsample=None, dropout_p=None):
        super(FastWRNBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu2 = nn.ReLU(True)        
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.downsample = downsample

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p is not None else None
    
    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        x = self.conv1(x)

        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        
        y = x + residual
        return y


if __name__ == "__main__":

    import torch

    torch.manual_seed(12)

    model = FastResnet(bn_kwargs={"bn_weight_init": 1.0})

    x = torch.rand(4, 3, 32, 32)
    y = model(x)
    print(y.shape)
