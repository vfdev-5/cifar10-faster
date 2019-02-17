
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
