import torch
import torch.nn as nn
        

class FastWideResNet(nn.Module):
    
    def __init__(self, width, depth, final_weight=1.0, dropout_p=None, **kwargs):
        super(FastWideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6 * n + 4"
        self.width = width
        self.depth = depth
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
