import torch
import torch.nn as nn 
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, identity_downsample=None):
        super().__init__()
        width = out_channels
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width) # batchNorm of the 1x1 conv products from the z1 = W1*x
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.down = identity_downsample
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.down is not None:
            identity = self.down(x)
        out += identity  # residual add the vanish gradient fix
        return self.relu(out)
    
class ResNetBlock(nn.Module):
    def __init__(self, layers = (3, 4, 6, 3), in_channels=3):
        super().__init__() # super is to call the parent class constructor
        self.inplanes = 64 # inplanes is the number of channels in the input image
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
        # layers 
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, out_channels, stride, downsample))
        self.inplanes = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x  = self.relu(self.bn1(self.conv1(x)))
        x  = self.maxpool(x)     # /4
        c2 = self.layer1(x)      # /4,  256 ch
        c3 = self.layer2(c2)     # /8,  512 ch
        c4 = self.layer3(c3)     # /16, 1024 ch
        c5 = self.layer4(c4)     # /32, 2048 ch
        return {"C2": c2, "C3": c3, "C4": c4, "C5": c5}

def ResNet50_backbone():  return ResNetBlock((3,4,6,3))