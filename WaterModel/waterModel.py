import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class WaterModel(nn.Module):
    def __init__(self):
        super().__init__()
        effnetB3 = EfficientNet.from_pretrained("efficientnet-b3")
        layers = list(effnetB3.children())
        layers += [nn.AdaptiveAvgPool2D(), nn.Flatten()]
        layers += [nn.Dropout(p = 0.5)]
        layers += [nn.BatchNorm1d()]
        layers += [nn.Linear()]