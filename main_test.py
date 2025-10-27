import torch
from model.backbone.resnet_backbone import ResNetBlock
from model.neck.fpn import FPN 

backbone = ResNetBlock()
fpn = FPN()

x = torch.randn(1, 3, 224, 224)  # Example input tensor 1= batch size, 3=channels, 224x224=image size
features = backbone(x)
pyramid = fpn(features)

for name, fm in pyramid.items(): #fm is feature map where name is P2,P3,P4,P5
    print(f"{name}: {fm.shape}")