import torch
from model.backbone.regnetX import RegNetX800mfBackbone
from model.neck.fpn import FPN 

backbone = RegNetX800mfBackbone()
fpn = FPN()

x = torch.randn(1, 3, 224, 224) 
features = backbone(x)
pyramid = fpn(features)

for name, fm in pyramid.items(): #fm is feature map where name is P2,P3,P4,P5
    print(f"{name}: {fm.shape}")