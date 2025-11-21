import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_cha=(64, 128, 288, 672), out_cha=256): #make all layers output to 256 out_channels
        super().__init__()
        C2, C3, C4, C5 = in_cha
        self.lat2 = nn.Conv2d(C2, out_cha, 1) #1 is kernel size for lateral conv layers
        self.lat3 = nn.Conv2d(C3, out_cha, 1)
        self.lat4 = nn.Conv2d(C4, out_cha, 1)
        self.lat5 = nn.Conv2d(C5, out_cha, 1)

        self.s2 = nn.Conv2d(out_cha, out_cha, 3, padding=1) #3 is kernel size for smoothing conv layers
        self.s3 = nn.Conv2d(out_cha, out_cha, 3, padding=1)
        self.s4 = nn.Conv2d(out_cha, out_cha, 3, padding=1)
        self.s5 = nn.Conv2d(out_cha, out_cha, 3, padding=1)
        
    def forward(self, features):
        c2, c3, c4, c5 = features["C2"], features["C3"], features["C4"], features["C5"]
        l2, l3, l4, l5 = self.lat2(c2), self.lat3(c3), self.lat4(c4), self.lat5(c5)
        p5 = l5 # top layer
        p4 = l4 + F.interpolate(p5, size=l4.shape[2:], mode="nearest") #interpolate is upsample, size to match l4 of the [2: height,width]
        p3 = l3 + F.interpolate(p4, size=l3.shape[-2:], mode="nearest")
        p2 = l2 + F.interpolate(p3, size=l2.shape[-2:], mode="nearest")
        
        p2, p3, p4, p5 = self.s2(p2), self.s3(p3), self.s4(p4), self.s5(p5)
        return {"P2": p2, "P3": p3, "P4": p4, "P5": p5}
        