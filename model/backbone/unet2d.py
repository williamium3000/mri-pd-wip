import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F


class UNet2D(nn.Module):
    def __init__(self,
                 in_channels,
                 base_channels=64,
                 bilinear=True,
                 nearest=False,
                 add_residual=True):
        super(UNet2D, self).__init__()
        self.add_residual = add_residual
        self.two_convs = twoConvs(in_channels, base_channels, base_channels)
        self.down1 = downSample(base_channels, base_channels * 2)
        self.down2 = downSample(base_channels * 2, base_channels * 4)
        self.down3 = downSample(base_channels * 4, base_channels * 8)
        factor = 2 if (bilinear or nearest) else 1
        self.down4 = downSample(base_channels * 8, (base_channels * 16) // factor)
        self.up1 = upSample(base_channels * 16, (base_channels * 8 )// factor, bilinear, nearest)
        self.up2 = upSample(base_channels * 8, (base_channels * 4) // factor, bilinear, nearest)
        self.up3 = upSample(base_channels * 4, (base_channels * 2) // factor, bilinear, nearest)
        self.up4 = upSample(base_channels * 2, in_channels, bilinear, nearest)
        
    def forward(self, x):
        self._check_input_divisible(x)
        x1 = self.two_convs(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out1 = self.up1(x5, x4)
        out2 = self.up2(out1, x3)
        out3 = self.up3(out2, x2)
        out4 = self.up4(out3, x1)
        if self.add_residual:
            return out4 + x
        else:
            return out4
        
    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, 5):
            whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}'


class twoConvs(nn.Module): 
    def __init__(self, in_channels, out_channels, inter_channels):
        super().__init__()
        self.two_convs =  nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = inter_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = inter_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )
    def forward(self, x):
        return self.two_convs(x)

class downSample(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size = 2)
        self.two_convs = twoConvs(in_channels, out_channels, out_channels)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.two_convs(x)
        return x

class upSample(nn.Module): 
    def __init__(self, in_channels, out_channels, bilinear, nearest):
        super().__init__()  
        assert not (nearest and bilinear)
        if bilinear or nearest:
            self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2) if bilinear else \
                nn.UpsamplingNearest2d(scale_factor=2)
            self.two_convs = twoConvs(in_channels, out_channels, in_channels // 2)
        else:
            self.up_sample = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.two_convs = twoConvs(in_channels, out_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up_sample(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        cat = torch.cat([x1, x2], dim = 1)
        x = self.two_convs(cat)
        return x