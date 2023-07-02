import torch
from torch import nn

class FCN3DHead(nn.Module):
    def __init__(self, 
                 channels,
                 num_classes,
                 in_channels=None,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 ):
        super(FCN3DHead, self).__init__()
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        conv_padding = (kernel_size // 2)
        self.in_channels = in_channels
        self.channels = channels
        
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                nn.Sequential(
                    nn.Conv3d(
                        _in_channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding),
                    nn.BatchNorm3d(self.channels),
                    nn.ReLU()
                )
            )

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = nn.Sequential(
                    nn.Conv3d(
                        self.in_channels + self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding),
                    nn.BatchNorm3d(self.channels),
                    nn.ReLU()
                )
        self.classifier = nn.Conv3d(self.channels, num_classes, 1, bias=True)

    def forward(self, feats):
        feature = self.convs(feats[-1])
        if self.concat_input:
            feature = self.conv_cat(torch.cat([feats[-1], feature], dim=1))
        out = self.classifier(feature)
        return out
