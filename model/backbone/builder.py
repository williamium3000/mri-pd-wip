
from .unet3d import UNet3D

def build_backbone(cfg):
    backbone_name = cfg['backbone']["name"]
    if backbone_name == "UNet3D":
        backbone = UNet3D(
            **cfg['backbone']["kwargs"]
        )
    else:
        raise NotImplementedError(f"{backbone_name} not implemented!")
    return backbone