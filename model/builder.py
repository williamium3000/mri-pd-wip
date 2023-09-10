from .backbone.unet2d import UNet2D
from .backbone.unet3d import UNet3D
from .pix2pix.networks2d import define_G, define_D, NLayerDiscriminator3D, NLayerDiscriminator

def build_model(cfg):
    if cfg["model"]["name"] == "unet2d":
        return UNet2D(**cfg["model"]["kwargs"]), None
    elif cfg["model"]["name"] == "unet3d":
        return UNet3D(**cfg["model"]["kwargs"]), None
    elif cfg["model"]["name"] == "pix2pix":
        return define_G(**cfg["model"]["generator"]), define_D(**cfg["model"]["discriminator"])
    
    elif cfg["model"]["name"] == "my_pix2pix":
        g_name = cfg["model"]["generator"]["name"]
        d_name = cfg["model"]["discriminator"]["name"]
        if g_name == "unet2d":
            generator = UNet2D(**cfg["model"]["generator"]["kwargs"])
        elif g_name == "unet3d":
            generator = UNet3D(**cfg["model"]["generator"]["kwargs"])
        
        if d_name == "patch2d":
            discriminator = NLayerDiscriminator(**cfg["model"]["discriminator"]["kwargs"])
        elif d_name == "patch3d":
            discriminator = NLayerDiscriminator3D(**cfg["model"]["discriminator"]["kwargs"])

        return generator, discriminator
    