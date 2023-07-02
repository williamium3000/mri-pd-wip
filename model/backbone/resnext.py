
import timm
import torch

def resnext50_32x4d(pretrained=False, output_stride=16, out_indices=(1, 2, 3, 4), **kwargs):
    return timm.create_model("resnext50_32x4d", features_only=True, output_stride=output_stride, pretrained=pretrained, out_indices=out_indices, **kwargs)


def resnext101_32x4d(pretrained=False, output_stride=16, out_indices=(1, 2, 3, 4), **kwargs):
    return timm.create_model("resnext101_32x4d", features_only=True, output_stride=output_stride, pretrained=pretrained, out_indices=out_indices, **kwargs)

# model = resnext50_32x4d(pretrained=True)
# tensor = torch.rand(3, 3, 224, 224)
# print([p.shape for p in model(tensor)])