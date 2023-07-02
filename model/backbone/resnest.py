
import timm
import torch

def resnest50d(pretrained=False, output_stride=16, out_indices=(1, 2, 3, 4), **kwargs):
    return timm.create_model("resnest50d", features_only=True, output_stride=output_stride, pretrained=pretrained, out_indices=out_indices, **kwargs)


def resnest101e(pretrained=False, output_stride=16, out_indices=(1, 2, 3, 4), **kwargs):
    return timm.create_model("resnest101e", features_only=True, output_stride=output_stride, pretrained=pretrained, out_indices=out_indices, **kwargs)

# model = resnest50d(pretrained=True)
# tensor = torch.rand(3, 3, 224, 224)
# print([p.shape for p in model(tensor)])