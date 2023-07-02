
import timm
import torch

def mobilenetv3_large(pretrained=False, output_stride=16, **kwargs):
    return timm.create_model("mobilenetv3_large_100", features_only=True, output_stride=output_stride, pretrained=pretrained, **kwargs)


def mobilenetv3_small(pretrained=False, output_stride=16, **kwargs):
    return timm.create_model("mobilenetv3_small_100", features_only=True, output_stride=output_stride, pretrained=pretrained, **kwargs)
