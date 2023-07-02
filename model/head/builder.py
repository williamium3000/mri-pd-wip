from .fcn3d import FCN3DHead

def build_head(cfg):
    head_name = cfg['decode_head']["name"]
    if head_name == "FCN3DHead":
        head = FCN3DHead(
            num_classes=cfg["nclass"],
            **cfg['decode_head']["kwargs"]
        )
    else:
        raise NotImplementedError(f"{head_name} not implemented!")
    return head