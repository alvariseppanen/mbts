from models.common.backbones.image_encoder import ImageEncoder
from models.common.backbones.monodepth2 import Monodepth2
from models.common.backbones.spatial_encoder import SpatialEncoder

from models.common.backbones.monoscene_modules.monoscene import MonoScene
from models.common.backbones.semantic import SemanticSegmentor


def make_backbone(conf, **kwargs):
    enc_type = conf.get("type", "monodepth2")  # monodepth2 | spatial | global | volumetric | semantic
    if enc_type == "monodepth2":
        net = Monodepth2.from_conf(conf, **kwargs)
    elif enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    elif enc_type == "volumetric":
        net = MonoScene.from_conf(conf, **kwargs)
    elif enc_type == "semantic":
        net = SemanticSegmentor.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported encoder type: {enc_type}")
    return net
