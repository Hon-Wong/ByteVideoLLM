import torch.nn as nn
from .dynamic_spatial_pooling import DynamicSpatialPooling
from .projector import build_projector


class ConcatAdapter(nn.Module):
    
    def __init__(self, num_features=1408, **kwargs):
        super().__init__()
        self.hidden_size = num_features

    def forward(self, image_embeds, n_frames):
        feature_size = image_embeds.shape[-1]
        ret = [x.reshape(-1, feature_size) for x in image_embeds.split(n_frames)]
        return ret


class SepAdapter(nn.Module):
    
    def __init__(self, num_features=1408, **kwargs):
        super().__init__()
        self.hidden_size = num_features

    def forward(self, image_embeds, n_frames):
        ret = image_embeds.split(n_frames)
        return ret


adapter_factory = {
    "dynamic_spatial_pooling": DynamicSpatialPooling,
    "none": SepAdapter,
}

def build_adapter(**kwargs):
    adapter_type = kwargs.pop("type")
    assert adapter_type in adapter_factory, f"Invalid adapter type: {adapter_type}"
    return adapter_factory[adapter_type](**kwargs)
