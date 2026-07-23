"""Model components required by GO-Renderer 5B and Wan2.1 14B."""

from transformers import AutoTokenizer

from .multiref_transformer3d import (
    CroodRefTransformer3DModel,
    CroodRefTransformer3DModel2_2,
)
from .wan_image_encoder import CLIPModel
from .wan_text_encoder import WanT5EncoderModel
from .wan_transformer3d import (
    Wan2_2Transformer3DModel,
    WanRMSNorm,
    WanSelfAttention,
    WanTransformer3DModel,
)
from .wan_vae import AutoencoderKLWan
from .wan_vae3_8 import AutoencoderKLWan3_8

__all__ = [
    "AutoTokenizer",
    "AutoencoderKLWan",
    "AutoencoderKLWan3_8",
    "CLIPModel",
    "CroodRefTransformer3DModel",
    "CroodRefTransformer3DModel2_2",
    "Wan2_2Transformer3DModel",
    "WanRMSNorm",
    "WanSelfAttention",
    "WanT5EncoderModel",
    "WanTransformer3DModel",
]
