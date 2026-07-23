"""Small media and configuration helpers used by GO-Renderer entry points."""

from __future__ import annotations

import inspect
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image


def filter_kwargs(cls, kwargs):
    """Keep only keyword arguments accepted by ``cls.__init__``."""

    signature = inspect.signature(cls.__init__)
    valid_parameters = set(signature.parameters) - {"self", "cls"}
    return {key: value for key, value in kwargs.items() if key in valid_parameters}


def resolve_model_source(model_name: str) -> str:
    """Use a local model directory or download a Hugging Face snapshot."""

    model_path = Path(model_name).expanduser()
    if model_path.is_dir():
        return str(model_path.resolve())

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_name)


def save_videos_grid(
    videos: torch.Tensor,
    path: str,
    *,
    rescale: bool = False,
    n_rows: int = 6,
    fps: int = 12,
) -> None:
    """Write a ``[B, C, T, H, W]`` tensor as an MP4 or GIF grid."""

    frames = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for frame in frames:
        frame = torchvision.utils.make_grid(frame, nrow=n_rows)
        frame = frame.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            frame = (frame + 1.0) / 2.0
        frame = (frame.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
        outputs.append(Image.fromarray(frame))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".mp4":
        imageio.mimsave(output_path, outputs, fps=fps)
    else:
        imageio.mimsave(output_path, outputs, duration=1000 / fps)


def padding_image(image: Image.Image, new_width: int, new_height: int) -> Image.Image:
    """Fit an image into a white canvas while preserving its aspect ratio."""

    canvas = Image.new("RGB", (new_width, new_height), (255, 255, 255))
    aspect_ratio = image.width / image.height
    target_ratio = new_width / new_height
    if aspect_ratio > target_ratio:
        resized_width = new_width
        resized_height = int(resized_width / aspect_ratio)
    else:
        resized_height = new_height
        resized_width = int(resized_height * aspect_ratio)
    resized = image.resize((resized_width, resized_height))
    canvas.paste(
        resized,
        ((new_width - resized_width) // 2, (new_height - resized_height) // 2),
    )
    return canvas


def get_image_latent(ref_image=None, sample_size=None, padding=False):
    """Convert one validation image into a ``[1, C, 1, H, W]`` tensor."""

    if ref_image is None:
        return None
    if isinstance(ref_image, str):
        ref_image = Image.open(ref_image).convert("RGB")
        if padding:
            ref_image = padding_image(ref_image, sample_size[1], sample_size[0])
        ref_image = ref_image.resize((sample_size[1], sample_size[0]))
    tensor = torch.from_numpy(np.asarray(ref_image).copy())
    return tensor.unsqueeze(0).permute(3, 0, 1, 2).unsqueeze(0) / 255
