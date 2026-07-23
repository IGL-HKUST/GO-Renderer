#!/usr/bin/env python
"""Batch inference for the GO-Renderer Wan2.1 14B model."""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from go_renderer.data.validation import (
    letterbox_image,
    load_validation_manifest,
    resolve_background_path,
    validate_generation_shape,
    validate_rope_gap,
    validate_validation_media,
)
from go_renderer.models import AutoTokenizer, AutoencoderKLWan, CLIPModel, WanT5EncoderModel
from go_renderer.models.multiref_transformer3d import CroodRefTransformer3DModel
from go_renderer.pipeline import WanFunCroodRefPipeline
from go_renderer.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from go_renderer.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from go_renderer.utils.utils import filter_kwargs, resolve_model_source, save_videos_grid


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="config/wan2.1/wan_civitai.yaml")
    parser.add_argument(
        "--model_name",
        default="models/Diffusion_Transformer/GO-Renderer-14B",
        help="Complete GO-Renderer 14B model directory containing the Transformer, T5, VAE, tokenizer, and CLIP encoder",
    )
    parser.add_argument("--validation_json", required=True)
    parser.add_argument("--validation_samples", type=int, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_ref_frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/go_renderer_14b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weight_dtype",
        default="bfloat16",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument(
        "--sampler_name",
        default="Flow_Unipc",
        choices=("Flow", "Flow_Unipc", "Flow_DPM++"),
    )
    parser.add_argument(
        "--gpu_memory_mode",
        default="model_full_load",
        choices=("model_full_load", "model_cpu_offload", "sequential_cpu_offload"),
    )
    parser.add_argument(
        "--rope_gap",
        type=int,
        default=None,
        help="Optional override; by default use the complete model's Transformer config, then fall back to 5",
    )
    args = parser.parse_args()
    if args.validation_samples is not None and args.validation_samples < 1:
        parser.error("--validation_samples must be positive")
    if not 3 <= args.num_ref_frames <= 8:
        parser.error("--num_ref_frames must be in the inclusive range [3, 8]")
    return args


def _model_rope_gap(model_path, transformer_subpath, fallback):
    config_path = os.path.join(model_path, transformer_subpath, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            return int(json.load(file).get("rope_gap", fallback))
    return fallback


def _media_files(path):
    if os.path.isdir(path):
        return sorted(
            file
            for file in glob.glob(os.path.join(path, "*"))
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS
        )
    return [path]


def _load_frames(path, height, width, pad=False):
    files = _media_files(path)
    if not files:
        raise ValueError(f"No images found in directory: {path}")

    frames = []
    if len(files) > 1 or Path(files[0]).suffix.lower() in IMAGE_EXTENSIONS:
        for image_path in files:
            frames.append(Image.open(image_path).convert("RGB"))
    else:
        video = VideoReader(files[0])
        if len(video) == 0:
            raise ValueError(f"Video contains no frames: {path}")
        frames.extend(Image.fromarray(video[index].asnumpy()).convert("RGB") for index in range(len(video)))

    arrays = []
    for frame in frames:
        if pad:
            frame = letterbox_image(frame, width, height)
        else:
            frame = frame.resize((width, height), Image.Resampling.BILINEAR)
        arrays.append(np.asarray(frame, dtype=np.uint8))
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255.0
    return tensor.unsqueeze(0)  # [1, F, C, H, W]


def _sample_reference_frames(reference, count, rng):
    total_frames = reference.shape[2]
    if total_frames < count:
        raise ValueError(f"Reference has {total_frames} frames, but {count} are required")
    if total_frames == count:
        indices = list(range(total_frames))
    else:
        indices = [0]
        boundaries = np.linspace(0, total_frames - 1, count, dtype=int)
        for index in range(1, count):
            start = max(boundaries[index - 1] + 1, 1)
            end = max(start, boundaries[index])
            indices.append(int(rng.integers(start, end + 1)))
    return reference[:, :, indices], indices


def _match_video(video, frames, height, width):
    return F.interpolate(video, size=(frames, height, width), mode="nearest")


def _build_pipeline(args, config, device, weight_dtype):
    transformer_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    transformer_subpath = config["transformer_additional_kwargs"].get(
        "transformer_low_noise_model_subpath",
        config["transformer_additional_kwargs"].get("transformer_subpath", "."),
    )
    rope_gap = args.rope_gap
    if rope_gap is None:
        rope_gap = _model_rope_gap(args.model_name, transformer_subpath, fallback=5)
    rope_gap = validate_rope_gap(rope_gap, args.num_ref_frames)
    transformer_kwargs["rope_gap"] = rope_gap
    transformer = CroodRefTransformer3DModel.from_pretrained(
        os.path.join(args.model_name, transformer_subpath),
        transformer_additional_kwargs=transformer_kwargs,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        torch_dtype=weight_dtype,
    ).eval()
    image_encoder_path = os.path.join(
        args.model_name,
        config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"),
    )
    clip_image_encoder = (
        CLIPModel.from_pretrained(image_encoder_path).to(device, dtype=weight_dtype).eval()
        if os.path.exists(image_encoder_path)
        else None
    )
    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    scheduler_config = OmegaConf.to_container(config["scheduler_kwargs"])
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        scheduler_config["shift"] = 1
    scheduler = scheduler_cls(**filter_kwargs(scheduler_cls, scheduler_config))
    pipeline = WanFunCroodRefPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
    if args.gpu_memory_mode == "sequential_cpu_offload":
        from go_renderer.utils.fp8_optimization import replace_parameters_by_name

        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device)
    return pipeline, vae, rope_gap


def main():
    args = parse_args()
    args.model_name = resolve_model_source(args.model_name)
    validate_generation_shape(args.height, args.width, args.num_frames, spatial_multiple=16)
    manifest = load_validation_manifest(args.validation_json)
    if args.validation_samples is not None:
        manifest = manifest[: args.validation_samples]
    manifest = validate_validation_media(
        manifest,
        args.validation_json,
        num_ref_frames=args.num_ref_frames,
        num_frames=args.num_frames,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    weight_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.weight_dtype]
    config = OmegaConf.load(args.config_path)
    pipeline, vae, rope_gap = _build_pipeline(args, config, device, weight_dtype)
    for index, sample in enumerate(tqdm(manifest, desc="GO-Renderer 14B inference")):
        ref_path = sample["ref"]
        ref_coordmap_path = sample["ref_coordmap"]
        target_coordmap_path = sample["fg_coordmap"]
        bg_path = resolve_background_path(sample)
        gt_path = sample.get("gt", sample.get("video_path"))
        firstframe_path = sample.get("firstframe")

        reference = _load_frames(ref_path, args.height, args.width, pad=True).permute(0, 2, 1, 3, 4)
        ref_coordmap = _load_frames(ref_coordmap_path, args.height, args.width)
        if reference.shape[2] != ref_coordmap.shape[1]:
            raise ValueError(
                f"Sample {index}: ref has {reference.shape[2]} frames but ref_coordmap has "
                f"{ref_coordmap.shape[1]}"
            )
        sample_seed = int(sample.get("seed", args.seed))
        rng = np.random.default_rng(sample_seed)
        reference, ref_indices = _sample_reference_frames(reference, args.num_ref_frames, rng)
        ref_coordmap = ref_coordmap[:, ref_indices]

        target_coordmap = _load_frames(target_coordmap_path, args.height, args.width)
        background = _load_frames(bg_path, args.height, args.width) if bg_path else None
        if background is not None and background.shape[1] > args.num_frames:
            background = background[:, : args.num_frames]
        ground_truth = _load_frames(gt_path, args.height, args.width) if gt_path else None
        start_image = None
        if firstframe_path:
            start_image = _load_frames(firstframe_path, args.height, args.width, pad=True)[:, :1]
            start_image = start_image.permute(0, 2, 1, 3, 4)

        generator = torch.Generator(device=device).manual_seed(sample_seed)
        with torch.no_grad():
            generated = pipeline(
                sample["text"],
                num_frames=args.num_frames,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                height=args.height,
                width=args.width,
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                ref_image=reference.to(device=device, dtype=weight_dtype),
                ref_coordmap=ref_coordmap.to(device=device, dtype=weight_dtype),
                fg_coordmap=target_coordmap.to(device=device, dtype=weight_dtype),
                bg_video=background.to(device=device, dtype=weight_dtype) if background is not None else None,
                start_image=start_image.to(device=device, dtype=weight_dtype) if start_image is not None else None,
                shift=args.shift,
            ).videos
        if not torch.isfinite(generated).all():
            raise FloatingPointError(f"Sample {index}: generated video contains NaN or Inf")

        sample_dir = output_dir / f"sample_{index:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_videos_grid(generated, str(sample_dir / "generated.mp4"), fps=args.fps)
        comparison_parts = [
            _match_video(reference.cpu(), generated.shape[2], args.height, args.width),
            generated.cpu(),
            _match_video(
                target_coordmap.permute(0, 2, 1, 3, 4).cpu(),
                generated.shape[2],
                args.height,
                args.width,
            ),
        ]
        if start_image is not None:
            comparison_parts.append(_match_video(start_image.cpu(), generated.shape[2], args.height, args.width))
        if ground_truth is not None:
            comparison_parts.append(
                _match_video(ground_truth.permute(0, 2, 1, 3, 4), generated.shape[2], args.height, args.width)
            )
        comparison = torch.cat(comparison_parts, dim=3)
        save_videos_grid(comparison, str(sample_dir / "comparison.mp4"), fps=args.fps)
        (sample_dir / "prompt.txt").write_text(sample["text"] + "\n", encoding="utf-8")
        run_config = dict(vars(args))
        run_config.update(
            {
                "resolved_rope_gap": rope_gap,
                "resolved_seed": sample_seed,
                "sample_index": index,
                "resolved_sample": sample,
            }
        )
        (sample_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(f"Inference complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
