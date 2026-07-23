#!/usr/bin/env python
# coding=utf-8
"""Modified from the Diffusers text-to-image training example."""
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers
from decord import VideoReader

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from go_renderer.data.bucket_sampler import (
    ASPECT_RATIO_480,
    ASPECT_RATIO_RANDOM_CROP_480,
    ASPECT_RATIO_RANDOM_CROP_PROB_480,
    RandomSampler,
    get_closest_ratio,
)
from go_renderer.data.dataset_image_video import ImageVideoRefDataset, ImageVideoSampler
from go_renderer.data.validation import (
    IMAGE_EXTENSIONS,
    inspect_media,
    load_training_manifest,
    validate_generation_shape,
    validate_rope_gap,
)
from go_renderer.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel
from go_renderer.models.multiref_transformer3d import CroodRefTransformer3DModel
from go_renderer.pipeline import WanFunCroodRefPipeline
from go_renderer.utils.discrete_sampler import DiscreteSampling
from go_renderer.utils.utils import padding_image, resolve_model_source, save_videos_grid

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

def run_memory_profiler(global_step, output_dir, accelerator):
    """
    Run PyTorch Profiler with memory profiling for one step.
    Returns a context manager that profiles the step and saves results.

    Args:
        global_step: Current training step (for naming output files)
        output_dir: Directory to save profiler outputs
        accelerator: Accelerator instance for device info

    Returns:
        Profiler context manager or None if profiling not available
    """
    if not torch.cuda.is_available():
        return None

    rank = accelerator.process_index
    profile_dir = os.path.join(output_dir, "memory_profile", f"rank_{rank}")
    os.makedirs(profile_dir, exist_ok=True)

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
        with_stack=False,  # Disable stack trace to reduce output size
    )

    return profiler


def export_memory_snapshot(global_step, output_dir, rank=0):
    """
    Export CUDA memory snapshot for detailed memory analysis.
    Requires PyTorch 2.1+ for full functionality.

    Args:
        global_step: Current training step (for naming output files)
        output_dir: Directory to save snapshot
        rank: GPU/process rank for identifying the device
    """
    if not torch.cuda.is_available():
        return

    try:
        snapshot_dir = os.path.join(output_dir, "memory_profile")
        os.makedirs(snapshot_dir, exist_ok=True)

        # Export memory snapshot (PyTorch 2.1+)
        snapshot_path = os.path.join(snapshot_dir, f"memory_snapshot_step_{global_step}_rank{rank}.pickle")
        torch.cuda.memory._dump_snapshot(snapshot_path)
        logger.info(f"[Rank {rank}] Memory snapshot saved to {snapshot_path}")

    except Exception as e:
        logger.warning(f"[Rank {rank}] Failed to export memory snapshot: {e}")


def get_memory_stats_for_logging(device_id=0, rank=None):
    """
    Get GPU memory statistics for wandb logging (in GB).
    Uses torch.cuda.memory_stats() for detailed breakdown.

    Args:
        device_id: CUDA device ID (local device)
        rank: GPU/process rank for metric naming. If None, no rank suffix is added.

    Returns:
        Dictionary of memory statistics for wandb
    """
    if not torch.cuda.is_available():
        return {}

    torch.cuda.synchronize(device_id)

    # Basic memory info
    allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(device_id) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)

    # Add rank suffix to metric names if rank is provided
    suffix = f"_rank{rank}" if rank is not None else ""

    stats = {
        f"memory/allocated_gb{suffix}": allocated,
        f"memory/reserved_gb{suffix}": reserved,
        f"memory/max_allocated_gb{suffix}": max_allocated,
        f"memory/max_reserved_gb{suffix}": max_reserved,
        f"memory/total_gpu_gb{suffix}": total,
        f"memory/free_gb{suffix}": total - reserved,
        f"memory/utilization_pct{suffix}": (allocated / total) * 100,
    }

    # Get detailed memory stats if available
    try:
        mem_stats = torch.cuda.memory_stats(device_id)

        # Active memory (currently in use)
        if "active_bytes.all.current" in mem_stats:
            stats[f"memory/active_gb{suffix}"] = mem_stats["active_bytes.all.current"] / (1024 ** 3)
        if "active_bytes.all.peak" in mem_stats:
            stats[f"memory/active_peak_gb{suffix}"] = mem_stats["active_bytes.all.peak"] / (1024 ** 3)

        # Allocated memory breakdown by size
        if "active_bytes.large_pool.current" in mem_stats:
            stats[f"memory/large_pool_gb{suffix}"] = mem_stats["active_bytes.large_pool.current"] / (1024 ** 3)
        if "active_bytes.small_pool.current" in mem_stats:
            stats[f"memory/small_pool_gb{suffix}"] = mem_stats["active_bytes.small_pool.current"] / (1024 ** 3)

        # Allocation counts (useful for debugging fragmentation)
        if "allocation.all.current" in mem_stats:
            stats[f"memory/num_allocations{suffix}"] = mem_stats["allocation.all.current"]

    except Exception:
        pass  # Older PyTorch versions may not have detailed stats

    return stats

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _validate_zero2_optimizer_partition_size(param_groups, num_processes, zero_stage):
    """Reject ZeRO-2 flat optimizer partitions that exceed 32-bit kernel indexing."""

    if zero_stage != 2:
        return
    max_partition_numel = 2**31 - 1
    for group_index, group in enumerate(param_groups):
        group_numel = sum(parameter.numel() for parameter in group["params"])
        if group_numel == 0:
            continue
        partition_numel = math.ceil(group_numel / num_processes)
        if partition_numel > max_partition_numel:
            required_processes = math.ceil(group_numel / max_partition_numel)
            raise ValueError(
                "DeepSpeed ZeRO-2 optimizer group "
                f"{group_index} would contain {partition_numel:,} elements per process, "
                f"exceeding the safe 32-bit indexing limit of {max_partition_numel:,}. "
                f"Use at least {required_processes} processes, reduce --trainable_modules, "
                "or use a different sharding strategy."
            )


def _ckpt_step(checkpoint_name):
    name = str(checkpoint_name)
    parts = name.split("-")
    if len(parts) == 2 and parts[0] == "checkpoint" and parts[1].isdigit():
        return int(parts[1])
    if (
        len(parts) == 3
        and parts[0] == "checkpoint"
        and parts[1] == "validation"
        and parts[2].isdigit()
    ):
        return int(parts[2])
    return -1


def _resolve_resume_checkpoint(output_dir, resume_from_checkpoint):
    """Resolve `latest`, output-relative names, and explicit checkpoint paths."""

    if not resume_from_checkpoint:
        return None

    output_root = Path(output_dir).expanduser()
    if resume_from_checkpoint == "latest":
        if not output_root.is_dir():
            return None
        candidates = [
            path
            for path in output_root.iterdir()
            if path.is_dir() and _ckpt_step(path.name) >= 0
        ]
        return max(candidates, key=lambda path: _ckpt_step(path.name)).resolve() if candidates else None

    checkpoint = Path(resume_from_checkpoint).expanduser()
    if not checkpoint.is_absolute() and checkpoint.parent == Path("."):
        checkpoint = output_root / checkpoint
    if _ckpt_step(checkpoint.name) < 0:
        raise ValueError(
            "--resume_from_checkpoint must name checkpoint-N or checkpoint-validation-N, "
            f"got: {resume_from_checkpoint}"
        )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    return checkpoint.resolve()


def _save_local_rng_state(output_dir, process_index, rng, index_rng, torch_rng):
    """Save process-local RNG streams without executable pickle payloads."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "rng": rng.bit_generator.state if rng is not None else None,
        "index_rng": index_rng.bit_generator.state,
    }
    (output_path / f"local_rng_state_{process_index}.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    if torch_rng is not None:
        from safetensors.torch import save_file

        save_file(
            {"state": torch_rng.get_state().cpu()},
            str(output_path / f"local_torch_rng_state_{process_index}.safetensors"),
        )


def _load_local_rng_state(input_dir, process_index, rng, index_rng, torch_rng):
    """Restore process-local RNG streams saved by `_save_local_rng_state`."""

    input_path = Path(input_dir)
    json_path = input_path / f"local_rng_state_{process_index}.json"
    torch_path = input_path / f"local_torch_rng_state_{process_index}.safetensors"
    if not json_path.is_file():
        logger.warning(
            "Checkpoint has no process-local RNG state; resume remains compatible but "
            "will not reproduce an uninterrupted run exactly."
        )
        return

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or "index_rng" not in payload:
        raise ValueError(f"Invalid process-local RNG state: {json_path}")
    if rng is not None:
        if payload.get("rng") is None:
            raise ValueError(f"Checkpoint is missing the seeded NumPy RNG state: {json_path}")
        rng.bit_generator.state = payload["rng"]
    index_rng.bit_generator.state = payload["index_rng"]

    if torch_rng is not None:
        if not torch_path.is_file():
            raise FileNotFoundError(f"Checkpoint is missing Torch RNG state: {torch_path}")
        from safetensors.torch import load_file

        torch_rng.set_state(load_file(str(torch_path), device="cpu")["state"])


def sample_ref_frames(total_frames, num_ref_frames_to_use=6, rng=None):
    """
    Sample reference frames with first-frame-aware stratified sampling.
    10% probability to sample only from the early portion of the sequence.

    Args:
        total_frames: Total number of frames available
        num_ref_frames_to_use: Number of frames to sample
        rng: Random number generator (optional)

    Returns:
        List of frame indices to sample (as Python int)
    """
    if total_frames <= num_ref_frames_to_use:
        return list(range(total_frames))
    effective_total_frames = total_frames
    prob = rng.random() if rng is not None else np.random.random()

    if prob < 0.1:
        min_limit = 2 * num_ref_frames_to_use + 1
        if total_frames > min_limit:
            upper_bound = max(min_limit, total_frames // 3)
            effective_total_frames = upper_bound

    ref_batch_index = [0]

    if num_ref_frames_to_use > 1:
        segment_boundaries = np.linspace(0, effective_total_frames - 1, num_ref_frames_to_use, dtype=int)

        for i in range(1, num_ref_frames_to_use):
            segment_start = segment_boundaries[i - 1]
            segment_end = segment_boundaries[i]

            if segment_start >= segment_end:
                segment_end = min(segment_start + 1, effective_total_frames - 1)

            if rng is None:
                random_idx = np.random.randint(max(segment_start + 1, 1), segment_end + 1)
            else:
                random_idx = rng.integers(max(segment_start + 1, 1), segment_end + 1)

            ref_batch_index.append(int(random_idx))

    return ref_batch_index

def log_validation(
    vae,
    text_encoder,
    tokenizer,
    transformer3d,
    args,
    config,
    accelerator,
    weight_dtype,
    global_step,
    num_ref_frames_in_vid=8,
    clip_image_encoder=None,
):

    logger.info("Running validation... ")

    # Set validation dimensions
    if args.validation_size is not None:
        val_height, val_width, val_frames = args.validation_size
        logger.info(f"Using custom validation size: {val_height}x{val_width}, {val_frames} frames")
    else:
        val_height = val_width = args.video_sample_size
        val_frames = args.video_sample_n_frames
        logger.info(f"Using default validation size: {val_height}x{val_width}, {val_frames} frames")
    validate_generation_shape(
        val_height,
        val_width,
        val_frames,
        spatial_multiple=16,
    )

    required_media = {}
    for argument_name, media_name in (
        ("validation_ref_path", "reference RGB"),
        ("validation_ref_coordmap_path", "reference coordinate map"),
        ("validation_fg_coordmap_path", "target coordinate map"),
    ):
        media_value = getattr(args, argument_name, None)
        if media_value is None or not str(media_value).strip():
            raise ValueError(
                f"Validation requires --{argument_name.replace('_', '-')} ({media_name})"
            )
        media_path = Path(media_value).expanduser()
        try:
            media_info = inspect_media(media_path)
        except Exception as exc:
            raise ValueError(
                f"Validation preflight failed for required {media_name} at {media_path}: {exc}"
            ) from exc
        required_media[argument_name] = (media_path, media_info)

    ref_path, ref_info = required_media["validation_ref_path"]
    ref_coordmap_path, ref_coordmap_info = required_media[
        "validation_ref_coordmap_path"
    ]
    fg_coordmap_path, fg_coordmap_info = required_media[
        "validation_fg_coordmap_path"
    ]
    if ref_info.frames != ref_coordmap_info.frames:
        raise ValueError(
            f"Validation reference frame mismatch: RGB {ref_path} has {ref_info.frames} "
            f"frames, but coordinate map {ref_coordmap_path} has "
            f"{ref_coordmap_info.frames} frames"
        )
    if (ref_info.width, ref_info.height) != (
        ref_coordmap_info.width,
        ref_coordmap_info.height,
    ):
        raise ValueError(
            f"Validation reference size mismatch: RGB {ref_path} is "
            f"{ref_info.width}x{ref_info.height}, but coordinate map "
            f"{ref_coordmap_path} is {ref_coordmap_info.width}x"
            f"{ref_coordmap_info.height}"
        )
    if ref_info.frames < 3:
        raise ValueError(
            f"Validation reference {ref_path} has {ref_info.frames} frames; "
            "at least 3 aligned RGB/coordinate-map frames are required"
        )
    if fg_coordmap_info.frames != val_frames:
        raise ValueError(
            f"Validation target coordinate map {fg_coordmap_path} has "
            f"{fg_coordmap_info.frames} frames; exactly {val_frames} are required"
        )
    if not 3 <= num_ref_frames_in_vid <= 8:
        raise ValueError(
            "num_ref_frames_in_vid must be in the inclusive range [3, 8], "
            f"got {num_ref_frames_in_vid}"
        )

    def load_required_media_frames(media_path, frame_indices, media_name, *, pad):
        """Decode selected frames and fail with the required input's context."""

        try:
            if media_path.is_dir():
                image_paths = sorted(
                    candidate
                    for candidate in media_path.iterdir()
                    if candidate.is_file()
                    and candidate.suffix.lower() in IMAGE_EXTENSIONS
                )

                def read_frame(frame_index):
                    with Image.open(image_paths[frame_index]) as image:
                        return image.convert("RGB")

            elif media_path.suffix.lower() in IMAGE_EXTENSIONS:
                if frame_indices != [0]:
                    raise ValueError(
                        f"single image cannot provide frame indices {frame_indices}"
                    )

                def read_frame(frame_index):
                    del frame_index
                    with Image.open(media_path) as image:
                        return image.convert("RGB")

            else:
                video_reader = VideoReader(str(media_path))

                def read_frame(frame_index):
                    return Image.fromarray(
                        video_reader[frame_index].asnumpy()
                    ).convert("RGB")

            arrays = []
            for frame_index in frame_indices:
                frame = read_frame(frame_index)
                if pad:
                    frame = padding_image(frame, val_width, val_height)
                else:
                    frame = frame.resize((val_width, val_height))
                arrays.append(np.asarray(frame, dtype=np.uint8))
            return (
                torch.from_numpy(np.stack(arrays))
                .permute(0, 3, 1, 2)
                .float()
                / 255.0
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load required validation {media_name} from "
                f"{media_path}: {exc}"
            ) from exc

    reference_count = min(num_ref_frames_in_vid, ref_info.frames)
    reference_rng = np.random.default_rng(args.seed)
    ref_frame_indices = sample_ref_frames(
        ref_info.frames,
        num_ref_frames_to_use=reference_count,
        rng=reference_rng,
    )
    logger.info(f"Using aligned validation reference frame indices: {ref_frame_indices}")

    validation_ref_frames = load_required_media_frames(
        ref_path,
        ref_frame_indices,
        "reference RGB",
        pad=True,
    )
    validation_ref = validation_ref_frames.permute(1, 0, 2, 3).unsqueeze(0)
    validation_ref_coordmap = load_required_media_frames(
        ref_coordmap_path,
        ref_frame_indices,
        "reference coordinate map",
        pad=True,
    ).unsqueeze(0)
    validation_fg_coordmap = load_required_media_frames(
        fg_coordmap_path,
        list(range(val_frames)),
        "target coordinate map",
        pad=False,
    ).unsqueeze(0)
    logger.info(
        "Loaded required validation media: ref=%s, ref_coordmap=%s, "
        "fg_coordmap=%s",
        tuple(validation_ref.shape),
        tuple(validation_ref_coordmap.shape),
        tuple(validation_fg_coordmap.shape),
    )

    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    # FSDP validation must keep the distributed wrapper so every rank
    # participates in parameter all-gathers during the forward pass.
    transformer3d_val = transformer3d if args.use_fsdp else accelerator.unwrap_model(transformer3d)
    transformer3d_val.eval()

    # Reuse the frozen CLIP module from training. Loading a duplicate copy at
    # validation time wastes several GiB per rank.
    if clip_image_encoder is None:
        from go_renderer.models import CLIPModel

        clip_image_encoder_path = os.path.join(
            args.model_name,
            config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder'),
        )
        if os.path.exists(clip_image_encoder_path):
            clip_image_encoder = CLIPModel.from_pretrained(clip_image_encoder_path)
        else:
            logger.warning(f"CLIPModel not found at {clip_image_encoder_path}, using None")
    if clip_image_encoder is not None:
        clip_image_encoder.to(accelerator.device, dtype=weight_dtype).eval()

    vae_val = accelerator.unwrap_model(vae).to(accelerator.device, dtype=weight_dtype)
    # A manually sharded T5 must keep its FSDP wrapper so all ranks take part
    # in parameter all-gathers, just like the Transformer.
    text_encoder_val = text_encoder if args.use_fsdp else accelerator.unwrap_model(text_encoder)
    if args.use_fsdp:
        text_encoder_val.to(accelerator.device)

    pipeline = WanFunCroodRefPipeline(
        vae=vae_val,
        text_encoder=text_encoder_val,
        tokenizer=tokenizer,
        transformer=transformer3d_val,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )

    if not args.use_fsdp:
        pipeline = pipeline.to(accelerator.device)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.no_grad():
            os.makedirs(os.path.join(args.output_dir, f"validation/step-{global_step}"), exist_ok=True)

            # Save concatenated ref.gif (validation_ref + validation_ref_coordmap)
            if validation_ref is not None and accelerator.is_main_process:
                os.makedirs(os.path.join(args.output_dir, f"validation"), exist_ok=True)

                if validation_ref_coordmap is not None:
                    # Assert frame counts match
                    ref_frames = validation_ref.shape[2]  # [1, C, F, H, W]
                    coordmap_frames = validation_ref_coordmap.shape[1]  # [1, F, C, H, W]
                    assert ref_frames == coordmap_frames, \
                        f"Frame count mismatch: validation_ref has {ref_frames} frames but validation_ref_coordmap has {coordmap_frames} frames"

                    # Convert validation_ref_coordmap from [1, F, C, H, W] to [1, C, F, H, W]
                    validation_ref_coordmap_reformat = validation_ref_coordmap.permute(0, 2, 1, 3, 4)  # [1, C, F, H, W]

                    # Use comparison_list to concatenate ref and coordmap
                    comparison_list = [validation_ref, validation_ref_coordmap_reformat]
                    ref_and_coordmap = torch.cat(comparison_list, dim=0)  # [2, C, F, H, W]
                    save_videos_grid(ref_and_coordmap, os.path.join(args.output_dir, f"validation/ref.gif"), fps=16)
                    logger.info(f"Saved validation ref.gif with ref+coordmap concatenated, shape: {ref_and_coordmap.shape}")
                else:
                    # Only save validation_ref if coordmap is not available
                    save_videos_grid(validation_ref, os.path.join(args.output_dir, f"validation/ref.gif"), fps=16)
                    logger.info(f"Saved validation ref.gif with ref only, shape: {validation_ref.shape}")

            # Load validation fg (control_video) if provided
            validation_fg = None
            if args.validation_fg_path is not None:
                fg_path = args.validation_fg_path
                logger.info(f"Loading validation fg from: {fg_path}")
                try:
                    vr = VideoReader(fg_path)
                    total_frames = len(vr)
                    # Sample frames to match val_frames
                    if total_frames <= val_frames:
                        frame_indices = list(range(total_frames))
                    else:
                        frame_indices = np.linspace(0, total_frames - 1, val_frames, dtype=int).tolist()

                    fg_frames = []
                    for idx in frame_indices:
                        frame = vr[idx].asnumpy()
                        frame_pil = Image.fromarray(frame)
                        # Resize to validation size
                        frame_pil = frame_pil.resize((val_width, val_height))
                        fg_frames.append(np.array(frame_pil))

                    # Stack to [F, H, W, C]
                    validation_fg = np.stack(fg_frames, axis=0)
                    # Convert to torch and normalize to [0, 1]
                    validation_fg = torch.from_numpy(validation_fg).permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W]
                    validation_fg = validation_fg.unsqueeze(0)  # [1, F, C, H, W]
                    logger.info(f"Loaded validation fg with shape: {validation_fg.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load validation fg: {e}")
                    validation_fg = None

            # Load validation bgvideo if provided
            validation_bgvideo = None
            if args.validation_bgvideo_path is not None:
                bgvideo_path = args.validation_bgvideo_path
                logger.info(f"Loading validation bgvideo from: {bgvideo_path}")
                try:
                    vr = VideoReader(bgvideo_path)
                    total_frames = len(vr)
                    # Sample frames to match val_frames
                    if total_frames <= val_frames:
                        frame_indices = list(range(total_frames))
                    else:
                        frame_indices = np.linspace(0, total_frames - 1, val_frames, dtype=int).tolist()

                    bgvideo_frames = []
                    for idx in frame_indices:
                        frame = vr[idx].asnumpy()
                        frame_pil = Image.fromarray(frame)
                        # Resize to validation size
                        frame_pil = frame_pil.resize((val_width, val_height))
                        bgvideo_frames.append(np.array(frame_pil))

                    # Stack to [F, H, W, C]
                    validation_bgvideo = np.stack(bgvideo_frames, axis=0)
                    # Convert to torch and normalize to [0, 1]
                    validation_bgvideo = torch.from_numpy(validation_bgvideo).permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W]
                    validation_bgvideo = validation_bgvideo.unsqueeze(0)  # [1, F, C, H, W]
                    logger.info(f"Loaded validation bgvideo with shape: {validation_bgvideo.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load validation bgvideo: {e}")
                    validation_bgvideo = None

            # Load optional GT only for the comparison visualization. It must
            # never become a generation condition.
            validation_gt = None
            if args.validation_gt_path is not None:
                gt_path = args.validation_gt_path
                logger.info(f"Loading validation GT from: {gt_path}")
                try:
                    # Check if it's an image or video
                    if gt_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        # Load as image and repeat val_frames times
                        logger.info(f"Loading GT as image and repeating {val_frames} times")
                        img = Image.open(gt_path).convert('RGB')
                        img = img.resize((val_width, val_height))
                        img_array = np.array(img)

                        # Repeat the image val_frames times
                        gt_frames = [img_array] * val_frames

                        # Stack to [F, H, W, C]
                        validation_gt = np.stack(gt_frames, axis=0)
                        # Convert to torch and normalize to [0, 1]
                        validation_gt = torch.from_numpy(validation_gt).permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W]
                        validation_gt = validation_gt.unsqueeze(0)  # [1, F, C, H, W]
                        logger.info(f"Loaded validation GT image with shape: {validation_gt.shape}")
                    else:
                        # Load as video
                        vr = VideoReader(gt_path)
                        total_frames = len(vr)
                        # Sample frames to match val_frames
                        if total_frames <= val_frames:
                            frame_indices = list(range(total_frames))
                        else:
                            frame_indices = np.linspace(0, total_frames - 1, val_frames, dtype=int).tolist()

                        gt_frames = []
                        for idx in frame_indices:
                            frame = vr[idx].asnumpy()
                            frame_pil = Image.fromarray(frame)
                            # Resize to validation size
                            frame_pil = frame_pil.resize((val_width, val_height))
                            gt_frames.append(np.array(frame_pil))

                        # Stack to [F, H, W, C]
                        validation_gt = np.stack(gt_frames, axis=0)
                        # Convert to torch and normalize to [0, 1]
                        validation_gt = torch.from_numpy(validation_gt).permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W]
                        validation_gt = validation_gt.unsqueeze(0)  # [1, F, C, H, W]
                        logger.info(f"Loaded validation GT video with shape: {validation_gt.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load validation GT: {e}")
                    validation_gt = None

            # Generate with reference if provided
            if validation_ref is not None:
                # Prepare ref_image: validation_ref is [1, C, F, H, W]
                # For WanFunCroodRefPipeline, ref_image expects [B, C, F, H, W]
                ref_image_input = validation_ref.to(device=accelerator.device, dtype=weight_dtype)

                # Prepare ref_coordmap if provided
                ref_coordmap_input = None
                if validation_ref_coordmap is not None:
                    # validation_ref_coordmap is [1, F, C, H, W]
                    ref_coordmap_input = validation_ref_coordmap.to(device=accelerator.device, dtype=weight_dtype)
                    logger.info(f"Ref coordmap input shape: {ref_coordmap_input.shape}")

                # Prepare fg_coordmap if provided
                fg_coordmap_input = None
                if validation_fg_coordmap is not None:
                    # validation_fg_coordmap is [1, F, C, H, W]
                    fg_coordmap_input = validation_fg_coordmap.to(device=accelerator.device, dtype=weight_dtype)
                    logger.info(f"FG coordmap input shape: {fg_coordmap_input.shape}")

                # Prepare bg_video if provided
                bg_video_input = None
                if validation_bgvideo is not None:
                    # validation_bgvideo is [1, F, C, H, W]
                    bg_video_input = validation_bgvideo.to(device=accelerator.device, dtype=weight_dtype)
                    logger.info(f"BG video input shape: {bg_video_input.shape}")

                sample_with_ref = pipeline(
                    args.validation_prompts[i],
                    num_frames = val_frames,
                    negative_prompt = DEFAULT_NEGATIVE_PROMPT,
                    height      = val_height,
                    width       = val_width,
                    generator   = generator,
                    ref_image   = ref_image_input,      # validation_ref as ref_image
                    ref_coordmap = ref_coordmap_input,  # validation_ref_coordmap
                    fg_coordmap = fg_coordmap_input,    # validation_fg_coordmap
                    bg_video    = bg_video_input,       # validation_bgvideo
                    start_image = None,
                    num_inference_steps = args.validation_num_inference_steps,
                ).videos

                if accelerator.is_main_process:
                    save_videos_grid(sample_with_ref, os.path.join(args.output_dir, f"validation/step-{global_step}/{i}.mp4"), fps=16)

                # spatial mismatch
                ref_h, ref_w = validation_ref.shape[3], validation_ref.shape[4]
                sample_h, sample_w = sample_with_ref.shape[3], sample_with_ref.shape[4]

                if ref_h != sample_h or ref_w != sample_w:
                    # Keep ratio resize + center crop
                    scale = max(ref_h / sample_h, ref_w / sample_w)
                    new_h, new_w = int(sample_h * scale), int(sample_w * scale)

                    # Resize
                    sample_resized = F.interpolate(
                        sample_with_ref.view(-1, sample_with_ref.shape[1], sample_h, sample_w),
                        size=(new_h, new_w), mode='bilinear', align_corners=False
                    )
                    sample_resized = sample_resized.view(sample_with_ref.shape[0], sample_with_ref.shape[1], sample_with_ref.shape[2], new_h, new_w)

                    # Center crop to match ref size
                    start_h = max(0, (new_h - ref_h) // 2)
                    start_w = max(0, (new_w - ref_w) // 2)
                    end_h = start_h + ref_h
                    end_w = start_w + ref_w
                    sample_with_ref = sample_resized[:, :, :, start_h:end_h, start_w:end_w]

                # Prepare comparison visualization
                # Use fg_coordmap for visualization if available, otherwise use zeros
                if fg_coordmap_input is not None:
                    # fg_coordmap_input is [1, F, C, H, W], convert to [1, C, F, H, W]
                    # Already in [0, 1] range, no conversion needed
                    viz_fg_coordmap = fg_coordmap_input.permute(0, 2, 1, 3, 4)
                else:
                    viz_fg_coordmap = torch.zeros((1, 3, val_frames, val_height, val_width), device=accelerator.device, dtype=weight_dtype)

                # Resample all videos to match val_frames before concatenation
                # Reference: pipeline_wan_croodref.py uses trilinear interpolation for fg_coordmap
                def resample_to_val_frames(video, target_frames):
                    """
                    Resample video to target_frames using interpolation.
                    video: [B, C, F, H, W] or [B, C, H, W] (single frame)
                    """
                    if video is None:
                        return None

                    # Handle single frame case
                    if video.dim() == 4:
                        # [B, C, H, W] -> [B, C, 1, H, W]
                        video = video.unsqueeze(2)

                    current_frames = video.shape[2]
                    if current_frames == target_frames:
                        return video

                    video_resampled = torch.nn.functional.interpolate(
                        video,
                        size=(target_frames, video.shape[3], video.shape[4]),
                        mode='nearest'
                    )

                    return video_resampled

                # Resample all videos to val_frames
                if val_frames == 1:
                    val_frames = 49
                validation_ref_resampled = resample_to_val_frames(validation_ref, val_frames)
                sample_with_ref_resampled = resample_to_val_frames(sample_with_ref, val_frames)
                viz_fg_coordmap_resampled = resample_to_val_frames(viz_fg_coordmap, val_frames)

                comparison_list = [validation_ref_resampled, sample_with_ref_resampled.cpu(), viz_fg_coordmap_resampled.cpu()]

                # Add validation_gt if available
                if validation_gt is not None:
                    # validation_gt is [1, F, C, H, W], convert to [1, C, F, H, W]
                    # Already in [0, 1] range, no conversion needed
                    validation_gt_reformat = validation_gt.permute(0, 2, 1, 3, 4)
                    validation_gt_resampled = resample_to_val_frames(validation_gt_reformat, val_frames)
                    comparison_list.append(validation_gt_resampled.cpu())

                comparison_video = torch.cat(comparison_list, dim=0)  # [3 or 4, C, F, H, W]
                if accelerator.is_main_process:
                    save_videos_grid(comparison_video, os.path.join(args.output_dir, f"validation/step-{global_step}/{i}_comparison.mp4"), fps=16)

                if i == 0 and accelerator.is_main_process: # Log only the first prompt's result
                    log_dict = {}

                    # Prepare comparison for logging
                    log_comparison = comparison_video.clone().detach().clamp(0, 1) * 255
                    log_comparison = log_comparison.permute(0, 2, 1, 3, 4).to(torch.uint8)  # [3 or 4, F, C, H, W]
                    caption = f"{args.validation_prompts[i].split('.')[0]}..."

                    if args.report_to == "wandb":
                        log_dict["validation/comparison"] = wandb.Video(log_comparison.cpu(), fps=16, format="gif", caption=caption)
                    else: # Tensorboard
                        log_dict["validation/comparison"] = log_comparison

                    accelerator.log(log_dict, step=global_step)

    del pipeline
    del transformer3d_val
    if args.low_vram:
        vae.to("cpu")
        text_encoder.to("cpu")
        if clip_image_encoder is not None:
            clip_image_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return images

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--model_name",
        default="models/Diffusion_Transformer/GO-Renderer-14B",
        help="Complete GO-Renderer 14B model directory containing the Transformer, T5, VAE, tokenizer, and CLIP encoder",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Root directory containing the training data.",
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        required=True,
        help="Path to the training metadata JSON file.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_ref_path",
        type=str,
        default=None,
        help=("Required reference image sequence or video when --validation_prompts is set."),
    )
    parser.add_argument(
        "--validation_fg_path",
        type=str,
        default=None,
        help=("Path to foreground video used for validation."),
    )
    parser.add_argument(
        "--validation_bgvideo_path",
        type=str,
        default=None,
        help=("Path to background video used for validation."),
    )
    parser.add_argument(
        "--validation_ref_coordmap_path",
        type=str,
        default=None,
        help=("Required aligned reference coordmap when --validation_prompts is set."),
    )
    parser.add_argument(
        "--validation_fg_coordmap_path",
        type=str,
        default=None,
        help=("Required target-trajectory coordmap when --validation_prompts is set."),
    )
    parser.add_argument(
        "--validation_gt_path",
        type=str,
        default=None,
        help=("Optional GT image or video used only in the validation comparison."),
    )
    parser.add_argument(
        "--validation_size",
        nargs=3,
        type=int,
        default=None,
        help=("Validation size as [height, width, frames]. If not provided, will use args.video_sample_size and args.video_sample_n_frames."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size per process."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        required=True,
        help="Explicit total number of training steps for the experimental 14B run.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=25,
        help="Number of denoising steps used by periodic training validation.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--auto_tile_batch_size", action="store_true", help="Whether to auto tile batch size.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true",
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=480,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=480,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--fix_sample_size",
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=81,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/wan2.1/wan_civitai.yaml",
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        '--trainable_modules',
        nargs='+',
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate',
        nargs='+',
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
    )
    parser.add_argument(
        '--tokenizer_max_length',
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="control",
        help=(
            'The format of training data. Support `"control"`'
            ' (default), `"control_ref"`, `"control_camera_ref"`.'
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--ref_frames_min",
        type=int,
        default=3,
        help="Minimum number of reference frames to sample during training.",
    )
    parser.add_argument(
        "--ref_frames_max",
        type=int,
        default=8,
        help="Maximum number of reference frames to sample during training (inclusive).",
    )
    parser.add_argument(
        "--coordmap_aug_prob",
        type=float,
        default=0.1,
        help="Probability of applying shared coordinate-value perturbation.",
    )
    parser.add_argument(
        "--coordmap_aug_scale",
        type=float,
        default=0.01,
        help="Maximum absolute coordinate translation and relative scale perturbation.",
    )
    parser.add_argument(
        "--enable_memory_probe",
        action="store_true",
        help="Enable PyTorch Profiler memory profiling. Runs profiler on the step before each validation, exports memory snapshot and tensorboard trace for each GPU. Metrics are logged to wandb with rank suffix (e.g., memory/allocated_gb_rank0).",
    )

    args = parser.parse_args()
    if not 3 <= args.ref_frames_min <= args.ref_frames_max <= 8:
        parser.error("reference-frame sampling must stay in the inclusive range [3, 8]")
    if not 0.0 <= args.coordmap_aug_prob <= 1.0:
        parser.error("--coordmap_aug_prob must be in [0, 1]")
    if args.coordmap_aug_scale < 0.0:
        parser.error("--coordmap_aug_scale must be non-negative")
    if args.video_sample_n_frames <= 0 or args.video_sample_n_frames % 4 != 1:
        parser.error("--video_sample_n_frames must be positive and satisfy frames % 4 == 1")
    if args.max_train_steps <= 0:
        parser.error("--max_train_steps must be positive")
    if args.video_sample_size <= 0 or args.video_sample_size % 16:
        parser.error("--video_sample_size must be a positive multiple of 16")
    if args.image_sample_size <= 0 or args.image_sample_size % 16:
        parser.error("--image_sample_size must be a positive multiple of 16")
    if args.fix_sample_size is not None and any(
        value <= 0 or value % 16 for value in args.fix_sample_size
    ):
        parser.error("--fix_sample_size height and width must be positive multiples of 16")
    if args.validation_size is not None:
        try:
            validate_generation_shape(
                *args.validation_size,
                spatial_multiple=16,
            )
        except ValueError as exc:
            parser.error(f"invalid --validation_size: {exc}")
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    config = OmegaConf.load(args.config_path)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    with accelerator.main_process_first():
        args.model_name = resolve_model_source(args.model_name)

    if accelerator.is_main_process:
        load_training_manifest(args.train_data_meta, args.train_data_dir)
    accelerator.wait_for_everyone()

    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"Using DeepSpeed Zero stage: {zero_stage}")

        args.use_deepspeed = True
        if zero_stage == 3:
            print(f"Auto set save_state to True because zero_stage == 3")
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None: # The fsdp_plugin.sharding_strategy is None in FSDP 2.
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")

        args.use_fsdp = True
        if fsdp_stage == 3:
            print(f"Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    resume_checkpoint = _resolve_resume_checkpoint(
        args.output_dir, args.resume_from_checkpoint
    )
    if args.resume_from_checkpoint == "latest" and resume_checkpoint is None:
        accelerator.print("No checkpoint found in the output directory; starting a new training run.")
        args.resume_from_checkpoint = None

    # Cast frozen inference-only components, such as the VAE and text encoder, to
    # the selected mixed-precision dtype.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=False,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()

        # Get Vae
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()

        # Get Clip Image Encoder
        clip_image_encoder_path = os.path.join(args.model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder'))
        if os.path.exists(clip_image_encoder_path):
            clip_image_encoder = CLIPModel.from_pretrained(clip_image_encoder_path)
            clip_image_encoder = clip_image_encoder.eval()
        else:
            print(f"Warning: CLIPModel not found at {clip_image_encoder_path}, using None")
            clip_image_encoder = None

    # A resume checkpoint owns its RoPE spacing; otherwise read it from the complete model.
    transformer_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    transformer_kwargs["rope_gap"] = 5
    transformer_subpath = config["transformer_additional_kwargs"].get(
        "transformer_subpath", "."
    )
    config_candidates = []
    if resume_checkpoint is not None:
        config_candidates.append(resume_checkpoint / "transformer" / "config.json")
    config_candidates.append(
        Path(args.model_name) / transformer_subpath / "config.json"
    )
    for transformer_config_path in config_candidates:
        if transformer_config_path.is_file():
            saved_transformer_config = json.loads(transformer_config_path.read_text(encoding="utf-8"))
            transformer_kwargs["rope_gap"] = int(saved_transformer_config.get("rope_gap", 5))
            break
    transformer_kwargs["rope_gap"] = validate_rope_gap(
        transformer_kwargs["rope_gap"],
        args.ref_frames_max,
    )

    # Get Transformer (use CroodRefTransformer3DModel for GO-Renderer training).
    transformer3d = CroodRefTransformer3DModel.from_pretrained(
        os.path.join(args.model_name, transformer_subpath),
        transformer_additional_kwargs=transformer_kwargs,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)
    if clip_image_encoder is not None:
        clip_image_encoder.requires_grad_(False)

    # A good trainable modules is showed below now.
    # For 3D Patch: trainable_modules = ['ff.net', 'pos_embed', 'attn2', 'proj_out', 'timepositionalencoding', 'h_position', 'w_position']
    # For 2D Patch: trainable_modules = ['ff.net', 'attn2', 'timepositionalencoding', 'h_position', 'w_position']
    transformer3d.train()
    if accelerator.is_main_process:
        accelerator.print(
            f"Trainable modules '{args.trainable_modules}'."
        )
    for name, param in transformer3d.named_parameters():
        for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    # Create EMA for the transformer3d.
    if args.use_ema:
        if zero_stage == 3 or fsdp_stage != 0:
            raise NotImplementedError("FSDP and DeepSpeed ZeRO-3 do not support EMA.")

        ema_transformer3d = CroodRefTransformer3DModel.from_pretrained(
            os.path.join(args.model_name, transformer_subpath),
            transformer_additional_kwargs=transformer_kwargs,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True,

        )

        ema_transformer3d = EMAModel(ema_transformer3d.parameters(), model_cls=CroodRefTransformer3DModel, model_config=ema_transformer3d.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file

                    transformer_output_dir = os.path.join(output_dir, "transformer")
                    os.makedirs(transformer_output_dir, exist_ok=True)
                    safetensor_save_path = os.path.join(
                        transformer_output_dir, "diffusion_pytorch_model.safetensors"
                    )
                    accelerate_state_dict = {k: v.to(dtype=weight_dtype) for k, v in accelerate_state_dict.items()}
                    save_file(accelerate_state_dict, safetensor_save_path, metadata={"format": "pt"})
                    accelerator.unwrap_model(models[-1]).save_config(transformer_output_dir)
                _save_local_rng_state(
                    output_dir, accelerator.process_index, rng, index_rng, torch_rng
                )

            def load_model_hook(models, input_dir):
                _load_local_rng_state(
                    input_dir, accelerator.process_index, rng, index_rng, torch_rng
                )

        elif zero_stage == 3:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                _save_local_rng_state(
                    output_dir, accelerator.process_index, rng, index_rng, torch_rng
                )

            def load_model_hook(models, input_dir):
                _load_local_rng_state(
                    input_dir, accelerator.process_index, rng, index_rng, torch_rng
                )
        else:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if args.use_ema:
                        ema_transformer3d.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    if not args.use_deepspeed:
                        weights.pop()
                _save_local_rng_state(
                    output_dir, accelerator.process_index, rng, index_rng, torch_rng
                )

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    ema_path = os.path.join(input_dir, "transformer_ema")
                    _, ema_kwargs = CroodRefTransformer3DModel.load_config(ema_path, return_unused_kwargs=True)
                    load_model = CroodRefTransformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer_ema",
                        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs'])
                    )
                    load_model = EMAModel(load_model.parameters(), model_cls=CroodRefTransformer3DModel, model_config=load_model.config)
                    load_model.load_state_dict(ema_kwargs)

                    ema_transformer3d.load_state_dict(load_model.state_dict())
                    ema_transformer3d.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = CroodRefTransformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer"
                    )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model
                _load_local_rng_state(
                    input_dir, accelerator.process_index, rng, index_rng, torch_rng
                )

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    in_already = []
    for name, param in transformer3d.named_parameters():
        high_lr_flag = False
        if name in in_already:
            continue
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                high_lr_flag = True
                trainable_params_optim[0]['params'].append(param)
                # if accelerator.is_main_process:
                #     print(f"Set {name} to lr : {args.learning_rate}")
                break
        if high_lr_flag:
            continue
        for trainable_module_name in args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                trainable_params_optim[1]['params'].append(param)
                # if accelerator.is_main_process:
                #     print(f"Set {name} to lr : {args.learning_rate / 2}")
                break

    _validate_zero2_optimizer_partition_size(
        trainable_params_optim,
        accelerator.num_processes,
        zero_stage,
    )
    optimizer = torch.optim.AdamW(
        trainable_params_optim,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the training dataset
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio

    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False

    # Get the dataset - using ImageVideoRefDataset which handles ref, bg_mask, fg, coordmap
    train_dataset = ImageVideoRefDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames,
        video_repeat=args.video_repeat,
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket,
        enable_inpaint=True if args.train_mode != "normal" else False,
    )

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn

    if args.enable_bucket:
        aspect_ratio_sample_size = {key : [x / 480 * args.video_sample_size for x in ASPECT_RATIO_480[key]] for key in ASPECT_RATIO_480.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator),
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            drop_last=True,
        )

        def collate_fn(examples):
            def get_length_to_frame_num(token_length):
                if args.image_sample_size > args.video_sample_size:
                    sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

                    if sample_sizes[-1] != args.image_sample_size:
                        sample_sizes.append(args.image_sample_size)
                else:
                    sample_sizes = [args.image_sample_size]

                length_to_frame_num = {
                    sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
                }

                return length_to_frame_num

            def get_random_downsample_ratio(sample_size, image_ratio=[],
                                            all_choices=False, rng=None):
                def _create_special_list(length):
                    if length == 1:
                        return [1.0]
                    if length >= 2:
                        first_element = 0.90
                        remaining_sum = 1.0 - first_element
                        other_elements_value = remaining_sum / (length - 1)
                        special_list = [first_element] + [other_elements_value] * (length - 1)
                        return special_list

                if sample_size >= 1536:
                    number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio
                elif sample_size >= 1024:
                    number_list = [1, 1.25, 1.5, 2] + image_ratio
                elif sample_size >= 768:
                    number_list = [1, 1.25, 1.5] + image_ratio
                elif sample_size >= 512:
                    number_list = [1] + image_ratio
                else:
                    number_list = [1]

                if all_choices:
                    return number_list

                number_list_prob = np.array(_create_special_list(len(number_list)))
                if rng is None:
                    return np.random.choice(number_list, p = number_list_prob)
                else:
                    return rng.choice(number_list, p = number_list_prob)

            # Get token length
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(target_token_length)

            # Create new output
            new_examples                 = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            # Used in Ref Mode
            new_examples["ref_pixel_values"] = []
            new_examples["bg_mask"] = []
            new_examples["fg"] = []
            new_examples["bg"] = []
            new_examples["ref_coordmap"] = []
            new_examples["fg_coordmap"] = []

            # Get downsample ratio in image and videos
            pixel_value     = examples[0]["pixel_values"]
            data_type       = examples[0]["data_type"]
            is_video_batch  = data_type == 'video'
            f, h, w, c      = np.shape(pixel_value)
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size])

                aspect_ratio_sample_size = {key : [x / 480 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_480[key]] for key in ASPECT_RATIO_480.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 480 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_480[key]] for key in ASPECT_RATIO_RANDOM_CROP_480.keys()}

                batch_video_length = 1
            else:
                if args.random_hw_adapt:
                    if args.training_with_video_token_length:
                        local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))

                        def get_random_downsample_probability(choice_list, token_sample_size):
                            length = len(choice_list)
                            if length == 1:
                                return [1.0]  # If there's only one element, it gets all the probability

                            # Find the index of the closest value to token_sample_size
                            closest_index = min(range(length), key=lambda i: abs(choice_list[i] - token_sample_size))

                            # Assign 50% to the closest index
                            first_element = 0.50
                            remaining_sum = 1.0 - first_element

                            # Distribute the remaining 50% evenly among the other elements
                            other_elements_value = remaining_sum / (length - 1) if length > 1 else 0.0

                            # Construct the probability distribution
                            probability_list = [other_elements_value] * length
                            probability_list[closest_index] = first_element

                            return probability_list

                        choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                        if len(choice_list) == 0:
                            choice_list = list(length_to_frame_num.keys())
                        probabilities = get_random_downsample_probability(choice_list, args.token_sample_size)
                        local_video_sample_size = np.random.choice(choice_list, p=probabilities)

                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size)
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                aspect_ratio_sample_size = {key : [x / 480 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_480[key]] for key in ASPECT_RATIO_480.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 480 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_480[key]] for key in ASPECT_RATIO_RANDOM_CROP_480.keys()}

            if args.fix_sample_size is not None:
                fix_sample_size = list(args.fix_sample_size)
            elif args.random_ratio_crop:
                if rng is None:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB_480)
                    ]
                else:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB_480)
                    ]
                random_sample_size = [int(x / 16) * 16 for x in random_sample_size]
            else:
                closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
                closest_size = [int(x / 16) * 16 for x in closest_size]

            for example in examples:
                # To 0~1
                pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.

                # Process ref_pixel_values from ImageVideoRefDataset
                ref_pixel_values = example["ref_pixel_values"]
                if isinstance(ref_pixel_values, np.ndarray):
                    ref_pixel_values = torch.from_numpy(ref_pixel_values).permute(0, 3, 1, 2).contiguous()
                    ref_pixel_values = ref_pixel_values / 255.
                elif isinstance(ref_pixel_values, torch.Tensor):
                    if ref_pixel_values.dtype == torch.uint8:
                        ref_pixel_values = ref_pixel_values.float() / 255.

                # Process bg_mask if exists
                bg_mask = example.get("bg_mask", None)
                if bg_mask is not None:
                    if isinstance(bg_mask, np.ndarray):
                        bg_mask = torch.from_numpy(bg_mask).permute(0, 3, 1, 2).contiguous()
                        bg_mask = bg_mask / 255.
                    elif isinstance(bg_mask, torch.Tensor):
                        if bg_mask.dtype == torch.uint8:
                            bg_mask = bg_mask.float() / 255.

                # Process fg if exists
                fg = example.get("fg", None)
                if fg is not None:
                    if isinstance(fg, np.ndarray):
                        fg = torch.from_numpy(fg).permute(0, 3, 1, 2).contiguous()
                        fg = fg / 255.
                    elif isinstance(fg, torch.Tensor):
                        if fg.dtype == torch.uint8:
                            fg = fg.float() / 255.

                if args.fix_sample_size is not None:
                    # Get adapt hw for resize
                    fix_sample_size = list(map(lambda x: int(x), fix_sample_size))
                    transform = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(fix_sample_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])

                    transform_no_normalize = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(fix_sample_size),
                    ])
                elif args.random_ratio_crop:
                    # Get adapt hw for resize
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)

                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])

                    transform_no_normalize = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                    ])
                else:
                    # Get adapt hw for resize
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]

                    transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])

                    transform_no_normalize = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                    ])

                new_examples["pixel_values"].append(transform(pixel_values))
                new_examples["ref_pixel_values"].append(transform(ref_pixel_values))

                # Append bg_mask if exists
                if bg_mask is not None:
                    new_examples["bg_mask"].append(transform(bg_mask))
                else:
                    new_examples["bg_mask"].append(None)

                # Append fg if exists
                if fg is not None:
                    new_examples["fg"].append(transform(fg))
                else:
                    new_examples["fg"].append(None)

                # Append bg if exists
                bg = example.get("bg", None)
                if bg is not None:
                    if isinstance(bg, np.ndarray):
                        bg = torch.from_numpy(bg).permute(0, 3, 1, 2).contiguous()
                        bg = bg / 255.
                    elif isinstance(bg, torch.Tensor):
                        if bg.dtype == torch.uint8:
                            bg = bg.float() / 255.
                    new_examples["bg"].append(transform(bg))
                else:
                    new_examples["bg"].append(None)

                # Append ref_coordmap if exists
                ref_coordmap = example.get("ref_coordmap", None)
                if ref_coordmap is not None:
                    if isinstance(ref_coordmap, np.ndarray):
                        ref_coordmap = torch.from_numpy(ref_coordmap).permute(0, 3, 1, 2).contiguous()
                        ref_coordmap = ref_coordmap / 255.
                    elif isinstance(ref_coordmap, torch.Tensor):
                        if ref_coordmap.dtype == torch.uint8:
                            ref_coordmap = ref_coordmap.float() / 255.
                    new_examples["ref_coordmap"].append(transform(ref_coordmap))
                else:
                    new_examples["ref_coordmap"].append(None)

                # Append fg_coordmap if exists
                fg_coordmap = example.get("fg_coordmap", None)
                if fg_coordmap is not None:
                    if isinstance(fg_coordmap, np.ndarray):
                        fg_coordmap = torch.from_numpy(fg_coordmap).permute(0, 3, 1, 2).contiguous()
                        fg_coordmap = fg_coordmap / 255.
                    elif isinstance(fg_coordmap, torch.Tensor):
                        if fg_coordmap.dtype == torch.uint8:
                            fg_coordmap = fg_coordmap.float() / 255.
                    new_examples["fg_coordmap"].append(transform(fg_coordmap))
                else:
                    new_examples["fg_coordmap"].append(None)

                new_examples["text"].append(example["text"])
                # Magvae needs the number of frames to be 4n + 1.
                if is_video_batch:
                    batch_video_length = int(
                        min(
                            batch_video_length,
                            (len(pixel_values) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1,
                        )
                    )
                    if batch_video_length == 0:
                        batch_video_length = 1
                else:
                    batch_video_length = 1

                # if args.train_mode != "control":
                #     if args.control_ref_image == "first_frame":
                #         clip_index = 0
                #     else:
                #         def _create_special_list(length):
                #             if length == 1:
                #                 return [1.0]
                #             if length >= 2:
                #                 first_element = 0.40
                #                 remaining_sum = 1.0 - first_element
                #                 other_elements_value = remaining_sum / (length - 1)
                #                 special_list = [first_element] + [other_elements_value] * (length - 1)
                #                 return special_list
                #         number_list_prob = np.array(_create_special_list(len(new_examples["pixel_values"][-1])))
                #         clip_index = np.random.choice(list(range(len(new_examples["pixel_values"][-1]))), p = number_list_prob)
                #     new_examples["clip_idx"].append(clip_index)
                #
                #     ref_pixel_values = new_examples["pixel_values"][-1][clip_index].unsqueeze(0)
                #     new_examples["ref_pixel_values"].append(ref_pixel_values)
                #
                #     clip_pixel_values = new_examples["pixel_values"][-1][clip_index].permute(1, 2, 0).contiguous()
                #     clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                #     new_examples["clip_pixel_values"].append(clip_pixel_values)

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["pixel_values"]])

            # Handle ref_pixel_values with different frame counts by cycling shorter ones
            all_ref_counts = [ref.size(0) for ref in new_examples["ref_pixel_values"]]
            if any(c is not None for c in new_examples.get("ref_coordmap", [])):
                all_ref_counts += [c.size(0) for c in new_examples["ref_coordmap"] if c is not None]
            target_max_ref_f = max(all_ref_counts) if all_ref_counts else 0

            # ref_pixel_values
            if len(new_examples["ref_pixel_values"]) > 0:
                aligned_ref_pixel_values = []
                for ref in new_examples["ref_pixel_values"]:
                    cur_f = ref.size(0)
                    if cur_f < target_max_ref_f:
                        repeat_times = target_max_ref_f // cur_f
                        remainder = target_max_ref_f % cur_f
                        cycled = torch.cat([ref.repeat(repeat_times, 1, 1, 1), ref[:remainder]], dim=0)
                        aligned_ref_pixel_values.append(cycled)
                    else:
                        aligned_ref_pixel_values.append(ref[:target_max_ref_f])
                new_examples["ref_pixel_values"] = torch.stack(aligned_ref_pixel_values)
            else:
                new_examples["ref_pixel_values"] = torch.stack(new_examples["ref_pixel_values"])

            # ref_coordmap
            if any(example is not None for example in new_examples["ref_coordmap"]):
                ref_e = next(e for e in new_examples["ref_coordmap"] if e is not None)
                aligned_ref_coordmap = []
                for ref_coord in new_examples["ref_coordmap"]:
                    if ref_coord is None:
                        aligned_ref_coordmap.append(torch.zeros((target_max_ref_f, *ref_e.shape[1:])))
                    else:
                        cur_f = ref_coord.size(0)
                        if cur_f < target_max_ref_f:
                            repeat_times = target_max_ref_f // cur_f
                            remainder = target_max_ref_f % cur_f
                            cycled = torch.cat([ref_coord.repeat(repeat_times, 1, 1, 1), ref_coord[:remainder]], dim=0)
                            aligned_ref_coordmap.append(cycled)
                        else:
                            aligned_ref_coordmap.append(ref_coord[:target_max_ref_f])
                new_examples["ref_coordmap"] = torch.stack(aligned_ref_coordmap)
            else:
                new_examples.pop("ref_coordmap")

            # Stack bg_mask if they exist
            if any(example is not None for example in new_examples["bg_mask"]):
                ref_e = next(e for e in new_examples["bg_mask"] if e is not None)
                res = []
                for e in new_examples["bg_mask"]:
                    if e is None:
                        res.append(torch.full((batch_video_length, *ref_e.shape[1:]), -1.0))
                    else:
                        cur_f = e.size(0)
                        if cur_f < batch_video_length:
                            res.append(torch.cat([e.repeat(batch_video_length // cur_f, 1, 1, 1), e[:batch_video_length % cur_f]], dim=0))
                        else:
                            res.append(e[:batch_video_length])
                new_examples["bg_mask"] = torch.stack(res)
            else:
                new_examples.pop("bg_mask")

            # Stack fg if they exist
            if any(example is not None for example in new_examples["fg"]):
                example_template = next(e for e in new_examples["fg"] if e is not None)
                result = []
                for e in new_examples["fg"]:
                    if e is None:
                        result.append(torch.zeros((batch_video_length, *example_template.shape[1:])))
                    else:
                        cur_f = e.size(0)
                        if cur_f < batch_video_length:
                            result.append(torch.cat([e.repeat(batch_video_length // cur_f, 1, 1, 1), e[:batch_video_length % cur_f]], dim=0))
                        else:
                            result.append(e[:batch_video_length])
                new_examples["fg"] = torch.stack(result)
            else:
                new_examples.pop("fg")

            # Stack bg if they exist
            if any(example is not None for example in new_examples["bg"]):
                example_template = next(e for e in new_examples["bg"] if e is not None)
                result = []
                for e in new_examples["bg"]:
                    if e is None:
                        result.append(torch.zeros((batch_video_length, *example_template.shape[1:])))
                    else:
                        cur_f = e.size(0)
                        if cur_f < batch_video_length:
                            result.append(torch.cat([e.repeat(batch_video_length // cur_f, 1, 1, 1), e[:batch_video_length % cur_f]], dim=0))
                        else:
                            result.append(e[:batch_video_length])
                new_examples["bg"] = torch.stack(result)
            else:
                new_examples.pop("bg")

            # Stack fg_coordmap if they exist
            if any(example is not None for example in new_examples["fg_coordmap"]):
                example_template = next(e for e in new_examples["fg_coordmap"] if e is not None)
                result = []
                for e in new_examples["fg_coordmap"]:
                    if e is None:
                        result.append(torch.zeros((batch_video_length, *example_template.shape[1:])))
                    else:
                        cur_f = e.size(0)
                        if cur_f < batch_video_length:
                            result.append(torch.cat([e.repeat(batch_video_length // cur_f, 1, 1, 1), e[:batch_video_length % cur_f]], dim=0))
                        else:
                            result.append(e[:batch_video_length])
                new_examples["fg_coordmap"] = torch.stack(result)
            else:
                new_examples.pop("fg_coordmap")


            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'],
                    max_length=args.tokenizer_max_length,
                    padding="max_length",
                    add_special_tokens=True,
                    truncation=True,
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(
                    prompt_ids.input_ids
                )[0]
                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = encoder_hidden_states

            return new_examples

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )
    else:
        # DataLoaders creation:
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(
            RandomSampler(train_dataset, generator=batch_sampler_generator),
            train_dataset,
            args.train_batch_size,
            drop_last=True,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )

    # Accelerate shards complete batches across processes.  With a custom batch
    # sampler it cannot manufacture a batch for an otherwise empty rank, so fail
    # before distributed initialization instead of reaching a rank-local divide
    # by zero after ``accelerator.prepare``.
    if len(train_dataloader) < accelerator.num_processes:
        raise ValueError(
            "The training dataloader must contain at least one complete batch per process "
            f"before sharding (got {len(train_dataloader)} batches for "
            f"{accelerator.num_processes} processes). Add samples, reduce the per-process "
            "batch size, or increase --video_repeat."
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    if fsdp_stage != 0:
        from functools import partial
        from go_renderer.dist.fsdp import shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        text_encoder = shard_fn(text_encoder)

    if args.use_ema:
        ema_transformer3d.to(accelerator.device)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device if not args.low_vram else "cpu")
    if clip_image_encoder is not None:
        clip_image_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if num_update_steps_per_epoch == 0:
        raise RuntimeError("The prepared training dataloader is empty on this process.")
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        tracker_config.pop("trainable_modules")
        tracker_config.pop("trainable_modules_low_learning_rate")
        tracker_config.pop("fix_sample_size")
        tracker_config.pop("validation_size")

        init_kwargs = {}

        if "wandb" in args.report_to.split(','):
            wandb_kwargs = {}
            run_name = os.path.basename(args.output_dir.rstrip("/"))
            wandb_kwargs["name"] = run_name

            wandb_run_id_file = os.path.join(args.output_dir, "wandb_run_id.txt")

            if args.resume_from_checkpoint:
                if os.path.exists(wandb_run_id_file):
                    with open(wandb_run_id_file, "r") as f:
                        wandb_run_id = f.read().strip()
                    if wandb_run_id:
                        logger.info(f"Resuming W&B run with ID: {wandb_run_id}")
                        wandb_kwargs["id"] = wandb_run_id
                        wandb_kwargs["resume"] = "allow"
                else:
                    logger.warning(
                        f"Trying to resume training, but no wandb run ID file found at {wandb_run_id_file}. "
                        "A new W&B run will be created."
                    )

            init_kwargs["wandb"] = wandb_kwargs

        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
            init_kwargs=init_kwargs
        )

        if "wandb" in args.report_to.split(',') and "id" not in init_kwargs.get("wandb", {}):
            try:
                wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
                run_id_to_save = wandb_tracker.id
                wandb_run_id_file = os.path.join(args.output_dir, "wandb_run_id.txt")
                with open(wandb_run_id_file, "w") as f:
                    f.write(run_id_to_save)
                logger.info(f"Saved new W&B run ID to {wandb_run_id_file}: {run_id_to_save}")
            except Exception as e:
                logger.error(f"Could not get or save wandb run ID: {e}")

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_batch_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if resume_checkpoint is None:
            raise RuntimeError("Resume checkpoint resolution unexpectedly returned no path")
        path = resume_checkpoint.name
        global_step = _ckpt_step(path)
        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
        resume_batch_step = (
            global_step % num_update_steps_per_epoch
        ) * args.gradient_accumulation_steps

        accelerator.print(f"Resuming from checkpoint {resume_checkpoint}")
        accelerator.load_state(str(resume_checkpoint))
        # Distributed prefetch makes a serialized sampler cursor ambiguous.
        # Derive the exact position from completed optimizer steps instead.
        if hasattr(batch_sampler.sampler, "_pos_start"):
            batch_sampler.sampler._pos_start = 0
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream and args.train_mode != "normal":
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # Pre-compute clip_fea template from dummy black image (only once before training)
    clip_fea_template = None
    if clip_image_encoder is not None:
        logger.info("Pre-computing clip_fea template from dummy black image...")
        with torch.no_grad():
            if args.low_vram:
                clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

            clip_image_dummy = Image.new("RGB", (512, 512), color=(0, 0, 0))
            clip_image_dummy = TF.to_tensor(clip_image_dummy).sub_(0.5).div_(0.5).to(clip_image_encoder.device, weight_dtype)
            clip_fea_template = clip_image_encoder([clip_image_dummy[:, None, :, :]])  # [1, 1, dim]

            if args.low_vram:
                clip_image_encoder.to('cpu')
                torch.cuda.empty_cache()
        logger.info(f"Pre-computed clip_fea template with shape: {clip_fea_template.shape}")

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        active_dataloader = train_dataloader
        if epoch == first_epoch and resume_batch_step > 0:
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_batch_step
            )
        for step, batch in enumerate(active_dataloader):
            # Check if this is the step before validation (for memory profiling)
            # Note: global_step is incremented at the end of the step, so we check (global_step + 1)
            # All processes participate in profiling to record each GPU's memory
            is_step_before_validation = (
                args.enable_memory_probe and
                args.validation_prompts is not None and
                ((global_step + 1) % args.validation_steps == 0 or args.max_train_steps - global_step == 1)
            )

            # Initialize profiler context (will be used if profiling this step)
            profiler_ctx = None
            if is_step_before_validation:
                logger.info(f"[Rank {accelerator.process_index}] Starting memory profiler for step {global_step} (step before validation)")
                # Start CUDA memory history recording for snapshot
                try:
                    torch.cuda.memory._record_memory_history(max_entries=100000)
                except Exception as e:
                    logger.warning(f"[Rank {accelerator.process_index}] Could not start memory history recording: {e}")
                profiler_ctx = run_memory_profiler(global_step, args.output_dir, accelerator)
                if profiler_ctx is not None:
                    profiler_ctx.__enter__()

            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)

                # Save ref_pixel_values
                ref_pixel_values = batch["ref_pixel_values"].cpu()
                ref_pixel_values = rearrange(ref_pixel_values, "b f c h w -> b c f h w")
                for idx, (ref_pixel_value, text) in enumerate(zip(ref_pixel_values, texts)):
                    ref_pixel_value = ref_pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(ref_pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}_ref.gif", rescale=True)

                # Save bg_mask if exists
                if batch.get("bg_mask") is not None:
                    bg_mask_pixel_values = batch["bg_mask"].cpu()
                    bg_mask_pixel_values = rearrange(bg_mask_pixel_values, "b f c h w -> b c f h w")
                    for idx, (bg_mask_pixel_value, text) in enumerate(zip(bg_mask_pixel_values, texts)):
                        bg_mask_pixel_value = bg_mask_pixel_value[None, ...]
                        gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                        save_videos_grid(bg_mask_pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}_bgmask.gif", rescale=True)

                # Save fg if exists
                if batch.get("fg") is not None:
                    fg_pixel_values = batch["fg"].cpu()
                    fg_pixel_values = rearrange(fg_pixel_values, "b f c h w -> b c f h w")
                    for idx, (fg_pixel_value, text) in enumerate(zip(fg_pixel_values, texts)):
                        fg_pixel_value = fg_pixel_value[None, ...]
                        gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                        save_videos_grid(fg_pixel_value, f"{args.output_dir}/sanity_check/{gif_name[:10]}_fg.gif", rescale=True)

            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                ref_pixel_values = batch["ref_pixel_values"].to(weight_dtype)  # [B, F, C, H, W]

                # Sample ref frames: each batch uses different number of frames
                # Randomly choose number of frames for this batch
                if rng is None:
                    num_ref_frames_to_use = np.random.randint(args.ref_frames_min, args.ref_frames_max + 1)
                else:
                    num_ref_frames_to_use = rng.integers(args.ref_frames_min, args.ref_frames_max + 1)

                # Sample ref_pixel_values for each sample in batch
                batch_size = ref_pixel_values.shape[0]
                sampled_ref_pixel_values = []
                sampled_ref_coordmap = []

                has_coord = batch.get("ref_coordmap") is not None
                ref_coordmap_data = batch["ref_coordmap"].to(weight_dtype) if has_coord else None

                # Unified Sampling and Alignment
                # This ensures intra-sample alignment and handles varying lengths for Non-Bucket mode
                for b in range(batch_size):
                    # Get current sample data (handles both Tensor from Bucket or List from Non-Bucket)
                    current_ref = ref_pixel_values[b]
                    total_f = current_ref.shape[0]

                    # Generate indices for this specific sample
                    ref_batch_index = sample_ref_frames(total_f, num_ref_frames_to_use, rng=rng)
                    ref_idx_tensor = torch.tensor(ref_batch_index, dtype=torch.long, device=accelerator.device)

                    def extract_and_enforce_length(source, idxs, target_l):
                        """Extract frames and ensure the sequence is exactly target_l long."""
                        extracted = source[idxs]
                        cur_l = extracted.shape[0]
                        if cur_l < target_l:
                            # Cycle to fill if sample_ref_frames returned fewer indices than requested
                            repeat_times = target_l // cur_l
                            remainder = target_l % cur_l
                            return torch.cat([extracted.repeat(repeat_times, 1, 1, 1), extracted[:remainder]], dim=0)
                        return extracted[:target_l]

                    # Apply same sampling and alignment to both ref and coordmap
                    sampled_ref_pixel_values.append(extract_and_enforce_length(current_ref, ref_idx_tensor, num_ref_frames_to_use))

                    if has_coord:
                        sampled_ref_coordmap.append(extract_and_enforce_length(ref_coordmap_data[b], ref_idx_tensor, num_ref_frames_to_use))

                # Final Stack
                ref_pixel_values = torch.stack(sampled_ref_pixel_values) # [B, num_ref_frames_to_use, C, H, W]

                ref_coordmap = None
                if has_coord:
                    ref_coordmap = torch.stack(sampled_ref_coordmap) # [B, num_ref_frames_to_use, C, H, W]

                if args.train_mode == "control_camera_ref":
                    control_camera_values = batch["control_camera_values"].to(weight_dtype)

                # Increase the batch size when the length of the latent sequence of the current sample is small
                if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                    if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (4, 1, 1, 1, 1))
                        control_pixel_values = torch.tile(control_pixel_values, (4, 1, 1, 1, 1))
                        if args.train_mode == "control_camera_ref":
                            control_camera_values = torch.tile(control_camera_values, (4, 1, 1, 1, 1))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (4, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (4, 1))
                        else:
                            batch['text'] = batch['text'] * 4
                    elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (2, 1, 1, 1, 1))
                        control_pixel_values = torch.tile(control_pixel_values, (2, 1, 1, 1, 1))
                        if args.train_mode == "control_camera_ref":
                            control_camera_values = torch.tile(control_camera_values, (2, 1, 1, 1, 1))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (2, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (2, 1))
                        else:
                            batch['text'] = batch['text'] * 2

                if args.train_mode != "control":
                    # clip_pixel_values = batch["clip_pixel_values"]
                    # clip_idx = batch["clip_idx"]
                    # Increase the batch size when the length of the latent sequence of the current sample is small
                    if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            # clip_pixel_values = torch.tile(clip_pixel_values, (4, 1, 1, 1))
                            ref_pixel_values = torch.tile(ref_pixel_values, (4, 1, 1, 1, 1))
                            # clip_idx = torch.tile(clip_idx, (4,))
                        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            # clip_pixel_values = torch.tile(clip_pixel_values, (2, 1, 1, 1))
                            ref_pixel_values = torch.tile(ref_pixel_values, (2, 1, 1, 1, 1))
                            # clip_idx = torch.tile(clip_idx, (2,))

                if args.random_frame_crop:
                    def _create_special_list(length):
                        if length == 1:
                            return [1.0]
                        if length >= 2:
                            last_element = 0.90
                            remaining_sum = 1.0 - last_element
                            other_elements_value = remaining_sum / (length - 1)
                            special_list = [other_elements_value] * (length - 1) + [last_element]
                            return special_list
                    select_frames = [_tmp for _tmp in list(range(sample_n_frames_bucket_interval + 1, args.video_sample_n_frames + sample_n_frames_bucket_interval, sample_n_frames_bucket_interval))]
                    select_frames_prob = np.array(_create_special_list(len(select_frames)))

                    if len(select_frames) != 0:
                        if rng is None:
                            temp_n_frames = np.random.choice(select_frames, p = select_frames_prob)
                        else:
                            temp_n_frames = rng.choice(select_frames, p = select_frames_prob)
                    else:
                        temp_n_frames = 1

                    # Magvae needs the number of frames to be 4n + 1.
                    temp_n_frames = ((temp_n_frames - 1) // sample_n_frames_bucket_interval) * sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :temp_n_frames, :, :]
                    # Note: ref_pixel_values may have different frame count, don't crop it here

                    if batch.get("bg_mask") is not None:
                        bg_mask = batch["bg_mask"].to(weight_dtype)
                        bg_mask = bg_mask[:, :temp_n_frames, :, :]

                    if batch.get("bg") is not None:
                        bg = batch["bg"].to(weight_dtype)
                        bg = bg[:, :temp_n_frames, :, :]

                    if batch.get("fg") is not None:
                        fg = batch["fg"].to(weight_dtype)
                        fg = fg[:, :temp_n_frames, :, :]

                    if batch.get("fg_coordmap") is not None:
                        fg_coordmap = batch["fg_coordmap"].to(weight_dtype)
                        fg_coordmap = fg_coordmap[:, :temp_n_frames, :, :]

                # Keep all node same token length to accelerate the traning when resolution grows.
                if args.keep_all_node_same_token_length:
                    if args.token_sample_size > 256:
                        numbers_list = list(range(256, args.token_sample_size + 1, 128))

                        if numbers_list[-1] != args.token_sample_size:
                            numbers_list.append(args.token_sample_size)
                    else:
                        numbers_list = [256]
                    numbers_list = [_number * _number * args.video_sample_n_frames for _number in  numbers_list]

                    actual_token_length = index_rng.choice(numbers_list)
                    actual_video_length = (min(
                            actual_token_length / pixel_values.size()[-1] / pixel_values.size()[-2], args.video_sample_n_frames
                    ) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
                    actual_video_length = int(max(actual_video_length, 1))

                    # Magvae needs the number of frames to be 4n + 1.
                    actual_video_length = (actual_video_length - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :actual_video_length, :, :]
                    # Note: ref_pixel_values may have different frame count, don't crop it here

                    if batch.get("bg_mask") is not None:
                        bg_mask = batch["bg_mask"].to(weight_dtype)
                        bg_mask = bg_mask[:, :actual_video_length, :, :]

                    if batch.get("bg") is not None:
                        bg = batch["bg"].to(weight_dtype)
                        bg = bg[:, :actual_video_length, :, :]

                    if batch.get("fg") is not None:
                        fg = batch["fg"].to(weight_dtype)
                        fg = fg[:, :actual_video_length, :, :]

                    if batch.get("fg_coordmap") is not None:
                        fg_coordmap = batch["fg_coordmap"].to(weight_dtype)
                        fg_coordmap = fg_coordmap[:, :actual_video_length, :, :]

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    if clip_image_encoder is not None:
                        clip_image_encoder.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to("cpu")

                with torch.no_grad():
                    def augment_coordmap(coordmap_tensor, scale, translation):
                        """Perturb XYZ values while preserving the all-white invalid background."""
                        if coordmap_tensor.shape[2] < 3:
                            raise ValueError("Coordinate maps must have at least three channels")
                        result = coordmap_tensor.clone()
                        xyz = result[:, :, :3]
                        invalid = torch.isclose(
                            xyz,
                            torch.ones((), device=xyz.device, dtype=xyz.dtype),
                            atol=1e-4,
                            rtol=0.0,
                        ).all(dim=2, keepdim=True)
                        scale = torch.as_tensor(scale, device=xyz.device, dtype=xyz.dtype).view(1, 1, 1, 1, 1)
                        translation = torch.as_tensor(
                            translation, device=xyz.device, dtype=xyz.dtype
                        ).view(1, 1, 3, 1, 1)
                        perturbed_xyz = (xyz * scale + translation).clamp(-1.0, 1.0)
                        result[:, :, :3] = torch.where(invalid, xyz, perturbed_xyz)
                        return result

                    # Reference and target coordinate maps use the same sampled transform.
                    if rng is None:
                        apply_coordmap_aug = np.random.rand() < args.coordmap_aug_prob
                    else:
                        apply_coordmap_aug = rng.random() < args.coordmap_aug_prob

                    if apply_coordmap_aug:
                        if rng is None:
                            coordmap_scale = 1.0 + np.random.uniform(
                                -args.coordmap_aug_scale, args.coordmap_aug_scale
                            )
                            coordmap_translation = np.random.uniform(
                                -args.coordmap_aug_scale, args.coordmap_aug_scale, size=3
                            )
                        else:
                            coordmap_scale = 1.0 + rng.uniform(
                                -args.coordmap_aug_scale, args.coordmap_aug_scale
                            )
                            coordmap_translation = rng.uniform(
                                -args.coordmap_aug_scale, args.coordmap_aug_scale, size=3
                            )
                    else:
                        coordmap_scale = 1.0
                        coordmap_translation = np.zeros(3)

                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            pixel_values_bs = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = pixel_values_bs.sample()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)

                    # Encode main latents
                    if vae_stream_1 is not None:
                        vae_stream_1.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream_1):
                            latents = _batch_encode_vae(pixel_values)
                    else:
                        latents = _batch_encode_vae(pixel_values)

                    # Encode ref_pixel_values as full_ref
                    def _batch_encode_vae_ref(ref_pixel_values):
                        ref_pixel_values = rearrange(ref_pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_ref_latents = []
                        for i in range(0, ref_pixel_values.shape[0], bs):
                            ref_pixel_values_bs = ref_pixel_values[i : i + bs]
                            ref_latents_list = []
                            for frame_idx in range(ref_pixel_values_bs.shape[2]):
                                single_frame = ref_pixel_values_bs[:, :, frame_idx:frame_idx+1, :, :]
                                frame_latent = vae.encode(single_frame)[0]
                                frame_latent = frame_latent.sample()
                                ref_latents_list.append(frame_latent)
                            ref_latents_bs = torch.cat(ref_latents_list, dim=2)
                            new_ref_latents.append(ref_latents_bs)
                        return torch.cat(new_ref_latents, dim = 0)

                    ref_latents = _batch_encode_vae_ref(ref_pixel_values)

                    # Process full_ref
                    if ref_latents.size(2) == 1:
                        full_ref = ref_latents.squeeze(2)  # [B, C, H, W]
                    else:
                        full_ref = ref_latents  # [B, C, F, H, W]

                    # Encode ref_coordmap as ref_coordmap_latents
                    ref_coordmap_latents = None
                    if ref_coordmap is not None:
                        if apply_coordmap_aug:
                            ref_coordmap = augment_coordmap(
                                ref_coordmap, coordmap_scale, coordmap_translation
                            )

                        # Process ref_coordmap with the same logic as ref_pixel_values
                        def _batch_encode_vae_ref_coordmap(ref_coordmap_values):
                            ref_coordmap_values = rearrange(ref_coordmap_values, "b f c h w -> b c f h w")
                            bs = args.vae_mini_batch
                            new_ref_coordmap_latents = []
                            for i in range(0, ref_coordmap_values.shape[0], bs):
                                ref_coordmap_values_bs = ref_coordmap_values[i : i + bs]
                                ref_coordmap_latents_list = []
                                for frame_idx in range(ref_coordmap_values_bs.shape[2]):
                                    single_frame = ref_coordmap_values_bs[:, :, frame_idx:frame_idx+1, :, :]
                                    frame_latent = vae.encode(single_frame)[0]
                                    frame_latent = frame_latent.sample()
                                    ref_coordmap_latents_list.append(frame_latent)
                                ref_coordmap_latents_bs = torch.cat(ref_coordmap_latents_list, dim=2)
                                new_ref_coordmap_latents.append(ref_coordmap_latents_bs)
                            return torch.cat(new_ref_coordmap_latents, dim = 0)

                        ref_coordmap_latents = _batch_encode_vae_ref_coordmap(ref_coordmap)
                        if ref_coordmap_latents.size(2) == 1:
                            ref_coordmap_latents = ref_coordmap_latents.squeeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]


                # Mode 0: first_frame only; Mode 1: bgvideo; Mode 2: none
                empty_prob = 0.2
                if rng is None:
                    mode = np.random.choice([0, 1, 2, 3], p=[(1-empty_prob)/2, (1-empty_prob)/4, (1-empty_prob)/4, empty_prob])
                else:
                    mode = rng.choice([0, 1, 2, 3], p=[(1-empty_prob)/2, (1-empty_prob)/4, (1-empty_prob)/4, empty_prob])

                if mode == 3:
                    # Mode 3: empty
                    appearance_latents = torch.zeros_like(latents)
                elif mode == 2 and batch.get("bg") is not None:
                    # Mode 2: bgvideo
                    bg = batch["bg"].to(weight_dtype)
                    appearance_latents = _batch_encode_vae(bg)
                elif mode == 1 and batch.get("bg") is not None:
                    # Mode 1: first_frame and bgvideo
                    bg = batch["bg"].to(weight_dtype)
                    first_frame = bg[:, 0:1, :, :, :]  # [B, 1, C, H, W]
                    bg_rest = bg[:, 1:, :, :, :]  # [B, F-1, C, H, W]
                    combined = torch.cat([first_frame, bg_rest], dim=1)  # [B, F, C, H, W]
                    appearance_latents = _batch_encode_vae(combined)
                else:
                    # Mode 0: first_frame only
                    first_frame = pixel_values[:, 0:1, :, :, :]  # [B, 1, C, H, W]

                    start_image_latentes = _batch_encode_vae(first_frame)  # [B, 16, 1, H/8, W/8]
                    appearance_latents = torch.zeros_like(latents)
                    if latents.size()[2] != 1:
                        appearance_latents[:, :, :1] = start_image_latentes

                fg_coordmap_latent = None
                if batch.get("fg_coordmap") is not None:
                    fg_coordmap = batch["fg_coordmap"].to(weight_dtype)  # [B, F, C, H, W]

                    if apply_coordmap_aug:
                        fg_coordmap = augment_coordmap(
                            fg_coordmap, coordmap_scale, coordmap_translation
                        )

                    # Encode fg_coordmap
                    fg_coordmap_latent = _batch_encode_vae(fg_coordmap)

                    # Apply 10% dropout
                    for bs_index in range(fg_coordmap_latent.size()[0]):
                        if rng is None:
                            zero_init_fg_coordmap = np.random.choice([0, 1], p=[0.9, 0.1])
                        else:
                            zero_init_fg_coordmap = rng.choice([0, 1], p=[0.9, 0.1])

                        if zero_init_fg_coordmap:
                            fg_coordmap_latent[bs_index] = fg_coordmap_latent[bs_index] * 0

                            # following dropout of fg_coordmap_latent, zero out ref_coordmap_latents
                            if ref_coordmap_latents is not None:
                                if ref_coordmap_latents.dim() == 4:  # [B, C, H, W]
                                    ref_coordmap_latents[bs_index] = ref_coordmap_latents[bs_index] * 0
                                else:  # [B, C, F, H, W]
                                    ref_coordmap_latents[bs_index] = ref_coordmap_latents[bs_index] * 0

                # 10% dropout of full_ref and ref_coordmap_latents
                if full_ref is not None or ref_coordmap_latents is not None:
                    batch_size = full_ref.size()[0] if full_ref is not None else ref_coordmap_latents.size()[0]
                    for bs_index in range(batch_size):
                        if rng is None:
                            zero_init_ref = np.random.choice([0, 1], p=[0.90, 0.10])
                        else:
                            zero_init_ref = rng.choice([0, 1], p=[0.90, 0.10])

                        if zero_init_ref:
                            # Zero out full_ref for this sample
                            if full_ref is not None:
                                if full_ref.dim() == 4:  # [B, C, H, W]
                                    full_ref[bs_index] = full_ref[bs_index] * 0
                                else:  # [B, C, F, H, W]
                                    full_ref[bs_index] = full_ref[bs_index] * 0

                            # Zero out ref_coordmap_latents for this sample
                            if ref_coordmap_latents is not None:
                                if ref_coordmap_latents.dim() == 4:  # [B, C, H, W]
                                    ref_coordmap_latents[bs_index] = ref_coordmap_latents[bs_index] * 0
                                else:  # [B, C, F, H, W]
                                    ref_coordmap_latents[bs_index] = ref_coordmap_latents[bs_index] * 0

                # Downsample bg_mask to latent space
                bg_mask_downsampled = None
                if batch.get("bg_mask") is not None:
                    bg_mask = batch["bg_mask"].to(weight_dtype)  # [B, F, C, H, W]
                    bg_mask_single_channel = bg_mask[:, :, 0:1, :, :]  # [B, F, 1, H, W]
                    bg_mask_for_downsample = rearrange(bg_mask_single_channel, "b f c h w -> b c f h w")
                    _, _, latent_f, latent_h, latent_w = latents.shape  # Use latents shape
                    bg_mask_downsampled = torch.nn.functional.interpolate(
                        bg_mask_for_downsample,
                        size=(latent_f, latent_h, latent_w),
                        mode='nearest'
                    )  # [B, 1, latent_f, latent_h, latent_w]
                    bg_mask_downsampled = ((bg_mask_downsampled + 1.0) / 2.0 > 0.5).float()  # [B, 1, latent_f, latent_h, latent_w], values in {0, 1}

                # wait for latents = vae.encode(pixel_values) to complete
                if vae_stream_1 is not None:
                    torch.cuda.current_stream().wait_stream(vae_stream_1)

                if args.low_vram:
                    vae.to('cpu')
                    if clip_image_encoder is not None:
                        clip_image_encoder.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['encoder_hidden_states'].to(device=latents.device)
                else:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            batch['text'],
                            padding="max_length",
                            max_length=args.tokenizer_max_length,
                            truncation=True,
                            add_special_tokens=True,
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask

                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()

                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Sample a random timestep for each image
                    # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Add noise
                target = noise - latents

                target_shape = (vae.config.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                # Use pre-computed clip_fea template and expand to batch size
                if clip_fea_template is not None:
                    clip_fea = torch.zeros_like(clip_fea_template).expand(bsz, -1, -1)
                else:
                    clip_fea = None

                # Predict the noise residual
                with torch.amp.autocast("cuda", dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timesteps,
                        seq_len=seq_len,
                        clip_fea=clip_fea,  # clip features from dummy black image
                        full_ref=full_ref,
                        full_ref_crood=ref_coordmap_latents,
                        fg_coordmap=fg_coordmap_latent,
                        appearance=appearance_latents,
                    )

                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    final_loss = masked_loss.mean()
                    return final_loss

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                loss = loss.mean()

                # # Add ref_loss based on the bg_mask_downsampled region.
                # if bg_mask_downsampled is not None:
                #     ref_mse_loss = F.mse_loss(noise_pred.float(), target.float(), reduction='none')
                #     ref_masked_loss = ref_mse_loss * bg_mask_downsampled.float()
                #     if weighting is not None:
                #         ref_masked_loss = ref_masked_loss * weighting.float()
                #     # Average over the masked region
                #     mask_sum = bg_mask_downsampled.sum()
                #     if mask_sum > 0:
                #         ref_loss = ref_masked_loss.sum() / mask_sum
                #     else:
                #         ref_loss = torch.tensor(0.0, device=loss.device)
                #     # Add ref_loss to total loss
                #     loss = loss + ref_loss

                if args.motion_sub_loss and noise_pred.size()[1] > 2:
                    gt_sub_noise = noise_pred[:, :, 1:].float() - noise_pred[:, :, :-1].float()
                    pre_sub_noise = target[:, :, 1:].float() - target[:, :, :-1].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_deepspeed and not args.use_fsdp:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm

                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer3d.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # global_step is the number of completed optimizer updates.
                global_step += 1

                if args.use_ema:
                    ema_transformer3d.step(transformer3d.parameters())

                if (
                    global_step % args.checkpointing_steps == 0
                    and global_step < args.max_train_steps
                ):
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            # Only consider regular checkpoints, exclude validation checkpoints
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint") and not d.startswith("checkpoint-validation")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint, ignore_errors=True)

                    accelerator.wait_for_everyone()

                    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompts is not None and (
                    global_step % args.validation_steps == 0
                    or global_step == args.max_train_steps
                ):
                    if args.use_fsdp:
                        logger.info(
                            f"[Rank {accelerator.process_index}] running collective FSDP validation"
                        )
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            transformer3d,
                            args,
                            config,
                            accelerator,
                            weight_dtype,
                            global_step,
                            clip_image_encoder=clip_image_encoder,
                        )
                        transformer3d.train()
                        accelerator.wait_for_everyone()
                    elif zero_stage == 3:
                        if accelerator.is_main_process:
                            logger.info(
                                "DeepSpeed ZeRO-3 detected; saving a validation checkpoint "
                                "instead of running an unsharded validation pipeline."
                            )

                            # Remove old validation checkpoints before saving new one
                            existing_checkpoints = os.listdir(args.output_dir)
                            validation_checkpoints = [d for d in existing_checkpoints if d.startswith("checkpoint-validation-")]
                            for old_validation_checkpoint in validation_checkpoints:
                                old_validation_checkpoint_path = os.path.join(args.output_dir, old_validation_checkpoint)
                                logger.info(f"Removing old validation checkpoint: {old_validation_checkpoint}")
                                shutil.rmtree(old_validation_checkpoint_path, ignore_errors=True)

                        accelerator.wait_for_everyone()

                        if args.use_deepspeed or accelerator.is_main_process:
                            validation_save_path = os.path.join(args.output_dir, f"checkpoint-validation-{global_step}")
                            accelerator.save_state(validation_save_path)
                            logger.info(f"Saved validation checkpoint to {validation_save_path}")

                        accelerator.wait_for_everyone()
                    else:
                        # Run validation normally for non-FSDP and non-Zero3
                        if accelerator.is_main_process:
                            logger.info(f"Main process [Rank {accelerator.process_index}] is running validation...")

                            # test vae encode and decode
                            if global_step == 0:
                                vae_decoder = vae.eval()
                                # Ensure VAE is on the same device as latents for decoding
                                if args.low_vram:
                                    vae_decoder.to(accelerator.device)

                                gt_decoded = vae_decoder.decode(latents.to(vae_decoder.dtype)).sample
                                # Decode full_ref frame by frame
                                # Ensure full_ref is 5D [B, C, F, H, W]
                                if full_ref.dim() == 4:
                                    full_ref_for_decode = full_ref.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
                                else:
                                    full_ref_for_decode = full_ref

                                full_ref_decoded_frames = []
                                for frame_idx in range(full_ref_for_decode.shape[2]):
                                    frame = full_ref_for_decode[:, :, frame_idx:frame_idx+1, :, :]
                                    decoded_frame = vae_decoder.decode(frame.to(vae_decoder.dtype)).sample
                                    full_ref_decoded_frames.append(decoded_frame)
                                full_ref_decoded = torch.cat(full_ref_decoded_frames, dim=2)

                                # Decode fg_coordmap_latent if it exists
                                fg_decoded = None
                                if fg_coordmap_latent is not None:
                                    fg_decoded = vae_decoder.decode(fg_coordmap_latent.to(vae_decoder.dtype)).sample

                                # Move VAE back to CPU if low_vram mode is enabled
                                if args.low_vram:
                                    vae_decoder.to('cpu')
                                    torch.cuda.empty_cache()

                                # Normalize to [0, 1]
                                gt_decoded = (gt_decoded / 2 + 0.5).clamp(0, 1).cpu().float()
                                gt = (pixel_values / 2 + 0.5).clamp(0, 1).cpu().float().permute(0,2,1,3,4)
                                full_ref_decoded = (full_ref_decoded / 2 + 0.5).clamp(0, 1).cpu().float()

                                # Align frame counts for full_ref_decoded
                                if full_ref_decoded.shape[2] < gt.shape[2]:
                                    last_frame = full_ref_decoded[:, :, -1:, :, :]
                                    repeat_times = gt.shape[2] - full_ref_decoded.shape[2]
                                    repeated_frames = last_frame.repeat(1, 1, repeat_times, 1, 1)
                                    full_ref_decoded = torch.cat([full_ref_decoded, repeated_frames], dim=2)
                                else:
                                    full_ref_decoded = full_ref_decoded[:, :, :gt.shape[2], :, :]

                                # Build comparison list: [gt, gt_decoded, full_ref_decoded, ref_coordmap_decoded, fg_decoded, fg_coordmap_decoded, bg_decoded]
                                comparison_list = [gt.cpu().float(), gt_decoded, full_ref_decoded]

                                # Add ref_coordmap_decoded if exists
                                if ref_coordmap_latents is not None:
                                    # Decode ref_coordmap_latents
                                    ref_coordmap_decoded_frames = []
                                    if ref_coordmap_latents.dim() == 4:
                                        # Single frame: [B, C, H, W]
                                        ref_coordmap_decoded = vae_decoder.decode(ref_coordmap_latents.to(vae_decoder.dtype)).sample
                                        ref_coordmap_decoded = (ref_coordmap_decoded / 2 + 0.5).clamp(0, 1).cpu().float()
                                        # Align frame count with gt
                                        if ref_coordmap_decoded.shape[2] < gt.shape[2]:
                                            last_frame = ref_coordmap_decoded[:, :, -1:, :, :]
                                            repeat_times = gt.shape[2] - ref_coordmap_decoded.shape[2]
                                            repeated_frames = last_frame.repeat(1, 1, repeat_times, 1, 1)
                                            ref_coordmap_decoded = torch.cat([ref_coordmap_decoded, repeated_frames], dim=2)
                                        else:
                                            ref_coordmap_decoded = ref_coordmap_decoded[:, :, :gt.shape[2], :, :]
                                        comparison_list.append(ref_coordmap_decoded)
                                    else:
                                        # Multiple frames: [B, C, F, H, W]
                                        for frame_idx in range(ref_coordmap_latents.shape[2]):
                                            frame = ref_coordmap_latents[:, :, frame_idx:frame_idx+1, :, :]
                                            decoded_frame = vae_decoder.decode(frame.to(vae_decoder.dtype)).sample
                                            ref_coordmap_decoded_frames.append(decoded_frame)
                                        ref_coordmap_decoded = torch.cat(ref_coordmap_decoded_frames, dim=2)
                                        ref_coordmap_decoded = (ref_coordmap_decoded / 2 + 0.5).clamp(0, 1).cpu().float()
                                        # Align frame count with gt
                                        if ref_coordmap_decoded.shape[2] < gt.shape[2]:
                                            last_frame = ref_coordmap_decoded[:, :, -1:, :, :]
                                            repeat_times = gt.shape[2] - ref_coordmap_decoded.shape[2]
                                            repeated_frames = last_frame.repeat(1, 1, repeat_times, 1, 1)
                                            ref_coordmap_decoded = torch.cat([ref_coordmap_decoded, repeated_frames], dim=2)
                                        else:
                                            ref_coordmap_decoded = ref_coordmap_decoded[:, :, :gt.shape[2], :, :]
                                        comparison_list.append(ref_coordmap_decoded)

                                # Add fg_decoded if exists (decoded from fg_coordmap_latent)
                                if fg_decoded is not None:
                                    fg_decoded = (fg_decoded / 2 + 0.5).clamp(0, 1).cpu().float()
                                    comparison_list.append(fg_decoded)

                                # Add fg_coordmap_decoded if exists
                                if fg_coordmap_latent is not None:
                                    fg_coordmap_decoded = vae_decoder.decode(fg_coordmap_latent.to(vae_decoder.dtype)).sample
                                    fg_coordmap_decoded = (fg_coordmap_decoded / 2 + 0.5).clamp(0, 1).cpu().float()
                                    comparison_list.append(fg_coordmap_decoded)

                                # Add bg_decoded (appearance_latents decoded) if exists
                                if batch.get("bg") is not None:
                                    bg_decoded = vae_decoder.decode(appearance_latents.to(vae_decoder.dtype)).sample
                                    bg_decoded = (bg_decoded / 2 + 0.5).clamp(0, 1).cpu().float()
                                    comparison_list.append(bg_decoded)

                                # Stack vertically
                                comparison = torch.cat(comparison_list, dim=3)  # stack in height dimension
                                save_videos_grid(comparison, os.path.join(args.output_dir, f"validation/gt_vae.mp4"), fps=16)

                                # Prepare comparison for logging
                                log_vae_comparison = comparison.clone().detach().clamp(0, 1) * 255
                                log_vae_comparison = log_vae_comparison.permute(0, 2, 1, 3, 4).to(torch.uint8)  # [2, F, C, H, W]

                                log_dict = {}

                                if args.report_to == "wandb":
                                    log_dict["validation/vae_decoded"] = wandb.Video(log_vae_comparison.cpu(), fps=16, format="gif")
                                else: # Tensorboard
                                    log_dict["validation/vae_decoded"] = log_vae_comparison

                                accelerator.log(log_dict, step=global_step)

                            if args.use_ema:
                                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                                ema_transformer3d.store(transformer3d.parameters())
                                ema_transformer3d.copy_to(transformer3d.parameters())

                            log_validation(
                                vae,
                                text_encoder,
                                tokenizer,
                                transformer3d,
                                args,
                                config,
                                accelerator,
                                weight_dtype,
                                global_step,
                                clip_image_encoder=clip_image_encoder,
                            )
                            transformer3d.train()

                            if args.use_ema:
                                # Switch back to the original transformer3d parameters.
                                ema_transformer3d.restore(transformer3d.parameters())

                        logger.info(f"[Rank {accelerator.process_index}] Syncing after validation step...")
                        accelerator.wait_for_everyone()

                progress_bar.update(1)

                current_lr = lr_scheduler.get_last_lr()[0]
                log_dict = {
                    "train/loss": train_loss,
                    "train/learning_rate": current_lr
                }

                accelerator.log(log_dict, step=global_step)

                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "num_ref": num_ref_frames_to_use}
            progress_bar.set_postfix(**logs)

            # Finalize memory profiling if this was a profiled step
            if profiler_ctx is not None:
                rank = accelerator.process_index
                local_device = accelerator.local_process_index
                try:
                    # Stop profiler and export trace
                    profiler_ctx.__exit__(None, None, None)
                    logger.info(f"[Rank {rank}] PyTorch Profiler trace saved for step {global_step}")

                    # Export memory snapshot for this GPU
                    export_memory_snapshot(global_step, args.output_dir, rank=rank)

                    # Stop memory history recording
                    torch.cuda.memory._record_memory_history(enabled=None)

                    # Log memory stats to wandb (each rank logs its own GPU stats)
                    memory_stats = get_memory_stats_for_logging(device_id=local_device, rank=rank)
                    accelerator.log(memory_stats, step=global_step)
                    logger.info(f"[Rank {rank}] Memory profiling completed for step {global_step}")
                except Exception as e:
                    logger.warning(f"[Rank {rank}] Error finalizing memory profiler: {e}")
                profiler_ctx = None

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer3d = unwrap_model(transformer3d)
        if args.use_ema:
            ema_transformer3d.copy_to(transformer3d.parameters())

    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
