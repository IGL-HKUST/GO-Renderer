import csv
import gc
import io
import json
import math
import os
import random
from contextlib import contextmanager
from pathlib import Path
from random import shuffle
from threading import Thread

# Avoid a network version check when importing training CLIs in offline pods.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from packaging import version as pver
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset

from .validation import IMAGE_EXTENSIONS, MediaInfo, inspect_media

VIDEO_READER_TIMEOUT = 20

def padding_image(images, new_width, new_height):
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    aspect_ratio = images.width / images.height
    if new_width / new_height > 1:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)
    else:
        if aspect_ratio > new_width / new_height:
            new_img_width = new_width
            new_img_height = int(new_img_width / aspect_ratio)
        else:
            new_img_height = new_height
            new_img_width = int(new_img_height * aspect_ratio)

    resized_img = images.resize((new_img_width, new_img_height))

    paste_x = (new_width - new_img_width) // 2
    paste_y = (new_height - new_img_height) // 2

    new_image.paste(resized_img, (paste_x, paste_y))

    return new_image

def get_image_resize(ref_image=None, sample_size=None, padding=False):
    if ref_image is not None:
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")
            if padding:
                ref_image = padding_image(ref_image, sample_size[1], sample_size[0])
            ref_image = ref_image.resize((sample_size[1], sample_size[0]))
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
        else:
            ref_image = torch.from_numpy(np.array(ref_image))
            ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

    return ref_image

def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05])
        else:
            mask_index = np.random.choice([0, 1], p = [0.2, 0.8])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

            for i in range(h):
                for j in range(w):
                    if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                        mask[:, :, i, j] = 1
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            for i in range(h):
                for j in range(w):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                        mask[:, :, i, j] = 1
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not define")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask

class ImageVideoSampler(BatchSampler):
    """Group image and video samples into separate batches.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if len(self.sampler) != len(self.dataset):
            raise ValueError(
                "ImageVideoSampler requires a full-dataset sampler so its batch "
                "count remains exact"
            )
        if getattr(self.sampler, "replacement", False):
            raise ValueError("ImageVideoSampler does not support replacement sampling")
        self._type_counts = {"image": 0, "video": 0}
        for index, entry in enumerate(self.dataset.dataset):
            content_type = entry.get("type", "image")
            if content_type not in self._type_counts:
                raise ValueError(
                    f"Dataset sample {index} has unsupported type: {content_type!r}"
                )
            self._type_counts[content_type] += 1

    def __iter__(self):
        buckets = {"image": [], "video": []}
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            if content_type not in buckets:
                raise ValueError(
                    f"Dataset sample {idx} has unsupported type: {content_type!r}"
                )
            buckets[content_type].append(idx)

            # Yield a batch containing one media type. Buckets are local to this
            # iterator so incomplete batches can never leak into the next epoch.
            if len(buckets['video']) == self.batch_size:
                bucket = buckets['video']
                yield bucket[:]
                del bucket[:]
            elif len(buckets['image']) == self.batch_size:
                bucket = buckets['image']
                yield bucket[:]
                del bucket[:]

        if not self.drop_last:
            for content_type in ("video", "image"):
                if buckets[content_type]:
                    yield buckets[content_type][:]

    def __len__(self):
        if self.drop_last:
            return sum(
                count // self.batch_size for count in self._type_counts.values()
            )
        return sum(
            math.ceil(count / self.batch_size)
            for count in self._type_counts.values()
            if count
        )

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)

    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

class ImageVideoRefDataset(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0,
        video_length_drop_end=1.0,
        enable_inpaint=False,
        return_file_name=False,
        num_ref_frames=None,
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))

        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'image') != 'video':
                    self.dataset.append(data)

            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.num_ref_frames = num_ref_frames
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ]
            )

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.image_minor_side = min(self.image_sample_size)
        self.video_minor_side = min(self.video_sample_size)
        # Dataset instances are copied into DataLoader workers, so this cache is
        # naturally local to each worker and avoids probing repeated paths again.
        self._media_info_cache: dict[Path, MediaInfo] = {}

    def _resolve_media_path(self, value):
        path = Path(value).expanduser()
        if not path.is_absolute() and self.data_root is not None:
            path = Path(self.data_root).expanduser() / path
        return path.resolve()

    def _inspect_media_cached(self, path):
        resolved_path = self._resolve_media_path(path)
        if resolved_path not in self._media_info_cache:
            self._media_info_cache[resolved_path] = inspect_media(resolved_path)
        return self._media_info_cache[resolved_path]

    @staticmethod
    def _media_container_type(path: Path) -> str:
        if path.is_dir():
            return "image directory"
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return "image"
        return "video"

    def _validate_media_alignment(self, idx, data_info):
        """Fail fast when a required training carrier is missing or misaligned."""

        data_type = data_info.get("type", "image")
        if data_type not in {"image", "video"}:
            raise ValueError(f"Dataset sample {idx} has unsupported type: {data_type!r}")

        required_paths = {}
        for field in ("file_path", "ref", "ref_coordmap", "fg_coordmap"):
            value = data_info.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Dataset sample {idx} requires a non-empty '{field}' media path"
                )
            required_paths[field] = value

        ref_path = self._resolve_media_path(required_paths["ref"])
        ref_coordmap_path = self._resolve_media_path(required_paths["ref_coordmap"])
        ref_container = self._media_container_type(ref_path)
        ref_coordmap_container = self._media_container_type(ref_coordmap_path)
        if ref_container != ref_coordmap_container:
            raise ValueError(
                f"Dataset sample {idx} mixes reference container types: ref is "
                f"{ref_container}, but ref_coordmap is {ref_coordmap_container}. "
                "Use matching containers so both receive the same geometry transform."
            )

        try:
            target_info = self._inspect_media_cached(required_paths["file_path"])
            ref_info = self._inspect_media_cached(required_paths["ref"])
            ref_coordmap_info = self._inspect_media_cached(required_paths["ref_coordmap"])
            target_coordmap_info = self._inspect_media_cached(required_paths["fg_coordmap"])
        except Exception as exc:
            raise ValueError(f"Dataset sample {idx} contains unreadable media: {exc}") from exc

        if ref_info.frames < 3:
            raise ValueError(
                f"Dataset sample {idx} has {ref_info.frames} reference frames; at least 3 are required"
            )
        if ref_info.frames != ref_coordmap_info.frames:
            raise ValueError(
                f"Dataset sample {idx} has {ref_info.frames} reference frames but "
                f"{ref_coordmap_info.frames} reference coordinate-map frames"
            )
        if (ref_info.width, ref_info.height) != (
            ref_coordmap_info.width,
            ref_coordmap_info.height,
        ):
            raise ValueError(
                f"Dataset sample {idx} reference dimensions "
                f"{ref_info.width}x{ref_info.height} do not match reference coordinate-map "
                f"dimensions {ref_coordmap_info.width}x{ref_coordmap_info.height}"
            )
        if target_coordmap_info.frames < target_info.frames:
            raise ValueError(
                f"Dataset sample {idx} has {target_info.frames} target frames but only "
                f"{target_coordmap_info.frames} target coordinate-map frames"
            )
        if (target_info.width, target_info.height) != (
            target_coordmap_info.width,
            target_coordmap_info.height,
        ):
            raise ValueError(
                f"Dataset sample {idx} target dimensions "
                f"{target_info.width}x{target_info.height} do not match target coordinate-map "
                f"dimensions {target_coordmap_info.width}x{target_coordmap_info.height}"
            )

    def _read_video_frames(self, video_path, batch_index, idx, apply_transforms=True, is_mask=False):
        """
        Unified video reading function.

        Args:
            video_path: Path to the video file.
            batch_index: Indices of the frames to read.
            idx: Index in the dataset (for error reporting).
            apply_transforms: Whether to apply video_transforms.
            is_mask: Whether this is a mask video (requires inversion and binarization).

        Returns:
            Processed video frame data.
        """
        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            try:
                sample_args = (video_reader, batch_index)
                pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
                resized_frames = []
                for i in range(len(pixel_values)):
                    frame = pixel_values[i]
                    resized_frame = resize_frame(frame, self.video_minor_side)
                    resized_frames.append(resized_frame)
                pixel_values = np.array(resized_frames)
            except FunctionTimedOut:
                raise ValueError(f"Read video {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            # invert mask from blender output
            if is_mask:
                pixel_values = 255 - pixel_values
                pixel_values = (pixel_values > 127.5).astype(np.float32) * 255.0

            if not self.enable_bucket:
                pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.
                del video_reader
            else:
                pixel_values = pixel_values

            if not self.enable_bucket and apply_transforms:
                pixel_values = self.video_transforms(pixel_values)

            return pixel_values

    def _ref_preprocess(self, ref_file_path, idx, data_type='image', target_size=None):
        """
        Process reference file which can be: image, video, or directory of images.
        Returns all ref_pixel_values without sampling (sampling will be done in training script).

        Args:
            target_size: Optional tuple (height, width) to resize to. If provided, uses this size directly.

        Returns:
            ref_pixel_values: processed reference frames (all frames, no sampling)
        """
        # Get full path to ref file
        if self.data_root is None:
            ref_file_id = ref_file_path
        else:
            ref_file_id = os.path.join(self.data_root, ref_file_path)

        # Select appropriate size based on data type or use target_size if provided
        if target_size is not None:
            minor_side = min(target_size)
        else:
            minor_side = self.image_minor_side if data_type == 'image' else self.video_minor_side

        # Check if ref_file_id is a directory
        if os.path.isdir(ref_file_id):
            # Load all images from the directory
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_files = []
            for file in sorted(os.listdir(ref_file_id)):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(ref_file_id, file))

            if len(image_files) == 0:
                raise ValueError(f"No image files found in directory: {ref_file_id}")

            # Load all images (no sampling)
            ref_frames_list = []
            for img_path in image_files:
                if not self.enable_bucket:
                    ref_image = Image.open(img_path).convert('RGB')
                    ref_frame = self.image_transforms(ref_image)
                    ref_frames_list.append(ref_frame)
                else:
                    # If target_size is provided, use it directly; otherwise compute from aspect ratio
                    if target_size is not None:
                        new_height, new_width = target_size
                    else:
                        # Load image to get original aspect ratio
                        ref_image_pil = Image.open(img_path).convert('RGB')
                        orig_width, orig_height = ref_image_pil.size
                        aspect_ratio = orig_width / orig_height
                        if orig_width > orig_height:
                            new_width = minor_side
                            new_height = int(new_width / aspect_ratio)
                        else:
                            new_height = minor_side
                            new_width = int(new_height * aspect_ratio)

                    ref_frame = get_image_resize(
                        ref_image=img_path,
                        sample_size=[new_height, new_width],
                        padding=True
                    )
                    ref_frame = ref_frame.squeeze(0).squeeze(1)
                    # Convert back to numpy [H, W, C] for bucket mode
                    ref_frame = ref_frame.permute(1, 2, 0).numpy() * 255
                    ref_frames_list.append(ref_frame.astype(np.uint8))

            if not self.enable_bucket:
                return torch.stack(ref_frames_list)  # [F, C, H, W]
            else:
                ref_frames_torch = [torch.from_numpy(frame).permute(2, 0, 1) for frame in ref_frames_list]
                return torch.stack(ref_frames_torch)  # [F, C, H, W]
        else:
            # Check if ref file is image or video by extension
            ref_file_ext = ref_file_path.lower().split('.')[-1]
            if ref_file_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
                # Process as image - single frame
                ref_image = Image.open(ref_file_id).convert('RGB')
                if not self.enable_bucket:
                    return self.image_transforms(ref_image).unsqueeze(0)  # Add frame dimension
                else:
                    return np.expand_dims(np.array(ref_image), 0)  # Add frame dimension
            else:
                # Process as video - load all frames (no sampling)
                with VideoReader_contextmanager(ref_file_id, num_threads=2) as ref_video_reader:
                    total_frames = len(ref_video_reader)
                    # Load all frames
                    ref_frames_indices = list(range(total_frames))

                    try:
                        sample_args = (ref_video_reader, ref_frames_indices)
                        ref_frames = func_timeout(
                            VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                        )

                        # Process each frame as image
                        ref_frames_list = []
                        for i in range(len(ref_frames)):
                            frame = ref_frames[i]
                            # Use target_size if provided for exact sizing
                            if target_size is not None:
                                resized_frame = resize_frame(frame, minor_side)
                                frame_pil = Image.fromarray(resized_frame)
                                # Resize to exact target_size
                                frame_pil = frame_pil.resize((target_size[1], target_size[0]), Image.BILINEAR)
                            else:
                                resized_frame = resize_frame(frame, minor_side)
                                frame_pil = Image.fromarray(resized_frame)

                            if not self.enable_bucket:
                                ref_frame = self.image_transforms(frame_pil)
                                ref_frames_list.append(ref_frame)
                            else:
                                ref_frames_list.append(np.array(frame_pil))

                        if not self.enable_bucket:
                            return torch.stack(ref_frames_list)  # [F, C, H, W]
                        else:
                            ref_frames_torch = [torch.from_numpy(frame).permute(2, 0, 1) for frame in ref_frames_list]
                            return torch.stack(ref_frames_torch)  # [F, C, H, W]

                    except FunctionTimedOut:
                        raise ValueError(f"Read {idx} timeout.")
                    except Exception as e:
                        raise ValueError(f"Failed to extract frames from ref video. Error is {e}.")

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        video_id, text = data_info['file_path'], data_info['text']

        # video
        if data_info.get('type', 'image') == 'video':
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, video_id)

            # Calculate batch_index for frame sampling
            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames,
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            # Read pixel_values using unified function
            pixel_values = self._read_video_frames(video_dir, batch_index, idx, apply_transforms=True, is_mask=False)

            # Random use no text generation
            if random.random() < self.text_drop_ratio:
                text = ''

            # Process ref file (can be video, image, or directory)
            ref_file_path = data_info.get('ref', '')

            # Handle empty ref: use first frame of gt video
            if not ref_file_path or ref_file_path.strip() == '':
                # Use first frame of gt video as ref
                if not self.enable_bucket:
                    ref_pixel_values = pixel_values[0:1]  # Take first frame, keep frame dimension
                else:
                    ref_pixel_values = np.expand_dims(pixel_values[0], 0)  # Add frame dimension
            else:
                # Get target size from pixel_values first frame
                if not self.enable_bucket:
                    target_size = (pixel_values.shape[2], pixel_values.shape[3])  # (H, W) from (F, C, H, W)
                else:
                    target_size = (pixel_values.shape[1], pixel_values.shape[2])  # (H, W) from (F, H, W, C)
                ref_pixel_values = self._ref_preprocess(ref_file_path, idx, data_type='video', target_size=target_size)

            bg_mask = None
            if 'mask' in data_info and data_info['mask']:
                mask_file_path = data_info['mask']
                if self.data_root is None:
                    mask_video_dir = mask_file_path
                else:
                    mask_video_dir = os.path.join(self.data_root, mask_file_path)

                # Use unified video reading function with mask processing
                bg_mask = self._read_video_frames(mask_video_dir, batch_index, idx, apply_transforms=True, is_mask=True)

            bg = None
            if 'bg' in data_info and data_info['bg']:
                bg_file_path = data_info['bg']
                if self.data_root is None:
                    bg_video_dir = bg_file_path
                else:
                    bg_video_dir = os.path.join(self.data_root, bg_file_path)

                # A supplied condition must be readable; silently dropping it changes
                # the training sample rather than repairing it.
                bg = self._read_video_frames(
                    bg_video_dir,
                    batch_index,
                    idx,
                    apply_transforms=True,
                    is_mask=False,
                )

            fg = None
            if 'fg' in data_info and data_info['fg']:
                fg_file_path = data_info['fg']
                if self.data_root is None:
                    fg_video_dir = fg_file_path
                else:
                    fg_video_dir = os.path.join(self.data_root, fg_file_path)

                # Use unified video reading function
                fg = self._read_video_frames(fg_video_dir, batch_index, idx, apply_transforms=True, is_mask=False)

            # Load pose data if available
            ref_pose = None
            video_pose = None
            if 'pose' in data_info and data_info['pose']:
                pose_file_path = data_info['pose']
                if self.data_root is None:
                    pose_file_full_path = pose_file_path
                else:
                    pose_file_full_path = os.path.join(self.data_root, pose_file_path)

                try:
                    with open(pose_file_full_path, 'r') as f:
                        pose_data = json.load(f)

                    # Process video pose
                    if 'video' in pose_data:
                        video_pose_list = []
                        for frame_idx in batch_index:
                            if frame_idx < len(pose_data['video']):
                                pose_matrix = pose_dict_to_matrix(pose_data['video'][frame_idx])
                                video_pose_list.append(pose_matrix)
                        if video_pose_list:
                            video_pose = np.stack(video_pose_list, axis=0)  # [F, 4, 4]

                    # Process ref pose - load all ref pose frames (sampling will be done in training script)
                    # Check if ref is empty (using first frame of video as ref)
                    if not ref_file_path or ref_file_path.strip() == '':
                        # Use first frame of video pose as ref pose
                        if video_pose is not None:
                            ref_pose = video_pose[0:1]  # [1, 4, 4]
                    else:
                        # Load all ref pose frames from pose file (no sampling)
                        if 'ref' in pose_data:
                            ref_pose_list = []
                            for frame_idx in range(len(pose_data['ref'])):
                                pose_matrix = pose_dict_to_matrix(pose_data['ref'][frame_idx])
                                ref_pose_list.append(pose_matrix)
                            if ref_pose_list:
                                ref_pose = np.stack(ref_pose_list, axis=0)  # [F, 4, 4]

                except Exception as e:
                    print(f"Warning: Failed to load pose from {pose_file_full_path}: {e}")
                    ref_pose = None
                    video_pose = None

            # Load ref_coordmap if available
            ref_coordmap_file_path = data_info['ref_coordmap']
            # Use _ref_preprocess to load all frames with the same sizing path as ref.
            if not self.enable_bucket:
                target_size = (pixel_values.shape[2], pixel_values.shape[3])
            else:
                target_size = (pixel_values.shape[1], pixel_values.shape[2])
            ref_coordmap = self._ref_preprocess(
                ref_coordmap_file_path,
                idx,
                data_type='video',
                target_size=target_size,
            )

            # Load fg_coordmap if available
            fg_coordmap_file_path = data_info['fg_coordmap']
            if self.data_root is None:
                fg_coordmap_video_dir = fg_coordmap_file_path
            else:
                fg_coordmap_video_dir = os.path.join(self.data_root, fg_coordmap_file_path)
            fg_coordmap = self._read_video_frames(
                fg_coordmap_video_dir,
                batch_index,
                idx,
                apply_transforms=True,
                is_mask=False,
            )

            return pixel_values, ref_pixel_values, text, "video", video_dir, bg_mask, bg, fg, ref_pose, video_pose, ref_coordmap, fg_coordmap

        # image
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                pixel_values = self.image_transforms(image).unsqueeze(0)
            else:
                pixel_values = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            # Process ref file (can be video, image, or directory)
            ref_file_path = data_info.get('ref', '')

            # Handle empty ref: use gt image as ref
            if not ref_file_path or ref_file_path.strip() == '':
                # Use gt image as ref
                ref_pixel_values = pixel_values.clone() if not self.enable_bucket else pixel_values.copy()
            else:
                # Get target size from pixel_values
                if not self.enable_bucket:
                    target_size = (pixel_values.shape[2], pixel_values.shape[3])  # (H, W) from (F, C, H, W)
                else:
                    target_size = (pixel_values.shape[1], pixel_values.shape[2])  # (H, W) from (F, H, W, C)
                ref_pixel_values = self._ref_preprocess(ref_file_path, idx, data_type='image', target_size=target_size)

            bg_mask = None
            if 'mask' in data_info and data_info['mask']:
                mask_file_path = data_info['mask']
                if self.data_root is not None:
                    mask_file_path = os.path.join(self.data_root, mask_file_path)
                mask_image = Image.open(mask_file_path).convert('RGB')
                if not self.enable_bucket:
                    bg_mask = self.image_transforms(mask_image).unsqueeze(0)
                else:
                    bg_mask = np.expand_dims(np.array(mask_image), 0)

                # Apply invert and binarization to mask
                if not self.enable_bucket:
                    # bg_mask shape: [1, C, H, W], values in [-1, 1]
                    # 1. Invert: -x
                    bg_mask = -bg_mask
                    # 2. Binarize: convert to [0, 1], then threshold at 0.5
                    bg_mask = (bg_mask + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                    bg_mask = (bg_mask > 0.5).float()
                    # Convert back to [-1, 1] range
                    bg_mask = bg_mask * 2.0 - 1.0
                else:
                    # bg_mask shape: [1, H, W, C], values in [0, 255]
                    # 1. Invert: 255 - x
                    bg_mask = 255 - bg_mask
                    # 2. Binarize: threshold at 127.5
                    bg_mask = (bg_mask > 127.5).astype(np.float32) * 255.0

            bg = None
            if 'bg' in data_info and data_info['bg']:
                bg_file_path = data_info['bg']
                if self.data_root is not None:
                    bg_file_path = os.path.join(self.data_root, bg_file_path)
                bg_image = Image.open(bg_file_path).convert('RGB')
                if not self.enable_bucket:
                    bg = self.image_transforms(bg_image).unsqueeze(0)
                else:
                    bg = np.expand_dims(np.array(bg_image), 0)

            fg = None
            if 'fg' in data_info and data_info['fg']:
                fg_file_path = data_info['fg']
                if self.data_root is not None:
                    fg_file_path = os.path.join(self.data_root, fg_file_path)
                fg_image = Image.open(fg_file_path).convert('RGB')
                if not self.enable_bucket:
                    fg = self.image_transforms(fg_image).unsqueeze(0)
                else:
                    fg = np.expand_dims(np.array(fg_image), 0)

            # Load ref_coordmap if available
            ref_coordmap_file_path = data_info['ref_coordmap']
            if not self.enable_bucket:
                target_size = (pixel_values.shape[2], pixel_values.shape[3])
            else:
                target_size = (pixel_values.shape[1], pixel_values.shape[2])
            ref_coordmap = self._ref_preprocess(
                ref_coordmap_file_path,
                idx,
                data_type='video',
                target_size=target_size,
            )

            # Load fg_coordmap if available
            fg_coordmap_file_path = data_info['fg_coordmap']
            if self.data_root is not None:
                fg_coordmap_file_path = os.path.join(self.data_root, fg_coordmap_file_path)

            fg_coordmap_image = Image.open(fg_coordmap_file_path).convert('RGB')

            if not self.enable_bucket:
                fg_coordmap = self.image_transforms(fg_coordmap_image).unsqueeze(0)
            else:
                fg_coordmap = np.expand_dims(np.array(fg_coordmap_image), 0)


            # For images, pose is not applicable
            ref_pose = None
            video_pose = None

            return pixel_values, ref_pixel_values, text, 'image', image_path, bg_mask, bg, fg, ref_pose, video_pose, ref_coordmap, fg_coordmap

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')
        self._validate_media_alignment(idx, data_info)
        sample = {}

        pixel_values, ref_pixel_values, name, data_type, file_path, bg_mask, bg, fg, ref_pose, video_pose, ref_coordmap, fg_coordmap = self.get_batch(idx)
        if ref_coordmap is None or fg_coordmap is None:
            raise ValueError(
                f"Dataset sample {idx} did not produce the required coordinate-map conditions"
            )
        if ref_pixel_values.shape[0] != ref_coordmap.shape[0]:
            raise ValueError(
                f"Dataset sample {idx} decoded {ref_pixel_values.shape[0]} reference frames but "
                f"{ref_coordmap.shape[0]} reference coordinate-map frames"
            )
        if ref_pixel_values.shape[0] < 3:
            raise ValueError(
                f"Dataset sample {idx} decoded {ref_pixel_values.shape[0]} reference frames; "
                "at least 3 are required"
            )
        if pixel_values.shape[0] != fg_coordmap.shape[0]:
            raise ValueError(
                f"Dataset sample {idx} decoded {pixel_values.shape[0]} target frames but "
                f"{fg_coordmap.shape[0]} target coordinate-map frames"
            )

        # Randomly shuffle frames of ref and its aligned coordinate map together.
        if ref_pixel_values.shape[0] > 1:
            perm = torch.randperm(ref_pixel_values.shape[0])
            ref_pixel_values = ref_pixel_values[perm]
            ref_coordmap = ref_coordmap[perm]
            if ref_pose is not None:
                ref_pose = ref_pose[perm.numpy()]

        # 10% probability to reverse the main video sequence only.
        # Keep ref/ref_coordmap unchanged because they are independent reference carriers.
        if data_type == "video" and pixel_values.shape[0] > 1 and random.random() < 0.1:
            def reverse_frames(frames):
                if isinstance(frames, torch.Tensor):
                    return frames.flip(0)
                return np.flip(frames, axis=0).copy()

            pixel_values = reverse_frames(pixel_values)
            if bg_mask is not None:
                bg_mask = reverse_frames(bg_mask)
            if bg is not None:
                bg = reverse_frames(bg)
            if fg is not None:
                fg = reverse_frames(fg)
            fg_coordmap = reverse_frames(fg_coordmap)
            if video_pose is not None:
                video_pose = reverse_frames(video_pose)

        sample["pixel_values"] = pixel_values
        sample["ref_pixel_values"] = ref_pixel_values
        sample["text"] = name
        sample["data_type"] = data_type
        sample["idx"] = idx

        if bg_mask is not None:
            sample["bg_mask"] = bg_mask
        if bg is not None:
            sample["bg"] = bg
        if fg is not None:
            sample["fg"] = fg
        if ref_pose is not None:
            sample["ref_pose"] = ref_pose
        if video_pose is not None:
            sample["video_pose"] = video_pose
        sample["ref_coordmap"] = ref_coordmap
        sample["fg_coordmap"] = fg_coordmap

        if self.return_file_name:
            sample["file_name"] = os.path.basename(file_path)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample
