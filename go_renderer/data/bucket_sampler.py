# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import (Generic, Iterable, Iterator, List, Optional, Sequence,
                    Sized, TypeVar, Union)

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset, Sampler
from decord import VideoReader

ASPECT_RATIO_512 = {
    '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
    '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
    '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
    '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
    '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
    '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
    '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
    '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
    '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
    '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
}
ASPECT_RATIO_RANDOM_CROP_512 = {
    '0.42': [320.0, 768.0], '0.5': [352.0, 704.0],
    '0.57': [384.0, 672.0], '0.68': [416.0, 608.0], '0.78': [448.0, 576.0], '0.88': [480.0, 544.0],
    '0.94': [480.0, 512.0], '1.0': [512.0, 512.0], '1.07': [512.0, 480.0],
    '1.13': [544.0, 480.0], '1.29': [576.0, 448.0], '1.46': [608.0, 416.0], '1.75': [672.0, 384.0],
    '2.0': [704.0, 352.0],  '2.4': [768.0, 320.0]
}
ASPECT_RATIO_RANDOM_CROP_PROB = [
    1, 2,
    4, 4, 4, 4,
    8, 8, 8,
    4, 4, 4, 4,
    2, 1
]
ASPECT_RATIO_RANDOM_CROP_PROB = np.array(ASPECT_RATIO_RANDOM_CROP_PROB) / sum(ASPECT_RATIO_RANDOM_CROP_PROB)

# 480 * 832 像素量约 400,000
ASPECT_RATIO_480 = {
    '0.25': [320.0, 1280.0], '0.26': [320.0, 1248.0], '0.28': [352.0, 1248.0], '0.3': [352.0, 1184.0],
    '0.32': [352.0, 1120.0], '0.35': [384.0, 1088.0], '0.38': [384.0, 1024.0], '0.41': [416.0, 1024.0],
    '0.43': [416.0, 960.0],  '0.47': [448.0, 960.0],  '0.5': [448.0, 896.0],  '0.54': [480.0, 896.0],
    '0.58': [480.0, 832.0],  # 16:9 横屏
    '0.62': [512.0, 832.0],  '0.67': [512.0, 768.0],  '0.71': [544.0, 768.0],
    '0.74': [544.0, 736.0],  # 4:3 横屏
    '0.78': [576.0, 736.0],  '0.83': [576.0, 672.0],  '0.89': [608.0, 672.0], '0.95': [608.0, 640.0],
    '1.0':  [640.0, 640.0],   # 1:1 正方形
    '1.05': [640.0, 608.0],  '1.12': [672.0, 608.0],  '1.17': [672.0, 576.0], '1.25': [704.0, 576.0],
    '1.31': [704.0, 544.0],
    '1.35': [736.0, 544.0],  # 3:4 竖屏
    '1.41': [736.0, 512.0],  '1.5': [768.0, 512.0],   '1.6': [768.0, 480.0],
    '1.73': [832.0, 480.0],  # 9:16 竖屏
    '1.85': [832.0, 448.0],  '2.0': [896.0, 448.0],   '2.14': [896.0, 416.0], '2.31': [960.0, 416.0],
    '2.5':  [960.0, 384.0],  '2.67': [1024.0, 384.0], '3.0': [1088.0, 352.0], '3.38': [1184.0, 352.0],
    '3.75': [1248.0, 320.0], '4.0': [1280.0, 320.0]
}

ASPECT_RATIO_RANDOM_CROP_480 = {
    '0.43': [416.0, 960.0], '0.5': [448.0, 896.0], '0.58': [480.0, 832.0],
    '0.67': [512.0, 768.0], '0.74': [544.0, 736.0], '0.83': [576.0, 672.0],
    '0.95': [608.0, 640.0], '1.0': [640.0, 640.0],  '1.05': [640.0, 608.0],
    '1.17': [672.0, 576.0], '1.35': [736.0, 544.0], '1.5': [768.0, 512.0],
    '1.73': [832.0, 480.0], '2.0': [896.0, 448.0], '2.31': [960.0, 416.0]
}

# 采样概率 (16:9, 4:3, 1:1 权重设为 8)
ASPECT_RATIO_RANDOM_CROP_PROB_480 = [
    1, 2, 8,
    4, 8, 4,
    4, 8, 4,
    4, 8, 4,
    8, 2, 1
]
ASPECT_RATIO_RANDOM_CROP_PROB_480 = np.array(ASPECT_RATIO_RANDOM_CROP_PROB_480) / sum(ASPECT_RATIO_RANDOM_CROP_PROB_480)

def get_closest_ratio(height: float, width: float, ratios: dict = ASPECT_RATIO_512):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

def get_image_size_without_loading(path):
    with Image.open(path) as img:
        return img.size  # (width, height)

class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self._pos_start = 0

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                xx = torch.randperm(n, generator=generator).tolist()
                if self._pos_start >= n:
                    self._pos_start = 0
                for idx in range(self._pos_start, n):
                    yield xx[idx]
                    self._pos_start = (self._pos_start + 1) % n
                self._pos_start = 0
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples

class AspectRatioBatchImageSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        train_folder: str = None,
        aspect_ratios: dict = ASPECT_RATIO_512,
        drop_last: bool = False,
        config=None,
        **kwargs
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.config = config
        # buckets for each aspect ratio
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios}
        # [str(k) for k, v in aspect_ratios]
        self.current_available_bucket_keys = list(aspect_ratios.keys())

    def __iter__(self):
        for idx in self.sampler:
            try:
                image_dict = self.dataset[idx]

                width, height = image_dict.get("width", None), image_dict.get("height", None)
                if width is None or height is None:
                    image_id, name = image_dict['file_path'], image_dict['text']
                    if self.train_folder is None:
                        image_dir = image_id
                    else:
                        image_dir = os.path.join(self.train_folder, image_id)

                    width, height = get_image_size_without_loading(image_dir)

                    ratio = height / width # self.dataset[idx]
                else:
                    height = int(height)
                    width = int(width)
                    ratio = height / width # self.dataset[idx]
            except Exception as e:
                print(e)
                continue
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        video_folder: str = None,
        train_data_format: str = "webvid",
        aspect_ratios: dict = ASPECT_RATIO_512,
        drop_last: bool = False,
        config=None,
        **kwargs
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.video_folder = video_folder
        self.train_data_format = train_data_format
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.config = config
        # buckets for each aspect ratio
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios}
        # [str(k) for k, v in aspect_ratios]
        self.current_available_bucket_keys = list(aspect_ratios.keys())

    def __iter__(self):
        for idx in self.sampler:
            try:
                video_dict = self.dataset[idx]
                width, height = video_dict.get("width", None), video_dict.get("height", None)

                if width is None or height is None:
                    if self.train_data_format == "normal":
                        video_id, name = video_dict['file_path'], video_dict['text']
                        if self.video_folder is None:
                            video_dir = video_id
                        else:
                            video_dir = os.path.join(self.video_folder, video_id)
                    else:
                        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
                        video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")

                    try:
                        vr = VideoReader(video_dir)
                        frame = vr[0].asnumpy()
                        height, width = frame.shape[0], frame.shape[1]
                    except Exception as e:
                        print(f"ERROR in bucket_sampler: Failed to read video with decord")
                        continue


                    ratio = height / width # self.dataset[idx]
                else:
                    height = int(height)
                    width = int(width)
                    ratio = height / width # self.dataset[idx]
            except Exception as e:
                print(f"ERROR in bucket_sampler: {type(e).__name__}: {e}")
                continue
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

class AspectRatioBatchImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

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
                 train_folder: str = None,
                 aspect_ratios: dict = ASPECT_RATIO_512,
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
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.current_available_bucket_keys = list(aspect_ratios.keys())
        self.bucket = {
            'image':{ratio: [] for ratio in aspect_ratios},
            'video':{ratio: [] for ratio in aspect_ratios}
        }

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset[idx].get('type', 'image')
            if content_type not in self.bucket:
                continue
            if content_type == 'image':
                try:
                    image_dict = self.dataset[idx]

                    width, height = image_dict.get("width", None), image_dict.get("height", None)
                    if width is None or height is None:
                        image_id, name = image_dict['file_path'], image_dict['text']
                        if self.train_folder is None:
                            image_dir = image_id
                        else:
                            image_dir = os.path.join(self.train_folder, image_id)

                        width, height = get_image_size_without_loading(image_dir)

                        ratio = height / width # self.dataset[idx]
                    else:
                        height = int(height)
                        width = int(width)
                        ratio = height / width # self.dataset[idx]
                except Exception as e:
                    print(f"ERROR in bucket_sampler (image): {type(e).__name__}: {e}")
                    continue
                # find the closest aspect ratio
                closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
                if closest_ratio not in self.current_available_bucket_keys:
                    continue
                bucket = self.bucket['image'][closest_ratio]
                bucket.append(idx)
                # yield a batch of indices in the same aspect ratio group
                if len(bucket) == self.batch_size:
                    yield bucket[:]
                    del bucket[:]
            else:
                try:
                    video_dict = self.dataset[idx]
                    width, height = video_dict.get("width", None), video_dict.get("height", None)

                    if width is None or height is None:
                        video_id, name = video_dict['file_path'], video_dict['text']
                        if self.train_folder is None:
                            video_dir = video_id
                        else:
                            video_dir = os.path.join(self.train_folder, video_id)

                        try:
                            vr = VideoReader(video_dir)
                            frame = vr[0].asnumpy()
                            height, width = frame.shape[0], frame.shape[1]
                        except Exception as e:
                            print(f"ERROR in bucket_sampler (video): Failed to read video with decord")
                            continue


                        ratio = height / width # self.dataset[idx]
                    else:
                        height = int(height)
                        width = int(width)
                        ratio = height / width # self.dataset[idx]
                except Exception as e:
                    print(f"ERROR in bucket_sampler (video): {type(e).__name__}: {e}")
                    continue
                # find the closest aspect ratio
                closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
                if closest_ratio not in self.current_available_bucket_keys:
                    continue
                bucket = self.bucket['video'][closest_ratio]
                bucket.append(idx)
                # yield a batch of indices in the same aspect ratio group
                if len(bucket) == self.batch_size:
                    yield bucket[:]
                    del bucket[:]
