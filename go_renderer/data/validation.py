"""Validation-manifest checks shared by the 5B and 14B inference CLIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_VALIDATION_FIELDS = ("text", "ref", "ref_coordmap", "fg_coordmap")
REQUIRED_TRAINING_FIELDS = (
    "type",
    "file_path",
    "text",
    "ref",
    "ref_coordmap",
    "fg_coordmap",
)
OPTIONAL_MEDIA_FIELDS = ("firstframe", "bg", "bgvideo", "gt", "video_path")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tiff", ".webp"}
ROPE_FREQUENCY_POSITIONS = 1024


@dataclass(frozen=True)
class MediaInfo:
    """Basic media properties used by inference preflight checks."""

    frames: int
    width: int
    height: int


def load_validation_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a validation manifest and verify all referenced local media."""

    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Validation manifest does not exist: {manifest_path}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in validation manifest {manifest_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("Validation manifest top level must be a list")
    if not payload:
        raise ValueError("Validation manifest must contain at least one entry")

    base_dir = manifest_path.resolve().parent
    checked: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(payload):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Validation manifest entry {index} must be an object")

        entry = dict(raw_entry)
        for field in REQUIRED_VALIDATION_FIELDS:
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Validation manifest entry {index} requires a non-empty '{field}' field"
                )

        seed = entry.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError(
                f"Validation manifest entry {index} field 'seed' must be an integer"
            )

        for field in REQUIRED_VALIDATION_FIELDS[1:] + OPTIONAL_MEDIA_FIELDS:
            value = entry.get(field)
            if value in (None, ""):
                continue
            if not isinstance(value, str):
                raise ValueError(
                    f"Validation manifest entry {index} field '{field}' must be a path string"
                )
            media_path = Path(value).expanduser()
            if not media_path.is_absolute():
                media_path = base_dir / media_path
            if not media_path.is_file() and not media_path.is_dir():
                raise FileNotFoundError(
                    f"Validation manifest entry {index} field '{field}' does not exist: {media_path}"
                )

        checked.append(entry)

    return checked


def load_training_manifest(
    path: str | Path,
    data_root: str | Path,
) -> list[dict[str, Any]]:
    """Validate the required training schema and all referenced local paths."""

    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in training manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("Training manifest must be a non-empty list")

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Training data root does not exist: {root}")

    checked: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(payload):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Training manifest entry {index} must be an object")
        entry = dict(raw_entry)
        for field in REQUIRED_TRAINING_FIELDS:
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Training manifest entry {index} requires a non-empty '{field}' field"
                )
        if entry["type"] not in {"image", "video"}:
            raise ValueError(
                f"Training manifest entry {index} has unsupported type: {entry['type']}"
            )
        for field in ("file_path", "ref", "ref_coordmap", "fg_coordmap"):
            media_path = Path(entry[field]).expanduser()
            if not media_path.is_absolute():
                media_path = root / media_path
            if not media_path.exists():
                raise FileNotFoundError(
                    f"Training manifest entry {index} field '{field}' does not exist: {media_path}"
                )
        checked.append(entry)
    return checked


def inspect_media(path: str | Path) -> MediaInfo:
    """Inspect an image, image directory, or video without importing the training stack.

    The function deliberately decodes image pixels and the first/last video frames so
    obvious corruption is reported before a training or inference model is loaded.
    Callers processing large datasets should cache the result within each worker.
    """

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Media path does not exist: {path}")

    if path.is_dir():
        image_paths = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"Image directory contains no supported images: {path}")

        from PIL import Image

        dimensions = set()
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image.load()
                    dimensions.add(image.size)
            except Exception as exc:
                raise ValueError(f"Could not read image file {image_path}: {exc}") from exc
        if len(dimensions) != 1:
            raise ValueError(f"Image directory contains inconsistent dimensions: {path}")
        width, height = dimensions.pop()
        return MediaInfo(frames=len(image_paths), width=width, height=height)

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        from PIL import Image

        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
        except Exception as exc:
            raise ValueError(f"Could not read image file {path}: {exc}") from exc
        return MediaInfo(frames=1, width=width, height=height)

    try:
        from decord import VideoReader

        reader = VideoReader(str(path))
        frames = len(reader)
        if frames < 1:
            raise ValueError(f"Video contains no frames: {path}")
        first_frame = reader[0]
        # Decode the last frame as well so truncated media fails before model loading.
        if frames > 1:
            reader[frames - 1]
        height, width = first_frame.shape[:2]
    except Exception as exc:
        raise ValueError(f"Could not read media file {path}: {exc}") from exc
    return MediaInfo(frames=frames, width=int(width), height=int(height))


def letterbox_image(image: Any, width: int, height: int) -> Any:
    """Resize a PIL image proportionally and center it on a white RGB canvas."""

    from PIL import Image

    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")

    image = image.convert("RGB")
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {image.size}")

    scale = min(width / source_width, height / source_height)
    resized_width = min(width, max(1, round(source_width * scale)))
    resized_height = min(height, max(1, round(source_height * scale)))
    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(
        resized,
        ((width - resized_width) // 2, (height - resized_height) // 2),
    )
    return canvas


def resolve_background_path(entry: dict[str, Any]) -> str | None:
    """Prefer a non-empty background video and otherwise fall back to an image."""

    return entry.get("bgvideo") or entry.get("bg")


def validate_rope_gap(
    rope_gap: int,
    num_ref_frames: int,
    *,
    max_positions: int = ROPE_FREQUENCY_POSITIONS,
) -> int:
    """Validate negative-time RoPE spacing against the frequency table."""

    if not isinstance(rope_gap, int) or isinstance(rope_gap, bool) or rope_gap <= 0:
        raise ValueError(f"rope_gap must be a positive integer, got {rope_gap!r}")
    if (
        not isinstance(num_ref_frames, int)
        or isinstance(num_ref_frames, bool)
        or num_ref_frames <= 0
    ):
        raise ValueError(
            f"num_ref_frames must be a positive integer, got {num_ref_frames!r}"
        )
    if rope_gap * num_ref_frames > max_positions:
        raise ValueError(
            "rope_gap exceeds the negative-time RoPE table: "
            f"{rope_gap} * {num_ref_frames} reference frames > {max_positions} positions"
        )
    return rope_gap


def validate_validation_media(
    entries: list[dict[str, Any]],
    manifest_path: str | Path,
    *,
    num_ref_frames: int,
    num_frames: int,
) -> list[dict[str, Any]]:
    """Resolve and validate all inference media before loading model weights."""

    if not 3 <= num_ref_frames <= 8:
        raise ValueError(f"num_ref_frames must be in the inclusive range [3, 8], got {num_ref_frames}")
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    manifest_dir = Path(manifest_path).expanduser().resolve().parent
    inspection_cache: dict[Path, MediaInfo] = {}

    def inspect(path: str | Path) -> MediaInfo:
        resolved_path = Path(path).expanduser().resolve()
        if resolved_path not in inspection_cache:
            inspection_cache[resolved_path] = inspect_media(resolved_path)
        return inspection_cache[resolved_path]

    resolved_entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries):
        entry = dict(raw_entry)
        for field in REQUIRED_VALIDATION_FIELDS[1:] + OPTIONAL_MEDIA_FIELDS:
            value = entry.get(field)
            if value in (None, ""):
                continue
            media_path = Path(value).expanduser()
            if not media_path.is_absolute():
                media_path = manifest_dir / media_path
            entry[field] = str(media_path.resolve())

        ref_info = inspect(entry["ref"])
        ref_coordmap_info = inspect(entry["ref_coordmap"])
        if ref_info.frames != ref_coordmap_info.frames:
            raise ValueError(
                f"Validation sample {index} has {ref_info.frames} reference frames but "
                f"{ref_coordmap_info.frames} reference coordinate-map frames"
            )
        if ref_info.frames < num_ref_frames:
            raise ValueError(
                f"Validation sample {index} has {ref_info.frames} reference frames; "
                f"at least {num_ref_frames} are required"
            )

        inspect(entry["fg_coordmap"])

        background_path = resolve_background_path(entry)
        if background_path:
            inspect(background_path)

        for optional_field in ("firstframe", "gt", "video_path"):
            optional_path = entry.get(optional_field)
            if optional_path:
                inspect(optional_path)

        resolved_entries.append(entry)

    return resolved_entries


def validate_generation_shape(
    height: int,
    width: int,
    num_frames: int,
    spatial_multiple: int,
) -> None:
    """Validate spatial divisibility and Wan VAE temporal frame layout."""

    values = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "spatial_multiple": spatial_multiple,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    if height % spatial_multiple or width % spatial_multiple:
        raise ValueError(
            f"height and width must be divisible by {spatial_multiple}; got {height}x{width}"
        )
    if num_frames % 4 != 1:
        raise ValueError(
            f"num_frames must satisfy num_frames % 4 == 1 for the Wan VAE; got {num_frames}"
        )
