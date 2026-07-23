"""Lightweight data-validation helpers for GO-Renderer.

Training code imports dataset implementations from ``dataset_image_video``
directly so importing this package does not eagerly initialize video libraries.
"""

from .validation import (
    load_training_manifest,
    load_validation_manifest,
    validate_generation_shape,
    validate_rope_gap,
    validate_validation_media,
)

__all__ = [
    "load_validation_manifest",
    "load_training_manifest",
    "validate_generation_shape",
    "validate_rope_gap",
    "validate_validation_media",
]
