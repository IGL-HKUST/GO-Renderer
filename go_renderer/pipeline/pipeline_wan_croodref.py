import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from transformers import T5Tokenizer

from ..models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                              WanT5EncoderModel)
from ..models.multiref_transformer3d import (
    CroodRefTransformer3DModel,
    CroodRefTransformer3DModel2_2,
)
from ..utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                get_sampling_sigmas)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for the GO-Renderer Wan pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class _ReferenceFrameVaeEncodingMixin:
    """Encode unordered reference views without temporal VAE compression."""

    def _encode_reference_frames(
        self,
        reference: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        """Encode each reference view independently and preserve its temporal slot."""
        if reference.ndim != 5:
            raise ValueError(
                "Reference input must have shape [B, C, F, H, W], "
                f"but received {tuple(reference.shape)}."
            )

        latent_frames = []
        for frame_index in range(reference.shape[2]):
            frame_latents = self.prepare_control_latents(
                None,
                reference[:, :, frame_index : frame_index + 1],
                batch_size,
                height,
                width,
                dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )[1]
            latent_frames.append(frame_latents)

        if not latent_frames:
            raise ValueError("Reference input must contain at least one frame.")
        return torch.cat(latent_frames, dim=2)


class WanFunCroodRefPipeline(_ReferenceFrameVaeEncodingMixin, DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan with coordinate-based reference support.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->clip_image_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: CroodRefTransformer3DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        clip_image_encoder: Optional[CLIPModel] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, clip_image_encoder=clip_image_encoder, scheduler=scheduler
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_control_latents(
        self, control, control_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if control is not None:
            control = control.to(device=device, dtype=dtype)
            bs = 1
            new_control = []
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                control_bs = self.vae.encode(control_bs)[0]
                control_bs = control_bs.mode()
                new_control.append(control_bs)
            control = torch.cat(new_control, dim = 0)

        if control_image is not None:
            control_image = control_image.to(device=device, dtype=dtype)
            bs = 1
            new_control_pixel_values = []
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = control_image[i : i + bs]
                control_pixel_values_bs = self.vae.encode(control_pixel_values_bs)[0]
                control_pixel_values_bs = control_pixel_values_bs.mode()
                new_control_pixel_values.append(control_pixel_values_bs)
            control_image_latents = torch.cat(new_control_pixel_values, dim = 0)
        else:
            control_image_latents = None

        return control, control_image_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        ref_image: Union[torch.FloatTensor] = None,
        ref_coordmap: Optional[torch.FloatTensor] = None,
        fg_coordmap: Optional[torch.FloatTensor] = None,
        bg_video: Optional[torch.FloatTensor] = None,
        start_image: Optional[torch.FloatTensor] = None,
        clip_image: Image = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.5,
        guide_scale_ref: float = 5.0,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        shift: int = 5,
        boundary: float = 0.900,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation.
            height (`int`, *optional*, defaults to 480):
                The height in pixels of the generated video.
            width (`int`, *optional*, defaults to 720):
                The width in pixels of the generated video.
            ref_image (`torch.FloatTensor`, *optional*):
                Reference images for subject guidance. Shape: [B, C, F, H, W] or [B, C, H, W].
            ref_coordmap (`torch.FloatTensor`, *optional*):
                Reference coordinate map. Shape: [B, F, C, H, W].
            fg_coordmap (`torch.FloatTensor`, *optional*):
                Foreground coordinate map. Shape: [B, F, C, H, W].
            bg_video (`torch.FloatTensor`, *optional*):
                Background video for compositing. Shape: [B, F, C, H, W].
            start_image (`torch.FloatTensor`, *optional*):
                Start image for video generation. Shape: [B, C, 1, H, W].
            num_frames (`int`, *optional*, defaults to 49):
                The number of video frames to generate.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Text guidance scale as defined in Classifier-Free Diffusion Guidance.
            guide_scale_ref (`float`, *optional*, defaults to 5.0):
                Reference image guidance scale.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of torch generator(s) to make generation deterministic.

        Examples:
            xxx:xxx

        Returns:
            `WanPipelineOutput` or `tuple`:
                If `return_dict` is `True`, returns `WanPipelineOutput`, otherwise a tuple.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        # 5. Prepare latents.
        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        # 6. Process reference images
        # NOTE: All input images/videos are expected to be in [0, 1] range, converted to [-1, 1] for VAE
        full_ref = None
        if ref_image is not None:
            # Handle both 4D and 5D inputs
            if ref_image.dim() == 4:
                # Single frame case: [B, C, H, W] -> [B, C, 1, H, W]
                ref_image = ref_image.unsqueeze(2)
                video_length = 1
            else:
                # Multi-frame case: [B, C, F, H, W]
                video_length = ref_image.shape[2]

            ref_image = self.image_processor.preprocess(
                rearrange(ref_image, "b c f h w -> (b f) c h w"),
                height=height, width=width
            )
            ref_image = ref_image.to(dtype=torch.float32)
            ref_image = rearrange(ref_image, "(b f) c h w -> b c f h w", f=video_length)

            ref_image_latents = self._encode_reference_frames(
                ref_image,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # Process full_ref
            if ref_image_latents.size(2) == 1:
                full_ref = ref_image_latents.squeeze(2)  # [B, C, H, W]
            else:
                full_ref = ref_image_latents  # [B, C, F, H, W]

        # 7. Process ref_coordmap
        ref_coordmap_latents = None
        if ref_coordmap is not None:
            # ref_coordmap: [B, F, C, H, W]
            video_length = ref_coordmap.shape[1]
            ref_coordmap = self.image_processor.preprocess(
                rearrange(ref_coordmap, "b f c h w -> (b f) c h w"),
                height=height, width=width
            )
            ref_coordmap = ref_coordmap.to(dtype=torch.float32)
            ref_coordmap = rearrange(ref_coordmap, "(b f) c h w -> b c f h w", f=video_length)

            ref_coordmap_latents = self._encode_reference_frames(
                ref_coordmap,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # Process ref_coordmap_latents similar to full_ref
            if ref_coordmap_latents.size(2) == 1:
                ref_coordmap_latents = ref_coordmap_latents.squeeze(2)  # [B, C, H, W]
            # else: keep as [B, C, F, H, W]

        # 8. Process appearance_latents (start_image and/or bg_video)
        appearance_latents = torch.zeros_like(latents)
        # bg_video if provided
        if bg_video is not None:
            video_length = bg_video.shape[1]  # [B, F, C, H, W]

            if video_length != num_frames:
                print(f"Warning: bg_video frame count ({video_length}) != num_frames ({num_frames}), resampling...")
                bg_video = rearrange(bg_video, "b f c h w -> b c f h w")
                bg_video = torch.nn.functional.interpolate(
                    bg_video,
                    size=(num_frames, bg_video.shape[3], bg_video.shape[4]),
                    mode='trilinear',
                    align_corners=False
                )
                bg_video = rearrange(bg_video, "b c f h w -> b f c h w")
                video_length = num_frames

            bg_video = self.image_processor.preprocess(
                rearrange(bg_video, "b f c h w -> (b f) c h w"),
                height=height, width=width
            )
            bg_video = bg_video.to(dtype=torch.float32)
            bg_video = rearrange(bg_video, "(b f) c h w -> b c f h w", f=video_length)

            appearance_latents = self.prepare_control_latents(
                None,
                bg_video,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]
        # override first frame with start_image if provided
        if start_image is not None:
            video_length = start_image.shape[2]
            start_image = self.image_processor.preprocess(rearrange(start_image, "b c f h w -> (b f) c h w"), height=height, width=width)
            start_image = start_image.to(dtype=torch.float32)
            start_image = rearrange(start_image, "(b f) c h w -> b c f h w", f=video_length)

            start_image_latents = self.prepare_control_latents(
                None,
                start_image,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

            if latents.size()[2] != 1:
                appearance_latents[:, :, :1] = start_image_latents

        # 9. Process fg_coordmap
        fg_coordmap_latent = None
        if fg_coordmap is not None:
            video_length = fg_coordmap.shape[1]  # [B, F, C, H, W]

            # Adjust fg_coordmap length to match num_frames if needed
            if video_length != num_frames:
                print(f"Warning: fg_coordmap frame count ({video_length}) != num_frames ({num_frames}), resampling...")
                fg_coordmap = rearrange(fg_coordmap, "b f c h w -> b c f h w")
                fg_coordmap = torch.nn.functional.interpolate(
                    fg_coordmap,
                    size=(num_frames, fg_coordmap.shape[3], fg_coordmap.shape[4]),
                    mode='trilinear',
                    align_corners=False
                )
                # Rearrange back to [B, F, C, H, W]
                fg_coordmap = rearrange(fg_coordmap, "b c f h w -> b f c h w")
                video_length = num_frames

            fg_coordmap = self.image_processor.preprocess(
                rearrange(fg_coordmap, "b f c h w -> (b f) c h w"),
                height=height, width=width
            )
            fg_coordmap = fg_coordmap.to(dtype=torch.float32)
            fg_coordmap = rearrange(fg_coordmap, "(b f) c h w -> b c f h w", f=video_length)

            fg_coordmap_latent = self.prepare_control_latents(
                None,
                fg_coordmap,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

        # 11. Prepare clip latent variables
        if clip_image is not None and self.clip_image_encoder is not None:
            clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype)
            clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
        else:
            if self.clip_image_encoder is not None:
                clip_image_dummy = Image.new("RGB", (512, 512), color=(0, 0, 0))
                clip_image_dummy = TF.to_tensor(clip_image_dummy).sub_(0.5).div_(0.5).to(device, weight_dtype)
                clip_context = self.clip_image_encoder([clip_image_dummy[:, None, :, :]])
                clip_context = torch.zeros_like(clip_context)
            else:
                clip_context = None

        # 12. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1, width // self.vae.spatial_compression_ratio, height // self.vae.spatial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1])

        # 13. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        if getattr(self, "transformer_2", None) is not None:
            self.transformer_2.num_inference_steps = num_inference_steps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                local_transformer = self.transformer
                if getattr(self, "transformer_2", None) is not None:
                    if t >= boundary * self.scheduler.config.num_train_timesteps:
                        local_transformer = self.transformer_2
                local_transformer.current_steps = i

                if self.interrupt:
                    continue

                latent_model_input = latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Prepare inputs for CFG using batch dimension
                if do_classifier_free_guidance:
                    if fg_coordmap_latent is not None:
                        fg_coordmap_zeros = torch.zeros_like(fg_coordmap_latent)
                    else:
                        fg_coordmap_zeros = torch.zeros_like(latents).to(latents.device, latents.dtype)

                    # Prepare reference inputs
                    full_ref_input = full_ref
                    if full_ref is not None:
                        full_ref_zeros = torch.zeros_like(full_ref)
                    else:
                        full_ref_zeros = None

                    if ref_coordmap_latents is not None:
                        ref_coordmap_zeros = torch.zeros_like(ref_coordmap_latents)
                    else:
                        ref_coordmap_zeros = None

                    # Prepare batched inputs for single forward pass
                    # Order: [neg, pos_i, pos_it]
                    # neg: negative text + negative reference
                    # pos_i: negative text + positive reference
                    # pos_it: positive text + positive reference

                    # Batch latents: [neg, pos_i, pos_it]
                    latent_model_input_batched = torch.cat([latent_model_input] * 3, dim=0)

                    # Batch fg_coordmap: [pos, pos, pos]
                    if fg_coordmap_latent is not None:
                        fg_coordmap_batched = torch.cat([fg_coordmap_latent, fg_coordmap_latent, fg_coordmap_latent], dim=0)
                    else:
                        fg_coordmap_batched = None

                    # Batch appearance: [appearance, appearance, appearance]
                    appearance_batched = torch.cat([appearance_latents] * 3, dim=0)

                    # Batch context: [neg, neg, pos]
                    context_batched = negative_prompt_embeds + negative_prompt_embeds + prompt_embeds

                    # Batch clip_fea
                    if clip_context is not None:
                        clip_fea_batched = torch.cat([clip_context] * 3, dim=0)
                    else:
                        clip_fea_batched = None

                    # Batch full_ref: [zeros, pos, pos]
                    if full_ref_input is not None:
                        if full_ref_input.dim() == 4:  # [B, C, H, W]
                            full_ref_batched = torch.cat([full_ref_zeros, full_ref_input, full_ref_input], dim=0)
                        else:  # [B, C, F, H, W]
                            full_ref_batched = torch.cat([full_ref_zeros, full_ref_input, full_ref_input], dim=0)
                    else:
                        full_ref_batched = None

                    # Batch ref_coordmap: [pos, pos, pos]
                    if ref_coordmap_latents is not None:
                        if ref_coordmap_latents.dim() == 4:  # [B, C, H, W]
                            ref_coordmap_batched = torch.cat([ref_coordmap_zeros, ref_coordmap_latents, ref_coordmap_latents], dim=0)
                        else:  # [B, C, F, H, W]
                            ref_coordmap_batched = torch.cat([ref_coordmap_zeros, ref_coordmap_latents, ref_coordmap_latents], dim=0)
                    else:
                        ref_coordmap_batched = None

                    # Broadcast timestep to batched dimension
                    timestep = t.expand(latent_model_input_batched.shape[0])

                    # Single forward pass with batched inputs
                    with torch.amp.autocast("cuda", dtype=weight_dtype), torch.cuda.device(device=device):
                        noise_pred_batched = local_transformer(
                            x=latent_model_input_batched,
                            context=context_batched,
                            t=timestep,
                            seq_len=seq_len,
                            clip_fea=clip_fea_batched,
                            full_ref=full_ref_batched,
                            full_ref_crood=ref_coordmap_batched,
                            fg_coordmap=fg_coordmap_batched,
                            appearance=appearance_batched,
                        )

                    # Split the batched predictions
                    noise_pred_neg, noise_pred_pos_i, noise_pred_pos_it = noise_pred_batched.chunk(3, dim=0)

                    # Dual CFG: neg + guide_scale_ref * (pos_i - neg) + guidance_scale * (pos_it - pos_i)
                    noise_pred = noise_pred_neg + guide_scale_ref * (noise_pred_pos_i - noise_pred_neg) + self.guidance_scale * (noise_pred_pos_it - noise_pred_pos_i)

                    # Note: Transformer automatically removes ref frames from output, no need to slice here
                else:
                    # Non-CFG
                    if clip_context is not None:
                        clip_context_input = clip_context
                    else:
                        clip_context_input = None

                    # broadcast to batch dimension
                    timestep = t.expand(latent_model_input.shape[0])

                    # predict noise model_output
                    with torch.amp.autocast("cuda", dtype=weight_dtype), torch.cuda.device(device=device):
                        noise_pred = local_transformer(
                            x=latent_model_input,
                            context=in_prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                            clip_fea=clip_context_input,
                            full_ref=full_ref,
                            full_ref_crood=ref_coordmap_latents,
                            fg_coordmap=fg_coordmap_latent,
                            appearance=appearance_latents,
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)


class WanFunCroodRefPipeline2_2(_ReferenceFrameVaeEncodingMixin, DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan2.2 5B with coordinate-based reference support.

    This pipeline is designed for CroodRefTransformer3DModel2_2 (Wan2.2 5B) and does NOT use CLIP.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"  # No CLIP

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae,  # AutoencoderKLWan or AutoencoderKLWan3_8
        transformer: CroodRefTransformer3DModel2_2,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_control_latents(
        self, control, control_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        if control is not None:
            control = control.to(device=device, dtype=dtype)
            bs = 1
            new_control = []
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                control_bs = self.vae.encode(control_bs)[0]
                control_bs = control_bs.mode()
                new_control.append(control_bs)
            control = torch.cat(new_control, dim = 0)

        if control_image is not None:
            control_image = control_image.to(device=device, dtype=dtype)
            bs = 1
            new_control_pixel_values = []
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = control_image[i : i + bs]
                control_pixel_values_bs = self.vae.encode(control_pixel_values_bs)[0]
                control_pixel_values_bs = control_pixel_values_bs.mode()
                new_control_pixel_values.append(control_pixel_values_bs)
            control_image_latents = torch.cat(new_control_pixel_values, dim = 0)
        else:
            control_image_latents = None

        return control, control_image_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = frames.cpu().float().numpy()
        return frames

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        ref_image: Union[torch.FloatTensor] = None,
        ref_coordmap: Optional[torch.FloatTensor] = None,
        fg_coordmap: Optional[torch.FloatTensor] = None,
        bg_video: Optional[torch.FloatTensor] = None,
        start_image: Optional[torch.FloatTensor] = None,
        clip_image: Image = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.5,
        guide_scale_ref: float = 5.0,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        shift: int = 5,
        boundary: float = 0.900,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Note: This pipeline does NOT use CLIP, unlike WanFunCroodRefPipeline.

        Example:
            ...
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        # 5. Prepare latents
        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        # 6. Process reference images
        full_ref = None
        if ref_image is not None:
            if ref_image.dim() == 4:
                ref_image = ref_image.unsqueeze(2)
                video_length = 1
            else:
                video_length = ref_image.shape[2]

            ref_image = self.image_processor.preprocess(
                rearrange(ref_image, "b c f h w -> (b f) c h w"),
                height=height, width=width
            )
            ref_image = ref_image.to(dtype=torch.float32)
            ref_image = rearrange(ref_image, "(b f) c h w -> b c f h w", f=video_length)

            ref_image_latents = self._encode_reference_frames(
                ref_image,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            if ref_image_latents.size(2) == 1:
                full_ref = ref_image_latents.squeeze(2)
            else:
                full_ref = ref_image_latents

        # 7. Process ref_coordmap
        ref_coordmap_latents = None
        if ref_coordmap is not None:
            video_length = ref_coordmap.shape[1]
            ref_coordmap = self.image_processor.preprocess(
                rearrange(ref_coordmap, "b f c h w -> (b f) c h w"),
                height=height, width=width
            )
            ref_coordmap = ref_coordmap.to(dtype=torch.float32)
            ref_coordmap = rearrange(ref_coordmap, "(b f) c h w -> b c f h w", f=video_length)

            ref_coordmap_latents = self._encode_reference_frames(
                ref_coordmap,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            if ref_coordmap_latents.size(2) == 1:
                ref_coordmap_latents = ref_coordmap_latents.squeeze(2)

        # 8. Process appearance_latents (start_image and/or bg_video)
        appearance_latents = torch.zeros_like(latents)
        if bg_video is not None:
            video_length = bg_video.shape[1]

            if video_length != num_frames:
                print(f"Warning: bg_video frame count ({video_length}) != num_frames ({num_frames}), resampling...")
                bg_video = rearrange(bg_video, "b f c h w -> b c f h w")
                bg_video = torch.nn.functional.interpolate(
                    bg_video,
                    size=(num_frames, bg_video.shape[3], bg_video.shape[4]),
                    mode='trilinear',
                    align_corners=False
                )
                bg_video = rearrange(bg_video, "b c f h w -> b f c h w")
                video_length = num_frames

            bg_video = self.image_processor.preprocess(
                rearrange(bg_video, "b f c h w -> (b f) c h w"),
                height=height, width=width
            )
            bg_video = bg_video.to(dtype=torch.float32)
            bg_video = rearrange(bg_video, "(b f) c h w -> b c f h w", f=video_length)

            appearance_latents = self.prepare_control_latents(
                None,
                bg_video,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

        if start_image is not None:
            start_image_length = start_image.shape[2]
            start_image = self.image_processor.preprocess(rearrange(start_image, "b c f h w -> (b f) c h w"), height=height, width=width)
            start_image = start_image.to(dtype=torch.float32)
            start_image = rearrange(start_image, "(b f) c h w -> b c f h w", f=start_image_length)

            start_image_latents = self.prepare_control_latents(
                None,
                start_image,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

            if latents.size()[2] != 1:
                appearance_latents[:, :, :1] = start_image_latents

        # 9. Process fg_coordmap
        fg_coordmap_latent = None
        if fg_coordmap is not None:
            video_length = fg_coordmap.shape[1]

            if video_length != num_frames:
                print(f"Warning: fg_coordmap frame count ({video_length}) != num_frames ({num_frames}), resampling...")
                fg_coordmap = rearrange(fg_coordmap, "b f c h w -> b c f h w")
                fg_coordmap = torch.nn.functional.interpolate(
                    fg_coordmap,
                    size=(num_frames, fg_coordmap.shape[3], fg_coordmap.shape[4]),
                    mode='trilinear',
                    align_corners=False
                )
                fg_coordmap = rearrange(fg_coordmap, "b c f h w -> b f c h w")
                video_length = num_frames

            fg_coordmap = self.image_processor.preprocess(
                rearrange(fg_coordmap, "b f c h w -> (b f) c h w"),
                height=height, width=width
            )
            fg_coordmap = fg_coordmap.to(dtype=torch.float32)
            fg_coordmap = rearrange(fg_coordmap, "(b f) c h w -> b c f h w", f=video_length)

            fg_coordmap_latent = self.prepare_control_latents(
                None,
                fg_coordmap,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

        # 10. Prepare CLIP compatibility features when a CLIP encoder is registered.
        if getattr(self, "clip_image_encoder", None) is not None:
            if clip_image is not None:
                clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype)
                clip_context = self.clip_image_encoder([clip_image[:, None, :, :]])
            else:
                clip_image_dummy = Image.new("RGB", (512, 512), color=(0, 0, 0))
                clip_image_dummy = TF.to_tensor(clip_image_dummy).sub_(0.5).div_(0.5).to(device, weight_dtype)
                clip_context = self.clip_image_encoder([clip_image_dummy[:, None, :, :]])
                clip_context = torch.zeros_like(clip_context)
        else:
            clip_context = None

        # 11. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1, width // self.vae.spatial_compression_ratio, height // self.vae.spatial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1])

        # 12. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        if getattr(self, "transformer_2", None) is not None:
            self.transformer_2.num_inference_steps = num_inference_steps

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                local_transformer = self.transformer
                if getattr(self, "transformer_2", None) is not None:
                    if t >= boundary * self.scheduler.config.num_train_timesteps:
                        local_transformer = self.transformer_2
                local_transformer.current_steps = i

                if self.interrupt:
                    continue

                latent_model_input = latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Prepare inputs for CFG using batch dimension
                if do_classifier_free_guidance:
                    if fg_coordmap_latent is not None:
                        fg_coordmap_zeros = torch.zeros_like(fg_coordmap_latent)
                    else:
                        fg_coordmap_zeros = torch.zeros_like(latents).to(latents.device, latents.dtype)

                    full_ref_input = full_ref
                    if full_ref is not None:
                        full_ref_zeros = torch.zeros_like(full_ref)
                    else:
                        full_ref_zeros = None

                    if ref_coordmap_latents is not None:
                        ref_coordmap_zeros = torch.zeros_like(ref_coordmap_latents)
                    else:
                        ref_coordmap_zeros = None

                    # Batch latents: [neg, pos_i, pos_it]
                    latent_model_input_batched = torch.cat([latent_model_input] * 3, dim=0)

                    # Batch fg_coordmap: [pos, pos, pos], matching the 5B CoordRef CFG path.
                    if fg_coordmap_latent is not None:
                        fg_coordmap_batched = torch.cat([fg_coordmap_latent, fg_coordmap_latent, fg_coordmap_latent], dim=0)
                    else:
                        fg_coordmap_batched = None

                    # Batch appearance: [appearance, appearance, appearance]
                    appearance_zeros = torch.zeros_like(appearance_latents)
                    appearance_batched = torch.cat([appearance_latents, appearance_latents, appearance_latents], dim=0)

                    # Batch context: [neg, neg, pos]
                    context_batched = negative_prompt_embeds + negative_prompt_embeds + prompt_embeds

                    # Batch full_ref: [zeros, pos, pos]
                    if full_ref_input is not None:
                        if full_ref_input.dim() == 4:
                            full_ref_batched = torch.cat([full_ref_zeros, full_ref_input, full_ref_input], dim=0)
                        else:
                            full_ref_batched = torch.cat([full_ref_zeros, full_ref_input, full_ref_input], dim=0)
                    else:
                        full_ref_batched = None

                    # Batch ref_coordmap: [zeros, pos, pos]
                    if ref_coordmap_latents is not None:
                        if ref_coordmap_latents.dim() == 4:
                            ref_coordmap_batched = torch.cat([ref_coordmap_zeros, ref_coordmap_latents, ref_coordmap_latents], dim=0)
                        else:
                            ref_coordmap_batched = torch.cat([ref_coordmap_zeros, ref_coordmap_latents, ref_coordmap_latents], dim=0)
                    else:
                        ref_coordmap_batched = None

                    timestep = t.expand(latent_model_input_batched.shape[0])

                    # Single forward pass with batched inputs (no CLIP)
                    with torch.amp.autocast("cuda", dtype=weight_dtype), torch.cuda.device(device=device):
                        noise_pred_batched = local_transformer(
                            x=latent_model_input_batched,
                            context=context_batched,
                            t=timestep,
                            seq_len=seq_len,
                            clip_fea=torch.cat([clip_context] * 3, dim=0) if clip_context is not None else None,
                            full_ref=full_ref_batched,
                            full_ref_crood=ref_coordmap_batched,
                            fg_coordmap=fg_coordmap_batched,
                            appearance=appearance_batched,
                        )

                    noise_pred_neg, noise_pred_pos_i, noise_pred_pos_it = noise_pred_batched.chunk(3, dim=0)

                    # Dual CFG
                    noise_pred = noise_pred_neg + guide_scale_ref * (noise_pred_pos_i - noise_pred_neg) + self.guidance_scale * (noise_pred_pos_it - noise_pred_pos_i)
                else:
                    timestep = t.expand(latent_model_input.shape[0])

                    with torch.amp.autocast("cuda", dtype=weight_dtype), torch.cuda.device(device=device):
                        noise_pred = local_transformer(
                            x=latent_model_input,
                            context=in_prompt_embeds,
                            t=timestep,
                            seq_len=seq_len,
                            clip_fea=clip_context,
                            full_ref=full_ref,
                            full_ref_crood=ref_coordmap_latents,
                            fg_coordmap=fg_coordmap_latent,
                            appearance=appearance_latents,
                        )

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)
