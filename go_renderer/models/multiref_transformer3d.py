# Modified from wan_transformer3d.py

import glob
import json
import math
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version

from .wan_transformer3d import (
    WanTransformer3DModel,
    Wan2_2Transformer3DModel,
    WanRMSNorm,
    WanLayerNorm,
    WanSelfAttention,
    WAN_CROSSATTENTION_CLASSES,
    WanAttentionBlock,
    Head,
    rope_apply_qk,
    sinusoidal_embedding_1d,
)


@amp.autocast('cuda',enabled=False)
@torch.compiler.disable()
def rope_apply_with_ref(x, grid_sizes, ref_grid_sizes, freqs, gap=5):
    """
    Apply RoPE to a tensor x which is a concatenation of a reference part and a main part.
    Only the F (temporal) dimension is given negative positions with gaps for the reference part.
    H and W dimensions use positive positions for both reference and main parts.

    Args:
        x (Tensor): Input tensor, shape [B, L, num_heads, C / num_heads]. (ref + main)
        grid_sizes (Tensor): Shape [B, 3], grid sizes (F, H, W) for the main part.
        ref_grid_sizes (Tensor): Shape [B, 3], grid sizes (rF, rH, rW) for the reference part.
        freqs (Tensor): RoPE frequencies.
        gap (int): Gap multiplier for reference frame positions. Default is 5, giving positions like -5, -10, -15, etc.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs for F, H, W dimensions
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (main_grid, ref_grid) in enumerate(zip(grid_sizes.tolist(), ref_grid_sizes.tolist())):
        f, h, w = main_grid
        rf, rh, rw = ref_grid

        main_seq_len = f * h * w
        ref_seq_len = rf * rh * rw
        seq_len = ref_seq_len + main_seq_len

        # For reference frames:
        # F dimension: negative positions with gaps: -gap, -2*gap, -3*gap, ..., -rf*gap
        # Create negative indices with gaps: [rf*gap, (rf-1)*gap, ..., 2*gap, gap]
        ref_f_indices = torch.arange(rf, 0, -1, device=freqs[0].device) * gap
        freqs_f_ref = torch.stack([freqs[0][idx - 1] for idx in ref_f_indices]).conj()

        # H and W dimensions: use positive positions [0, 1, 2, ..., rh-1] and [0, 1, 2, ..., rw-1]
        freqs_h_ref = freqs[1][:rh]
        freqs_w_ref = freqs[2][:rw]

        freqs_ref_i = torch.cat([
            freqs_f_ref.view(rf, 1, 1, -1).expand(rf, rh, rw, -1),
            freqs_h_ref.view(1, rh, 1, -1).expand(rf, rh, rw, -1),
            freqs_w_ref.view(1, 1, rw, -1).expand(rf, rh, rw, -1)
        ], dim=-1).reshape(ref_seq_len, 1, -1)

        # For main frames: all dimensions use positive positions
        freqs_main_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(main_seq_len, 1, -1)

        freqs_i = torch.cat([freqs_ref_i, freqs_main_i], dim=0)

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        if seq_len < x.size(1):
            x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)

    return torch.stack(output).to(x.dtype)


def rope_apply_qk_with_ref(q, k, grid_sizes, ref_grid_sizes, freqs, gap=5):
    q = rope_apply_with_ref(q, grid_sizes, ref_grid_sizes, freqs, gap)
    k = rope_apply_with_ref(k, grid_sizes, ref_grid_sizes, freqs, gap)
    return q, k


class WanSelfAttentionWithRef(WanSelfAttention):
    """
    WanSelfAttention with support for reference frames using negative RoPE positions.
    """

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, rope_gap=5):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        self.rope_gap = rope_gap

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0, ref_grid_sizes=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads] - concatenated main + ref
            seq_lens(Tensor): Shape [B] - total sequence lengths including ref
            grid_sizes(Tensor): Shape [B, 3], grid sizes (F, H, W) for the main part
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            ref_grid_sizes(Tensor): Shape [B, 3], grid sizes (rF, rH, rW) for the reference part
        """
        from .attention_utils import attention

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # Apply RoPE with reference support if ref_grid_sizes is provided
        if ref_grid_sizes is not None:
            q, k = rope_apply_qk_with_ref(q, k, grid_sizes, ref_grid_sizes, freqs, self.rope_gap)
        else:
            q, k = rope_apply_qk(q, k, grid_sizes, freqs)

        x = attention(
            q.to(dtype),
            k.to(dtype),
            v=v.to(dtype),
            k_lens=seq_lens,
            window_size=self.window_size)
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlockWithRef(WanAttentionBlock):
    """
    WanAttentionBlock with support for reference frames.
    """

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 rope_gap=5):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)

        # Replace self_attn with the ref-aware version
        self.self_attn = WanSelfAttentionWithRef(dim, num_heads, window_size, qk_norm, eps, rope_gap)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.bfloat16,
        t=0,
        ref_grid_sizes=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            ref_grid_sizes(Tensor): Shape [B, 3], grid sizes (rF, rH, rW) for the reference part
        """
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype, t=t, ref_grid_sizes=ref_grid_sizes)
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype, t=t)

            # ffn function
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)

            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class CroodRefTransformer3DModel(WanTransformer3DModel):
    r"""
    CroodRef Transformer3D model inherited from WanTransformer3DModel.
    """

    @register_to_config
    def __init__(
        self,
        model_type='i2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=48,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=48,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
        rope_gap=5,
    ):
        r"""
        Initialize the CroodRef diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 48):
                Input video channels (C_in) - 48 for CroodRef (16 ref + 32 coordmap)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            rope_gap (`int`, *optional*, defaults to 5):
                Gap multiplier for reference frame RoPE positions (e.g., -5, -10, -15...)
        """

        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=False,
            in_dim_control_adapter=in_dim_control_adapter,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
            add_ref_conv=False,
            in_dim_ref_conv=in_dim_ref_conv,
            cross_attn_type=cross_attn_type,  # Let parent class decide based on model_type
        )

        # Remove components not used in CroodRef
        if hasattr(self, "control_adapter"):
            del self.control_adapter
        if hasattr(self, "ref_conv"):
            del self.ref_conv

        # Replace blocks with ref-aware versions
        if cross_attn_type is None:
            cross_attn_type = 'i2v_cross_attn' if model_type == 'i2v' else 'cross_attn'

        self.blocks = nn.ModuleList([
            WanAttentionBlockWithRef(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, rope_gap)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        # Save the original patch_embedding weights (16 channels from parent class)
        original_patch_weight = self.patch_embedding.weight.data.clone()
        original_patch_bias = self.patch_embedding.bias.data.clone() if self.patch_embedding.bias is not None else None

        # Force recreate patch_embedding with 48 channels for CroodRef
        # This ensures patch_embedding always uses 48 channels regardless of loaded config
        self.patch_embedding = nn.Conv3d(
            48,  # Always 48 channels for CroodRef (16 ref + 32 coordmap)
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

        with torch.no_grad():
            original_channels = original_patch_weight.size(1)
            self.patch_embedding.weight[:, :original_channels, :, :, :] = original_patch_weight
            self.patch_embedding.weight[:, original_channels:, :, :, :] = 0
            if original_patch_bias is not None:
                self.patch_embedding.bias.copy_(original_patch_bias)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded CroodRefTransformer3DModel from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                try:
                    from diffusers.models.model_loading_utils import \
                        load_model_dict_into_meta
                except ImportError:
                    from diffusers.models.modeling_utils import \
                        load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate

                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )

                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )

        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        if 'patch_embedding.weight' in state_dict:
            pretrained_channels = state_dict['patch_embedding.weight'].size(1)
            model_channels = model.state_dict()['patch_embedding.weight'].size(1)

            if pretrained_channels != model_channels:
                print(f"### Expanding patch_embedding from {pretrained_channels} to {model_channels} channels")
                new_weight = torch.zeros_like(model.state_dict()['patch_embedding.weight'])
                new_weight[:, :pretrained_channels, :, :, :] = state_dict['patch_embedding.weight']
                state_dict['patch_embedding.weight'] = new_weight
                print(f"### First {pretrained_channels} channels copied from pretrained, remaining {model_channels - pretrained_channels} channels zero-initialized")
            elif pretrained_channels == model_channels == 48:
                # Both are 48 channels, load directly
                print(f"### Loading patch_embedding with {model_channels} channels (no expansion needed)")
        else:
            raise ValueError(f"patch_embedding.weight not found in checkpoint")

        tmp_state_dict = {}
        for key in state_dict:
            # Skip ref_conv and control_adapter for CroodRef
            if key.startswith('ref_conv.') or key.startswith('control_adapter.'):
                print(key, "Skipping ref_conv/control_adapter for CroodRef")
                continue
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")

        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        model = model.to(torch_dtype)
        return model

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        full_ref=None,
        full_ref_crood=None,
        fg_coordmap=None,
        appearance=None,
        cond_flag=True,
    ):
        r"""
        Forward pass through the CroodRef diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            full_ref (Tensor, *optional*):
                Full reference frames, 16 channels, shape [B, 16, F, H, W] or [B, 16, H, W]
            full_ref_crood (Tensor, *optional*):
                Coordinate map for full reference frames, 16 channels, shape [B, 16, F, H, W] or [B, 16, H, W]
            fg_coordmap (Tensor, *optional*):
                Foreground coordinate map, 16 channels, shape [B, 16, F, H, W]
            appearance (Tensor, *optional*):
                Appearance latents (background/start_image), 16 channels, shape [B, 16, F, H, W]
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        first_x = x[0] if isinstance(x, list) else x
        device = first_x.device
        dtype = first_x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        # Concatenate fg_coordmap and appearance to form the control input (32 channels)
        if fg_coordmap is not None and appearance is not None:
            control_input = torch.cat([fg_coordmap, appearance], dim=1)  # [B, 32, F, H, W]
        elif fg_coordmap is not None:
            # If only fg_coordmap, pad with zeros for appearance
            control_input = torch.cat([fg_coordmap, torch.zeros_like(fg_coordmap)], dim=1)
        elif appearance is not None:
            # If only appearance, pad with zeros for fg_coordmap
            control_input = torch.cat([torch.zeros_like(appearance), appearance], dim=1)
        else:
            control_input = None

        # Concatenate control_input with x to form the input
        if control_input is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, control_input)]

        full_ref_token_num = 0
        full_ref_frames_num = 0
        if full_ref is not None:
            # Ensure full_ref is 5D [B, C, F, H, W]
            if full_ref.dim() == 4:
                full_ref = full_ref.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
            if full_ref_crood is not None and full_ref_crood.dim() == 4:
                full_ref_crood = full_ref_crood.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

            if full_ref_crood is not None:
                # full_ref_crood is 16 channels, concat with zeros to make 32
                full_ref_crood = torch.cat([full_ref_crood, full_ref], dim=1)
            else:
                full_ref_crood = torch.cat([torch.zeros_like(full_ref), full_ref], dim=1)

            full_ref_combined = torch.cat([full_ref, full_ref_crood], dim=1)  # [B, 48, F, H, W] or [B, 48, H, W]

            full_ref_embedded = self.patch_embedding(full_ref_combined).flatten(2).transpose(1, 2)  # [B, seq_len, model_dim]

            if full_ref_combined.dim() > 4:
                full_ref_frames_num = full_ref_combined.size(2)
            else:
                full_ref_frames_num = 1

            full_ref_token_num = full_ref_embedded.size(1)
            seq_len += full_ref_token_num

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]).to(device)

        # Store original grid sizes for main content (without ref)
        x_grid_sizes = grid_sizes.clone()

        # Calculate ref_grid_sizes if full_ref exists
        ref_grid_sizes = None

        x = [u.flatten(2).transpose(1, 2) for u in x] # [B, seq_len, model_dim]

        # Concat full_ref in front of x
        if full_ref is not None:
            # Calculate ref_grid_sizes from full_ref_embedded shape
            # full_ref_embedded is [B, ref_tokens, dim]
            # ref_tokens = ref_frames * h_tokens * w_tokens
            # where h_tokens and w_tokens match x_grid_sizes
            ref_grid_sizes = torch.stack([
                torch.tensor([full_ref_frames_num, x_grid_sizes[i][1], x_grid_sizes[i][2]], dtype=torch.long)
                for i in range(len(x))
            ]).to(device=device, dtype=torch.long)

            grid_sizes = torch.stack([torch.tensor([u[0] + full_ref_frames_num, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref_embedded, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([padding, t], dim=1)

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long).to(device)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast('cuda',dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None and hasattr(self, 'img_emb'):
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]

        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                if t.dim() != 1:
                    modulated_inp = e0[:, -1, :]
                else:
                    modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc

        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            x_grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            t,
                            ref_grid_sizes,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=x_grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            t=t,
                            ref_grid_sizes=ref_grid_sizes
                        )
                        x = block(x, **kwargs)

                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        x_grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        dtype,
                        t,
                        ref_grid_sizes,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=x_grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype,
                        t=t,
                        ref_grid_sizes=ref_grid_sizes
                    )
                    x = block(x, **kwargs)

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # Remove full_ref tokens from output (use saved full_ref_token_num and full_ref_frames_num)
        if full_ref_token_num > 0:
            x = x[:, full_ref_token_num:]
            grid_sizes = torch.stack([torch.tensor([u[0] - full_ref_frames_num, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x


class CroodRefTransformer3DModel2_2(Wan2_2Transformer3DModel):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='i2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=148,      # 48 noisy + 48 fg_coordmap + 4 mask + 48 appearance
        dim=3072,        # 5B model hidden dimension
        ffn_dim=14336,   # 5B model FFN dimension
        freq_dim=256,
        text_dim=4096,
        out_dim=48,      # 5B VAE latent channels
        num_heads=24,    # 5B model attention heads
        num_layers=30,   # 5B model transformer layers
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=148,
        hidden_size=3072,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,  # Not using ref_conv, use patch_embedding instead
        in_dim_ref_conv=48,
        rope_gap=5,
    ):
        r"""
        Initialize the CroodRef 2.2 5B diffusion model backbone.


        Args:
            model_type (`str`, *optional*, defaults to 'i2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 148):
                Input channels: 48 (noisy) + 48 (fg_coordmap) + 4 (mask) + 48 (appearance)
            dim (`int`, *optional*, defaults to 3072):
                Hidden dimension of the transformer (5B model)
            ffn_dim (`int`, *optional*, defaults to 14336):
                Intermediate dimension in feed-forward network (5B model)
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 48):
                Output video channels (5B VAE latent channels)
            num_heads (`int`, *optional*, defaults to 24):
                Number of attention heads (5B model)
            num_layers (`int`, *optional*, defaults to 30):
                Number of transformer blocks (5B model)
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to True):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            rope_gap (`int`, *optional*, defaults to 5):
                Gap multiplier for reference frame RoPE positions (e.g., -5, -10, -15...)
        """
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=False,
            in_dim_control_adapter=in_dim_control_adapter,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
            add_ref_conv=False,
            in_dim_ref_conv=in_dim_ref_conv,
        )

        if hasattr(self, "control_adapter"):
            del self.control_adapter
        if hasattr(self, "ref_conv"):
            del self.ref_conv

        self.latent_channels = 48
        self.rope_gap = rope_gap
        cross_attn_type = 'cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlockWithRef(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, rope_gap)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        original_path = pretrained_model_path

        if subfolder is not None:
            subfolder_path = os.path.join(pretrained_model_path, subfolder)
            if os.path.isfile(os.path.join(subfolder_path, 'config.json')):
                pretrained_model_path = subfolder_path

        # Check if config.json exists at current path
        config_file = os.path.join(pretrained_model_path, 'config.json')

        # If not found, try parent directory (for pretrained models where config is in root)
        if not os.path.isfile(config_file):
            parent_path = os.path.dirname(pretrained_model_path)
            parent_config_file = os.path.join(parent_path, 'config.json')
            if os.path.isfile(parent_config_file):
                pretrained_model_path = parent_path
                config_file = parent_config_file

        print(f"loaded {cls.__name__} from {pretrained_model_path} ...")

        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist. Checked paths: {original_path}, {os.path.dirname(original_path)}")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            from diffusers.models.model_loading_utils import load_model_dict_into_meta
            from diffusers.utils import is_accelerate_available

            if not is_accelerate_available():
                raise ImportError(
                    "low_cpu_mem_usage=True requires Accelerate. Install accelerate or set "
                    "low_cpu_mem_usage=False."
                )

            import accelerate

            # Construct parameters directly on the meta device. In particular, this avoids
            # materializing the 5B backbone once in fp32 before converting it to torch_dtype.
            with accelerate.init_empty_weights():
                model = cls.from_config(config, **transformer_additional_kwargs)

            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location="cpu")
            elif os.path.exists(model_file_safetensors):
                from safetensors.torch import load_file

                state_dict = load_file(model_file_safetensors)
            else:
                from safetensors.torch import load_file

                model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                state_dict = {}
                for sharded_model_file in model_files_safetensors:
                    state_dict.update(load_file(sharded_model_file))

            model_state_dict = model.state_dict()

            # Preserve the existing CoordRef carrier expansion: copy the pretrained
            # channels and initialize the extra carrier channels to zero.
            if "patch_embedding.weight" in state_dict:
                pretrained_weight = state_dict["patch_embedding.weight"]
                pretrained_channels = pretrained_weight.size(1)
                model_channels = model_state_dict["patch_embedding.weight"].size(1)
                if pretrained_channels != model_channels:
                    if pretrained_channels > model_channels:
                        raise ValueError(
                            "Cannot shrink patch_embedding while loading a base checkpoint: "
                            f"checkpoint has {pretrained_channels} channels, model has {model_channels}."
                        )
                    expanded_weight = pretrained_weight.new_zeros(
                        model_state_dict["patch_embedding.weight"].shape
                    )
                    expanded_weight[:, :pretrained_channels] = pretrained_weight
                    state_dict["patch_embedding.weight"] = expanded_weight
                    print(
                        f"### Expanding patch_embedding from {pretrained_channels} to "
                        f"{model_channels} channels"
                    )

            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith(("ref_conv.", "control_adapter.")):
                    print(key, f"Skipping ref_conv/control_adapter for {cls.__name__}")
                    continue
                if key.startswith("img_emb."):
                    print(key, f"Skipping img_emb for {cls.__name__} (CLIP compatibility only)")
                    continue
                if key in model_state_dict and model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(key, "Size don't match, skip")

            missing_keys = sorted(set(model_state_dict) - set(filtered_state_dict))
            if missing_keys:
                raise ValueError(
                    f"Cannot load {cls.__name__} from {pretrained_model_path} with "
                    "low_cpu_mem_usage=True because parameters would remain on the meta device. "
                    f"Missing keys: {missing_keys[:20]}"
                )

            load_model_dict_into_meta(
                model,
                filtered_state_dict,
                dtype=torch_dtype,
                model_name_or_path=pretrained_model_path,
            )

            remaining_meta_keys = [
                key for key, value in model.state_dict().items() if value.device.type == "meta"
            ]
            if remaining_meta_keys:
                raise RuntimeError(
                    "Low-memory checkpoint loading left tensors on the meta device: "
                    f"{remaining_meta_keys[:20]}"
                )
            return model

        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]

        # Handle patch_embedding channel expansion when a CoordRef carrier has more
        # input channels than the checkpoint it is initialized from.
        if 'patch_embedding.weight' in state_dict:
            pretrained_channels = state_dict['patch_embedding.weight'].size(1)
            model_channels = model.state_dict()['patch_embedding.weight'].size(1)

            if pretrained_channels != model_channels:
                print(f"### Expanding patch_embedding from {pretrained_channels} to {model_channels} channels")
                new_weight = torch.zeros_like(model.state_dict()['patch_embedding.weight'])
                new_weight[:, :pretrained_channels, :, :, :] = state_dict['patch_embedding.weight']
                state_dict['patch_embedding.weight'] = new_weight
                print(f"### First {pretrained_channels} channels copied from pretrained, remaining {model_channels - pretrained_channels} channels zero-initialized")

        tmp_state_dict = {}
        for key in state_dict:
            # Skip ref_conv and control_adapter (CroodRef 2.2 doesn't use them)
            if key.startswith('ref_conv.') or key.startswith('control_adapter.'):
                print(key, f"Skipping ref_conv/control_adapter for {cls.__name__}")
                continue
            # Skip img_emb; CoordRef keeps CLIP as dummy-zero compatibility only.
            if key.startswith('img_emb.'):
                print(key, f"Skipping img_emb for {cls.__name__} (CLIP compatibility only)")
                continue

            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")

        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")

        model = model.to(torch_dtype)
        return model

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,  # Not used, kept for interface compatibility
        full_ref=None,
        full_ref_crood=None,
        fg_coordmap=None,
        appearance=None,
        cond_flag=True,
    ):
        r"""
        Forward pass through the Wan2.2 CoordRef diffusion model.

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                Dummy-zero compatibility feature; not used by CoordRef.
            full_ref (Tensor, *optional*):
                Full reference frames, latent_channels channels.
            full_ref_crood (Tensor, *optional*):
                Coordinate map for full reference frames, latent_channels channels.
            fg_coordmap (Tensor, *optional*):
                Foreground coordinate map, latent_channels channels.
            appearance (Tensor, *optional*):
                Appearance latents, latent_channels channels.
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            Tensor: Denoised video tensor with shape [B, C_out, F, H, W]
        """
        first_x = x[0] if isinstance(x, list) else x
        device = first_x.device
        dtype = first_x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        # Get full_ref frame count for mask building
        full_ref_frames_num = 0
        if full_ref is not None:
            if full_ref.dim() > 4:
                full_ref_frames_num = full_ref.size(2)
            else:
                full_ref_frames_num = 1

        if fg_coordmap is not None:
            batch_size, _, num_frames, h, w = fg_coordmap.shape
        else:
            # If fg_coordmap is None, use x dimensions
            # print("first_x.shape:", first_x.shape)
            num_frames = first_x.shape[-3]  # x shape: [C, F, H, W]
            h, w = first_x.shape[-2], first_x.shape[-1]
            batch_size = len(x) if isinstance(x, list) else x.shape[0]
        latent_channels = getattr(self, "latent_channels", 48)

        mask = torch.zeros(batch_size, 4, num_frames, h, w, device=device, dtype=dtype) # [B, C, F, H, W]
        # breakpoint()
        # control_input: fg_coordmap(48) + mask(4) + appearance(48)
        if fg_coordmap is not None and appearance is not None:
            control_input = torch.cat([fg_coordmap, mask, appearance], dim=1)
        elif fg_coordmap is not None:
            control_input = torch.cat([fg_coordmap, mask, torch.zeros_like(fg_coordmap)], dim=1)
        elif appearance is not None:
            # print("appearance.shape:", appearance.shape)
            # print("mask.shape:", mask.shape)
            # print("x.shape:", x.shape)
            control_input = torch.cat([torch.zeros_like(appearance), mask, appearance], dim=1)
        else:
            # No control input, create zeros
            zeros_latent = torch.zeros(batch_size, latent_channels, num_frames, h, w, device=device, dtype=dtype)
            control_input = torch.cat([zeros_latent, mask, zeros_latent], dim=1)

        x = [torch.cat([u, v], dim=0) for u, v in zip(x, control_input)]

        full_ref_token_num = 0
        if full_ref is not None:
            # Ensure full_ref is 5D [B, C, F, H, W]
            if full_ref.dim() == 4:
                full_ref = full_ref.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
            if full_ref_crood is not None and full_ref_crood.dim() == 4:
                full_ref_crood = full_ref_crood.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

            full_ref_mask = torch.ones(
                full_ref.shape[0], 4, *full_ref.shape[2:],
                device=device, dtype=dtype
            )  # ref carrier mask slot is fixed to one, matching 5B CoordRef.

            if full_ref_crood is not None:
                full_ref_combined = torch.cat([
                    full_ref,
                    full_ref_crood,
                    full_ref_mask,
                    full_ref,
                ], dim=1)
            else:
                full_ref_combined = torch.cat([
                    full_ref,
                    torch.zeros_like(full_ref),
                    full_ref_mask,
                    full_ref,
                ], dim=1)

            full_ref_embedded = self.patch_embedding(full_ref_combined).flatten(2).transpose(1, 2)

            if full_ref_combined.dim() > 4:
                full_ref_frames_num = full_ref_combined.size(2)
            else:
                full_ref_frames_num = 1

            full_ref_token_num = full_ref_embedded.size(1)
            seq_len += full_ref_token_num

        # patch_embedding for x
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]).to(device)

        x_grid_sizes = grid_sizes.clone()
        ref_grid_sizes = None

        x = [u.flatten(2).transpose(1, 2) for u in x]  # [B, seq_len, model_dim]

        # Prepend full_ref_embedded to x if exists
        if full_ref is not None and full_ref_token_num > 0:
            # full_ref_embedded: [B, ref_seq, dim], each x element: [1, seq, dim]
            x = [torch.cat([full_ref_embedded[i:i+1], u], dim=1) for i, u in enumerate(x)]
            # Update grid_sizes to include full_ref frames
            grid_sizes = torch.stack([
                torch.tensor([u[0] + full_ref_frames_num, u[1], u[2]], dtype=torch.long)
                for u in grid_sizes
            ]).to(device)

            # Calculate ref_grid_sizes for RoPE (used in WanAttentionBlockWithRef)
            ref_grid_sizes = torch.stack([
                torch.tensor([full_ref_frames_num, u[1], u[2]], dtype=torch.long)
                for u in x_grid_sizes
            ]).to(device)

        # sequence length processing
        # Each x element has shape [1, total_seq, dim], so use size(1) for seq length
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long).to(device)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast('cuda', dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank] if e0.dim() > 2 else e0
            e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank] if e.dim() > 2 else e

        # TeaCache handling (same as parent)
        if self.teacache is not None:
            if cond_flag:
                modulated_inp = e0
                if self.teacache.cnt == 0 or self.teacache.cnt == self.teacache.num_steps:
                    self.teacache.accumulated_rel_l1_distance = 0
                    self.should_calc = True
                else:
                    coefficients = self.teacache.coefficients
                    rescale_func = np.poly1d(coefficients)
                    self.teacache.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.teacache.previous_modulated_input).abs().mean() / self.teacache.previous_modulated_input.abs().mean()).cpu().item())
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc

        # TeaCache block processing
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)
                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, e0, seq_lens, x_grid_sizes, self.freqs, context, context_lens, dtype, t, ref_grid_sizes,
                            **ckpt_kwargs,
                        )
                    else:
                        kwargs = dict(
                            e=e0, seq_lens=seq_lens, grid_sizes=x_grid_sizes, freqs=self.freqs,
                            context=context, context_lens=context_lens, dtype=dtype, t=t, ref_grid_sizes=ref_grid_sizes
                        )
                        x = block(x, **kwargs)

                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, e0, seq_lens, x_grid_sizes, self.freqs, context, context_lens, dtype, t, ref_grid_sizes,
                        **ckpt_kwargs,
                    )
                else:
                    kwargs = dict(
                        e=e0, seq_lens=seq_lens, grid_sizes=x_grid_sizes, freqs=self.freqs,
                        context=context, context_lens=context_lens, dtype=dtype, t=t, ref_grid_sizes=ref_grid_sizes
                    )
                    x = block(x, **kwargs)

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # Remove full_ref tokens from output
        if full_ref_token_num > 0:
            x = x[:, full_ref_token_num:]
            grid_sizes = torch.stack([torch.tensor([u[0] - full_ref_frames_num, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x
