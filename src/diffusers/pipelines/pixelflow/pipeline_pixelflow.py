from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from ...models.transformers.transformer_pixelflow import PixelFlowTransformer2DModel
from ...schedulers.scheduling_pixelflow import PixelFlowScheduler
from ..._hf import load_hf_diffusers_submodules

_hf = load_hf_diffusers_submodules(
    "image_processor",
    "models.embeddings",
    "pipelines.pipeline_utils",
    "utils",
    "utils.torch_utils",
)
VaeImageProcessor = _hf["image_processor"].VaeImageProcessor
get_2d_rotary_pos_embed = _hf["models.embeddings"].get_2d_rotary_pos_embed
DiffusionPipeline = _hf["pipelines.pipeline_utils"].DiffusionPipeline
BaseOutput = _hf["utils"].BaseOutput
randn_tensor = _hf["utils.torch_utils"].randn_tensor


@dataclass
class PixelFlowPipelineOutput(BaseOutput):
    images: Union[torch.Tensor, List, np.ndarray]


class PixelFlowPipeline(DiffusionPipeline):
    """Pipeline for PixelFlow pixel-space flow generation (class-conditional or text-to-image)."""

    model_cpu_offload_seq = "text_encoder->transformer"
    _optional_components = ["text_encoder", "tokenizer"]

    def __init__(
        self,
        transformer: PixelFlowTransformer2DModel,
        scheduler: PixelFlowScheduler,
        text_encoder=None,
        tokenizer=None,
        max_token_length: int = 512,
    ):
        super().__init__()
        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False)
        self.class_cond = transformer.config.num_classes > 0
        self.max_token_length = max_token_length

    def sample_block_noise(self, bs, ch, height, width, eps=1e-6):
        gamma = self.scheduler.gamma
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(4),
            torch.eye(4) * (1 - gamma) + torch.ones(4, 4) * gamma + eps * torch.eye(4),
        )
        block_number = bs * ch * (height // 2) * (width // 2)
        noise = torch.stack([dist.sample() for _ in range(block_number)])
        noise = rearrange(
            noise,
            "(b c h w) (p q) -> b c (h p) (w q)",
            b=bs,
            c=ch,
            h=height // 2,
            w=width // 2,
            p=2,
            q=2,
        )
        return noise

    def _stage_guidance_scale(self, stage_idx: int) -> float:
        if not self.class_cond:
            return self._guidance_scale_value
        scale_dict = {0: 0, 1: 1 / 6, 2: 2 / 3, 3: 1}
        return (self._guidance_scale_value - 1) * scale_dict[stage_idx] + 1

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale_value > 0

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = "",
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Text-to-image generation requires `text_encoder` and `tokenizer`.")

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        max_length = max_length or self.max_token_length

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = self.text_encoder(
            text_input_ids,
            attention_mask=prompt_attention_mask,
        )[0]

        dtype = self.text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1).repeat(num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            if isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif isinstance(negative_prompt, list):
                if len(negative_prompt) != batch_size:
                    raise ValueError(
                        f"Negative prompt list length ({len(negative_prompt)}) must match prompt batch ({batch_size})."
                    )
                uncond_tokens = negative_prompt
            else:
                raise ValueError("Negative prompt must be a string or list of strings.")

            uncond_inputs = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_input_ids = uncond_inputs.input_ids.to(device)
            negative_prompt_attention_mask = uncond_inputs.attention_mask.to(device)
            negative_prompt_embeds = self.text_encoder(
                negative_input_ids,
                attention_mask=negative_prompt_attention_mask,
            )[0]

            seq_len_neg = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len_neg, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1).repeat(num_images_per_prompt, 1)

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        return prompt_embeds, prompt_attention_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        class_labels: Optional[Union[int, List[int], torch.Tensor]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Union[int, List[int]] = 10,
        guidance_scale: float = 4.0,
        shift: float = 1.0,
        negative_prompt: Union[str, List[str]] = "",
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[PixelFlowPipelineOutput, Tuple]:
        if height is None:
            height = int(self.transformer.config.sample_size)
        if width is None:
            width = int(self.transformer.config.sample_size)

        device = self._execution_device
        self._guidance_scale_value = guidance_scale

        if isinstance(num_inference_steps, int):
            num_inference_steps = [num_inference_steps] * self.scheduler.num_stages

        prompt_attention_mask = None
        if self.class_cond:
            if class_labels is None:
                raise ValueError("`class_labels` are required for class-conditional PixelFlow checkpoints.")
            if isinstance(class_labels, int):
                class_labels = [class_labels]
            if not torch.is_tensor(class_labels):
                class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
            else:
                class_labels = class_labels.to(device=device, dtype=torch.long)

            batch_size = class_labels.shape[0]
            prompt_embeds = class_labels
            negative_prompt_embeds = torch.full_like(prompt_embeds, self.transformer.config.num_classes)
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            if prompt is None:
                raise ValueError("`prompt` is required for text-to-image PixelFlow checkpoints.")
            if isinstance(prompt, str):
                prompt = [prompt]
            batch_size = len(prompt)
            prompt_embeds, prompt_attention_mask = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance and guidance_scale > 1.0,
                negative_prompt=negative_prompt,
            )

        init_factor = 2 ** (self.scheduler.num_stages - 1)
        height, width = height // init_factor, width // init_factor
        latents = randn_tensor(
            (batch_size * num_images_per_prompt, 3, height, width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        for stage_idx in range(self.scheduler.num_stages):
            self.scheduler.set_timesteps(num_inference_steps[stage_idx], stage_idx, device=device, shift=shift)
            timesteps = self.scheduler.Timesteps

            if stage_idx > 0:
                height, width = height * 2, width * 2
                latents = F.interpolate(latents, size=(height, width), mode="nearest")
                original_start_t = self.scheduler.original_start_t[stage_idx]
                gamma = self.scheduler.gamma
                alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
                beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)

                noise = self.sample_block_noise(*latents.shape)
                noise = noise.to(device=device, dtype=latents.dtype)
                latents = alpha * latents + beta * noise

            size_tensor = torch.tensor([latents.shape[-1] // self.transformer.patch_size], dtype=torch.int32, device=device)
            pos_embed = get_2d_rotary_pos_embed(
                embed_dim=self.transformer.attention_head_dim,
                crops_coords=((0, 0), (latents.shape[-1] // self.transformer.patch_size, latents.shape[-1] // self.transformer.patch_size)),
                grid_size=(latents.shape[-1] // self.transformer.patch_size, latents.shape[-1] // self.transformer.patch_size),
                device=device,
                output_type="pt",
            )
            rope_pos = torch.stack(pos_embed, -1)

            autocast_enabled = device.type == "cuda"
            autocast_dtype = torch.bfloat16 if autocast_enabled else torch.float32
            for timestep in timesteps:
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep_batch = timestep.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
                with torch.autocast(device.type, enabled=autocast_enabled, dtype=autocast_dtype):
                    if self.class_cond:
                        noise_pred = self.transformer(
                            latent_model_input,
                            timestep=timestep_batch,
                            class_labels=prompt_embeds,
                            latent_size=size_tensor,
                            pos_embed=rope_pos,
                        ).sample
                    else:
                        noise_pred = self.transformer(
                            latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            timestep=timestep_batch,
                            latent_size=size_tensor,
                            pos_embed=rope_pos,
                        ).sample

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self._stage_guidance_scale(stage_idx) * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(model_output=noise_pred, sample=latents).prev_sample

        image = (latents / 2 + 0.5).clamp(0, 1)

        if output_type == "pt":
            pass
        elif output_type in ("pil", "np"):
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return PixelFlowPipelineOutput(images=image)
