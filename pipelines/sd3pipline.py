# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
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
# limitations under the License.
import re
import torch.nn.functional as F
import inspect
from typing import Any, Callable, Dict, List, Optional, Union,Tuple
import numpy as np
import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from tqdm import tqdm
from torch.optim.adam import Adam
from utils3.ptp_utils import AttnProcessor, AttentionStore
from utils3.attn_utils import fn_smoothing_func, fn_get_topk, fn_clean_mask, fn_get_otsu_mask, fn_show_attention_plus_2
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
o_attr = False
o_self = False
o_attri_activate = False
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
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
    r"""
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


class StableDiffusion3Pipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
    ):
        super().__init__()
        self.cross_attention_maps_cache=None
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

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

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if (
            height % (self.vae_scale_factor * self.patch_size) != 0
            or width % (self.vae_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * self.patch_size} but are {height} and {width}."
                f"You can use height {height - height % (self.vae_scale_factor * self.patch_size)} and width {width - width % (self.vae_scale_factor * self.patch_size)}."
            )

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
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_3 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

###########################################################################
    def fn_initno1(
            self,
            latents: torch.Tensor,
            indices: List[List[Union[int,str]]],
            text_embeddings: torch.Tensor,
            pooled_text_embeddings: torch.Tensor,
            initno_lr: float = 1e-2,
            max_step: int = 20,
            round: int = 0,
            tau_cross_attn: float = 0.2,
            tau_self_attn: float = 0.3,
            num_inference_steps: int = 50,
            device: str = "",
            denoising_step_for_loss: int = 1,
            guidance_scale: float = 1.0,
            eta: float = 0.0,
            do_classifier_free_guidance: bool = False,
            K: int = 1,
            attention_res: int = 64,
            from_where: List[int] = None
    ):
        '''InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization'''

        latents = latents.clone().detach()
        log_var, mu = torch.zeros_like(latents), torch.zeros_like(latents)
        log_var, mu = log_var.clone().detach().requires_grad_(True), mu.clone().detach().requires_grad_(True)
        optimizer = Adam([log_var, mu], lr=initno_lr, eps=1e-3)

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        optimization_succeed = False
        for iteration in tqdm(range(max_step)):

            optimized_latents = latents * (torch.exp(0.5 * log_var)) + mu

            # prepare scheduler
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # loss records
            joint_loss_list, cross_attn_loss_list, self_cross_attn_loss_list = [], [], []

            # denoising loop
            for i, t in enumerate(timesteps):
                if i >= denoising_step_for_loss: break
                timestep = t.expand(optimized_latents.shape[0])


                # Forward pass of denoising with text conditioning
                noise_pred_text = self.transformer(
                    hidden_states=optimized_latents,
                    timestep=timestep,
                    encoder_hidden_states=text_embeddings[1].unsqueeze(0),  # take the positive prompt
                    pooled_projections=pooled_text_embeddings[1].unsqueeze(0),  # take the positive prompt
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False, )[0]

                joint_loss, cross_attn_loss, self_cross_attn_loss = self.fn_compute_loss1(indices=indices, K=K,
                                                                                         attention_res=attention_res, from_where=from_where,
                                                                                         t5_indices=self.t5_token_indices)
                print("joint_loss:", joint_loss.item(), "cross_attn_loss:", cross_attn_loss.item(), "self_cross_attn_loss:", self_cross_attn_loss.item())
                joint_loss_list.append(joint_loss), cross_attn_loss_list.append(cross_attn_loss), self_cross_attn_loss_list.append(self_cross_attn_loss)

                if denoising_step_for_loss > 1:
                    with torch.no_grad():
                        noise_pred_uncond = self.transformer(
                            hidden_states=optimized_latents,
                            timestep=timestep,
                            encoder_hidden_states=text_embeddings[0].unsqueeze(0),  # take the negative prompt
                            pooled_projections=pooled_text_embeddings[0].unsqueeze(0),  # take the negative prompt
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False, )[0]

                    if do_classifier_free_guidance: noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond)
                    else: noise_pred = noise_pred_text
                    # compute the previous noisy sample x_t -> x_t-1
                    optimized_latents = self.scheduler.step(noise_pred, t, optimized_latents, return_dict=False)[0]

            joint_loss = sum(joint_loss_list) / denoising_step_for_loss
            cross_attn_loss = max(cross_attn_loss_list)
            self_cross_attn_loss = max(self_cross_attn_loss_list)


            if cross_attn_loss < tau_cross_attn and self_cross_attn_loss < tau_self_attn:
                optimization_succeed = True
                break

            self.transformer.zero_grad()
            optimizer.zero_grad()
            joint_loss = joint_loss.mean()
            joint_loss.backward()
            optimizer.step()

            # update kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            while kld_loss > 0.001:
                optimizer.zero_grad()
                kld_loss = kld_loss.mean()
                kld_loss.backward()
                optimizer.step()
                kld_loss = self.fn_calc_kld_loss_func(log_var, mu)

        optimized_latents = (latents * (torch.exp(0.5 * log_var)) + mu).clone().detach()
        return optimized_latents, optimization_succeed, cross_attn_loss + self_cross_attn_loss
###########################################################################

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.transformer.attn_processors.keys():
            if cross_att_count in self.from_where:
                place_in_transformer = cross_att_count
                attn_procs[name] = AttnProcessor(attnstore=self.attention_store, place_in_transformer=place_in_transformer, from_where=self.from_where, use_t5=self.use_t5)
            else:
                attn_procs[name] = self.transformer.attn_processors[name]
            if name.startswith(f"transformer_blocks.{cross_att_count}.attn"):
                cross_att_count += 1
        self.transformer.set_attn_processor(attn_procs)
        # When use_t5, T5 cross attn also triggers __call__ (counted separately), but we use same layer counter
        self.attention_store.num_att_layers = len(self.from_where) * 2 

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """Update the latent according to the computed loss."""

        try:
            grad_cond = torch.autograd.grad(loss, [latents], retain_graph=True)[0]
        except RuntimeError as e:
            print(f"[错误] 计算图断裂或 latents 没有 requires_grad: {e}")
            return latents

        if grad_cond is not None:
            # grad_norm = torch.norm(grad_cond)

            # 打印归一化之前的原始统计（方便你继续监控）
            # grad_max = grad_cond.max().item()
            # grad_min = grad_cond.min().item()
            # grad_mean = grad_cond.mean().item()
            # print(
            #     f"Grad Stats | Raw Norm: {grad_norm.item():.4f} | Max(Normalized): {grad_max:.4f} | Min: {grad_min:.4f} | Mean: {grad_mean:.4e}")

            # 缩放倍数放大：如果归一化后发现步子还不够大，可以临时把 step_size 乘个系数（比如 * 10）
            # 注意：使用了归一化后，step_size 的绝对值就需要重新微调了。
            latents = latents - step_size * grad_cond
            # 也可以尝试使用符号梯度：latents = latents - step_size * torch.sign(grad_cond) 效果有时候更猛

        return latents

    def fn_initno(
            self,
            latents: torch.Tensor,
            indices: List[int],
            text_embeddings: torch.Tensor,
            pooled_text_embeddings: torch.Tensor,
            initno_lr: float = 1e-2,
            max_step: int = 50,
            round: int = 0,
            tau_cross_attn: float = 0.2,
            tau_self_attn: float = 0.3,
            num_inference_steps: int = 50,
            device: str = "",
            denoising_step_for_loss: int = 1,
            guidance_scale: float = 1.0,
            eta: float = 0.0,
            do_classifier_free_guidance: bool = False,
            K: int = 1,
            attention_res: int = 64,
            from_where: List[int] = None
    ):
        '''InitNO: Boosting Text-to-Image Diffusion Models via Initial Noise Optimization'''

        latents = latents.clone().detach()
        log_var, mu = torch.zeros_like(latents), torch.zeros_like(latents)
        log_var, mu = log_var.clone().detach().requires_grad_(True), mu.clone().detach().requires_grad_(True)
        optimizer = Adam([log_var, mu], lr=initno_lr, eps=1e-3)

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        optimization_succeed = False
        for iteration in tqdm(range(max_step)):

            optimized_latents = latents * (torch.exp(0.5 * log_var)) + mu

            # prepare scheduler
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # loss records
            joint_loss_list, cross_attn_loss_list, self_cross_attn_loss_list = [], [], []

            # denoising loop
            for i, t in enumerate(timesteps):
                if i >= denoising_step_for_loss: break
                timestep = t.expand(optimized_latents.shape[0])


                # Forward pass of denoising with text conditioning
                noise_pred_text = self.transformer(
                    hidden_states=optimized_latents,
                    timestep=timestep,
                    encoder_hidden_states=text_embeddings[1].unsqueeze(0),  # take the positive prompt
                    pooled_projections=pooled_text_embeddings[1].unsqueeze(0),  # take the positive prompt
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False, )[0]

                joint_loss, cross_attn_loss, self_cross_attn_loss = self.fn_compute_loss(indices=indices, K=K,
                                                                                         attention_res=attention_res, from_where=from_where)
                joint_loss_list.append(joint_loss), cross_attn_loss_list.append(cross_attn_loss), self_cross_attn_loss_list.append(self_cross_attn_loss)

                if denoising_step_for_loss > 1:
                    with torch.no_grad():
                        noise_pred_uncond = self.transformer(
                            hidden_states=optimized_latents,
                            timestep=timestep,
                            encoder_hidden_states=text_embeddings[0].unsqueeze(0),  # take the negative prompt
                            pooled_projections=pooled_text_embeddings[0].unsqueeze(0),  # take the negative prompt
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False, )[0]

                    if do_classifier_free_guidance: noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond)
                    else: noise_pred = noise_pred_text
                    # compute the previous noisy sample x_t -> x_t-1
                    optimized_latents = self.scheduler.step(noise_pred, t, optimized_latents, return_dict=False)[0]

            joint_loss = sum(joint_loss_list) / denoising_step_for_loss
            cross_attn_loss = max(cross_attn_loss_list)
            self_cross_attn_loss = max(self_cross_attn_loss_list)

            # # print loss records
            # joint_loss_list = [_.item() for _ in joint_loss_list]
            # cross_attn_loss_list = [_.item() for _ in cross_attn_loss_list]
            # self_cross_attn_loss_list = [_.item() for _ in self_cross_attn_loss_list]

            if cross_attn_loss < tau_cross_attn and self_cross_attn_loss < tau_self_attn:
                optimization_succeed = True
                break

            self.transformer.zero_grad()
            optimizer.zero_grad()
            joint_loss = joint_loss.mean()
            joint_loss.backward()
            optimizer.step()

            # update kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            kld_loss = self.fn_calc_kld_loss_func(log_var, mu)
            while kld_loss > 0.001:
                optimizer.zero_grad()
                kld_loss = kld_loss.mean()
                kld_loss.backward()
                optimizer.step()
                kld_loss = self.fn_calc_kld_loss_func(log_var, mu)

        optimized_latents = (latents * (torch.exp(0.5 * log_var)) + mu).clone().detach()
        # if self_attn_loss <= 1e-6: self_attn_loss = self_attn_loss + 1.
        return optimized_latents, optimization_succeed, cross_attn_loss + self_cross_attn_loss

    def fn_compute_loss1(
            self,
            indices: List[List[Union[int , str]]],  #[[2, 3:5], [6, 7]]
            smooth_attentions: bool = True,
            K: int = 1,
            attention_res: int = 64,
            from_where: List[int] = None,
            t5_indices=None) -> torch.Tensor:

        # -----------------------------
        # 1. 预处理与嵌套索引解析
        # -----------------------------
        aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
            from_where=from_where, is_cross=True)

        cross_attention_maps = aggregate_cross_attention_maps[:, :, 0:-1]
        cross_attention_maps = cross_attention_maps * 100
        cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps.float(), dim=-1)

        # T5 cross-attention maps (optional)
        t5_attn_maps = None
        if self.use_t5:
            _t5_raw = self.attention_store.aggregate_attention(from_where=from_where, is_cross="t5")
            if _t5_raw is not None:
                t5_attn_maps = _t5_raw * 100
                t5_attn_maps = torch.nn.functional.softmax(t5_attn_maps.float(), dim=-1)

        def _parse_idx(item):
            """Parse a single index item (int or 'a:b' str) into an int or tuple."""
            if isinstance(item, str) and ":" in item:
                s, e = map(int, item.split(":"))
                return tuple(range(s, e))
            return int(item)

        def _get_attn_map(clip_idx, t5_idx=None):
            """Return per-token attention map, optionally merged with T5 via element-wise max."""
            if isinstance(clip_idx, tuple):
                clip_map = cross_attention_maps[:, :, list(clip_idx)].max(dim=-1)[0]
            else:
                clip_map = cross_attention_maps[:, :, clip_idx]
            if self.use_t5 and t5_attn_maps is not None and t5_idx is not None:
                if isinstance(t5_idx, tuple):
                    t5_map = t5_attn_maps[:, :, list(t5_idx)].max(dim=-1)[0]
                else:
                    t5_map = t5_attn_maps[:, :, t5_idx]
                clip_map = torch.max(clip_map, t5_map)
            return clip_map

        entity_indices = []  # 用于存放名词（实体）的索引
        attribute_indices = []  # 用于存放属性词（修饰词）的索引
        all_shifted_indices = []
        shifted_binding_pairs = []
        # T5 parallel mapping: clip_idx -> t5_idx
        clip_to_t5 = {}
        for g_i, group in enumerate(indices):
            if not isinstance(group, list):
                group = [group]

            # 所有索引减 1 (偏移)
            shifted_group = []
            for item in group:
                if isinstance(item, str) and ":" in item:
                    start, end = map(int, item.split(":"))
                    # 使用 tuple 而不是 list，防止后续 dict.fromkeys 去重时报错
                    shifted_group.append(tuple(idx for idx in range(start, end)))
                else:
                    shifted_group.append(int(item))
            all_shifted_indices.extend(shifted_group)

            # Build clip -> t5 index mapping for this group (positional alignment)
            if self.use_t5 and t5_indices is not None and g_i < len(t5_indices):
                t5_group = t5_indices[g_i]
                if not isinstance(t5_group, list):
                    t5_group = [t5_group]
                for pos, clip_item in enumerate(shifted_group):
                    if pos < len(t5_group):
                        t5_parsed = _parse_idx(t5_group[pos])
                        clip_to_t5[clip_item] = t5_parsed

            if len(shifted_group) > 0:
                # 列表的最后一个元素作为名词
                ent = shifted_group[-1]
                entity_indices.append(ent)

                # 列表前面所有的元素作为属性词，批量添加到属性词列表中
                for attr in shifted_group[:-1]:
                    attribute_indices.append(attr)
                    shifted_binding_pairs.append((attr, ent))

        # 去重并保持顺序（防止不同 group 中出现重复索引）
        entity_indices = list(dict.fromkeys(entity_indices))
        attribute_indices = list(dict.fromkeys(attribute_indices))

        # -----------------------------
        # 2. 计算所有 Token 的激发损失 (Excitation)
        # -----------------------------
        cross_attn_loss_list = []
        clean_cross_attention_loss = 0.
        entity_otsu_masks = {}
        entity_cross_attn_maps = {}
        topk_average_cross_attn_value_list = []
        for idx in entity_indices:
            attn_map = _get_attn_map(idx, clip_to_t5.get(idx))
            if smooth_attentions: attn_map = fn_smoothing_func(attn_map)
            entity_cross_attn_maps[idx]= attn_map
            topk_coords, topk_vals = fn_get_topk(attn_map, K=K)

            clean_mask = fn_get_otsu_mask(attn_map)
            entity_otsu_masks[idx] = clean_mask

            clean_mask = fn_clean_mask(clean_mask, topk_coords[0][0], topk_coords[0][1])
            foreground = attn_map * clean_mask + (1 - clean_mask)
            background = attn_map * (1 - clean_mask)
            if background.max() > foreground.min():
                clean_cross_attention_loss += background.max()
            else:
                clean_cross_attention_loss += background.max() * 0.0

            val = torch.mean(torch.stack(topk_vals))
            topk_average_cross_attn_value_list.append(val)
        cross_attn_loss_list = [max(0 * curr_max, 0.8 - curr_max) for curr_max in topk_average_cross_attn_value_list]
        cross_attn_loss = max(cross_attn_loss_list) if cross_attn_loss_list else 0


        # ---------------------------------
        # 4. 提纯实体软掩码 (Soft Mask)
        # ---------------------------------
        self_attention_maps = self.attention_store.aggregate_attention(from_where=from_where, is_cross=False)
        entity_soft_masks = {}

        for e_idx in entity_indices:
            cross_attn_map = entity_cross_attn_maps[e_idx]
            soft_mask = torch.zeros_like(cross_attn_map)
            hard_mask = entity_otsu_masks[e_idx]

            # Vectorized: weighted sum of self-attn maps masked by hard_mask,
            # then one smoothing pass (valid due to linearity of Gaussian convolution).
            mask_flat = hard_mask.bool().view(-1)                          # (4096,)
            weights   = cross_attn_map.view(-1)[mask_flat]                 # (N,)
            # self_attention_maps: (64,64,4096) → (4096,4096)[mask] → (N,4096)
            self_maps = self_attention_maps.view(attention_res * attention_res,
                                                 attention_res * attention_res)[mask_flat]  # (N,4096)
            raw_mask  = (weights[:, None] * self_maps).sum(0)              # (4096,)
            raw_mask  = raw_mask.view(attention_res, attention_res)        # (64,64)
            if smooth_attentions: raw_mask = fn_smoothing_func(raw_mask)
            entity_soft_masks[e_idx] = raw_mask

        # ----------------------------------
        # 5. 实体间冲突损失 (Entity Conflict)
        # ----------------------------------
        self_cross_attn_loss = 0
        number_self_cross_loss_pair = 0
        for i in entity_indices:
            cross_i = entity_cross_attn_maps[i] / (torch.sum(entity_cross_attn_maps[i]) + 1e-4)
            for j in entity_indices:
                if i == j: continue
                self_j = entity_soft_masks[j] / (torch.sum(entity_soft_masks[j]) + 1e-4)
                self_cross_attn_loss += torch.min(self_j, cross_i).sum()
                number_self_cross_loss_pair += 1

        if number_self_cross_loss_pair > 0:
            self_cross_attn_loss /= number_self_cross_loss_pair
        else:
            self_cross_attn_loss = torch.tensor(0.0)

        # -------------
        # 6. 最终损失
        # -------------
        if cross_attn_loss > 0.6:
            self_cross_attn_loss = self_cross_attn_loss * 0.0
        if o_self:
            self_cross_attn_loss = self_cross_attn_loss * 0.0


        joint_loss = cross_attn_loss + self_cross_attn_loss + clean_cross_attention_loss * 0.1

        return joint_loss, cross_attn_loss, self_cross_attn_loss

    def fn_augmented_compute_loss1(
            self,
            indices: List[List[Union[int,str]]],
            smooth_attentions: bool = True,
            K: int = 1,
            attention_res: int = 64,
            from_where: List[int] = None,
            index=None,
            t5_indices=None) -> torch.Tensor:

        # -----------------------------
        # 1. 预处理与嵌套索引解析
        # -----------------------------
        aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
            from_where=from_where, is_cross=True)

        cross_attention_maps = aggregate_cross_attention_maps[:, :, 0:-1]
        cross_attention_maps = cross_attention_maps * 100
        cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps.float(), dim=-1)

        # T5 cross-attention maps (optional)
        t5_attn_maps = None
        if self.use_t5:
            _t5_raw = self.attention_store.aggregate_attention(from_where=from_where, is_cross="t5")
            if _t5_raw is not None:
                t5_attn_maps = _t5_raw * 100
                t5_attn_maps = torch.nn.functional.softmax(t5_attn_maps.float(), dim=-1)

        def _parse_idx(item):
            """Parse a single index item (int or 'a:b' str) into an int or tuple."""
            if isinstance(item, str) and ":" in item:
                s, e = map(int, item.split(":"))
                return tuple(range(s, e))
            return int(item)

        def _get_attn_map(clip_idx, t5_idx=None):
            """Return per-token attention map, optionally merged with T5 via element-wise max."""
            if isinstance(clip_idx, tuple):
                clip_map = cross_attention_maps[:, :, list(clip_idx)].max(dim=-1)[0]
            else:
                clip_map = cross_attention_maps[:, :, clip_idx]
            if self.use_t5 and t5_attn_maps is not None and t5_idx is not None:
                if isinstance(t5_idx, tuple):
                    t5_map = t5_attn_maps[:, :, list(t5_idx)].max(dim=-1)[0]
                else:
                    t5_map = t5_attn_maps[:, :, t5_idx]
                clip_map = torch.max(clip_map, t5_map)
            return clip_map

        entity_indices = []
        attribute_indices = []
        shifted_binding_pairs = []
        all_shifted_indices = []
        # T5 parallel mapping: clip_idx -> t5_idx
        clip_to_t5 = {}

        for g_i, group in enumerate(indices):
            if not isinstance(group, list): group = [group]
            shifted_group = []
            for item in group:
                if isinstance(item, str) and ":" in item:
                    start, end = map(int, item.split(":"))
                    shifted_group.append(tuple(idx for idx in range(start, end)))
                else:
                    shifted_group.append(int(item))
            all_shifted_indices.extend(shifted_group)

            # Build clip -> t5 index mapping for this group (positional alignment)
            if self.use_t5 and t5_indices is not None and g_i < len(t5_indices):
                t5_group = t5_indices[g_i]
                if not isinstance(t5_group, list):
                    t5_group = [t5_group]
                for pos, clip_item in enumerate(shifted_group):
                    if pos < len(t5_group):
                        t5_parsed = _parse_idx(t5_group[pos])
                        clip_to_t5[clip_item] = t5_parsed

            if len(shifted_group) > 0:
                ent = shifted_group[-1]
                entity_indices.append(ent)
                for attr in shifted_group[:-1]:
                    attribute_indices.append(attr)
                    shifted_binding_pairs.append((attr, ent))

        entity_indices = list(dict.fromkeys(entity_indices))
        attribute_indices = list(dict.fromkeys(attribute_indices))
        # -----------------------------
        # 2. 计算所有 Token 的激发损失 (Excitation)
        # -----------------------------
        cross_attn_loss_list = []
        clean_cross_attention_loss = 0.
        entity_otsu_masks = {}
        entity_cross_attn_maps = {}
        topk_average_cross_attn_value_list = []
        for idx in entity_indices:
            attn_map = _get_attn_map(idx, clip_to_t5.get(idx))
            if smooth_attentions: attn_map = fn_smoothing_func(attn_map)
            entity_cross_attn_maps[idx]= attn_map
            topk_coords, topk_vals = fn_get_topk(attn_map, K=K)

            clean_mask = fn_get_otsu_mask(attn_map)
            entity_otsu_masks[idx] = clean_mask

            clean_mask = fn_clean_mask(clean_mask, topk_coords[0][0], topk_coords[0][1])
            foreground = attn_map * clean_mask + (1 - clean_mask)
            background = attn_map * (1 - clean_mask)
            if background.max() > foreground.min():
                clean_cross_attention_loss += background.max()
            else:
                clean_cross_attention_loss += background.max() * 0.0

            val = torch.mean(torch.stack(topk_vals))
            topk_average_cross_attn_value_list.append(val)
        cross_attn_loss_list = [max(0 * curr_max, 0.8 - curr_max) for curr_max in topk_average_cross_attn_value_list]
        cross_attn_loss = max(cross_attn_loss_list) if cross_attn_loss_list else 0

        # ---------------------------------
        # 4. 提纯实体软掩码 (Soft Mask)
        # ---------------------------------
        self_attention_maps = self.attention_store.aggregate_attention(from_where=from_where, is_cross=False)
        entity_soft_masks = {}

        for e_idx in entity_indices:
            cross_attn_map = entity_cross_attn_maps[e_idx]
            soft_mask = torch.zeros_like(cross_attn_map)
            hard_mask = entity_otsu_masks[e_idx]

            # Vectorized: weighted sum of self-attn maps masked by hard_mask,
            # then one smoothing pass (valid due to linearity of Gaussian convolution).
            mask_flat = hard_mask.bool().view(-1)                          # (4096,)
            weights   = cross_attn_map.view(-1)[mask_flat]                 # (N,)
            # self_attention_maps: (64,64,4096) → (4096,4096)[mask] → (N,4096)
            self_maps = self_attention_maps.view(attention_res * attention_res,
                                                 attention_res * attention_res)[mask_flat]  # (N,4096)
            raw_mask  = (weights[:, None] * self_maps).sum(0)              # (4096,)
            raw_mask  = raw_mask.view(attention_res, attention_res)        # (64,64)
            if smooth_attentions: raw_mask = fn_smoothing_func(raw_mask)
            entity_soft_masks[e_idx] = raw_mask

        # ----------------------------------
        # 5. 属性绑定损失 (Attribute Binding)
        # ----------------------------------
        attribute_binding_loss = 0.
        activation_loss_list = []
        average_loss_list = []

        activation_loss = 0.
        for attr_idx, ent_idx in shifted_binding_pairs:
            attr_attn = _get_attn_map(attr_idx, clip_to_t5.get(attr_idx))
            if smooth_attentions: attr_attn = fn_smoothing_func(attr_attn)

            ent_soft_mask = entity_soft_masks[ent_idx]
            ent_hard_mask = entity_otsu_masks[ent_idx]

            attr_norm = attr_attn / (torch.sum(attr_attn) + 1e-4)
            attr_peak_norm = attr_norm /(attr_attn.max() + 1e-4)
            ent_soft_norm = ent_soft_mask / (ent_soft_mask.max() + 1e-4)

            temperature = 10.0  # 控制截断的平滑度
            ent_mask_soft_bin = torch.sigmoid((ent_soft_norm - 0.3) * temperature)
            # F.max_pool2d 可以处理软掩码的膨胀
            ent_mask_dilate = F.max_pool2d(ent_mask_soft_bin[None, ...], kernel_size=3, stride=1, padding=1)[0]

            max_inside_hard = (attr_peak_norm * ent_hard_mask).max()
            max_outside_hard = (attr_peak_norm * (1.0 - ent_hard_mask)).max()
            margin = 0.5
            activation_loss = F.relu(margin + max_outside_hard - max_inside_hard)
            activation_loss_list.append(activation_loss)
            # 最大化形容词 attention 在对应实体区域内的质量占比（1 - 区域内占比 = 需最小化的量）
            inside_ratio = (attr_norm * ent_mask_dilate).sum() / (attr_norm.sum() + 1e-4)
            average_loss = 1.0 - inside_ratio
            average_loss_list.append(average_loss)




        if len(shifted_binding_pairs) > 0:
            activation_loss = sum(activation_loss_list)/len(shifted_binding_pairs) if activation_loss_list else 0
            average_loss = sum(average_loss_list)/len(shifted_binding_pairs) if average_loss_list else 0
        else:
            average_loss = 0.
            activation_loss = 0.

        if o_attri_activate:
            attribute_binding_loss = average_loss
        else:
            attribute_binding_loss = activation_loss + average_loss


        #attribute_binding_loss = activation_loss + average_loss
        # print("activation_loss:", activation_loss.item())
        # print("align_loss:", align_loss.item())
        # print("average_loss:", average_loss.item())


        # ----------------------------------
        # 6. EMA Cache 更新 (仅针对实体)
        # ----------------------------------
        ordered_entity_keys = list(entity_cross_attn_maps.keys())
        current_entity_cross_stack = torch.stack([entity_cross_attn_maps[k] for k in ordered_entity_keys])
        current_entity_soft_stack = torch.stack([entity_soft_masks[k] for k in ordered_entity_keys])

        alpha = 0.9
        if self.cross_attention_maps_cache is None or self.cross_attention_maps_cache.shape[0] != len(
                ordered_entity_keys):
            self.cross_attention_maps_cache = current_entity_cross_stack.detach().clone()
            self.self_attention_maps_cache = current_entity_soft_stack.detach().clone()
        else:
            self.cross_attention_maps_cache = self.cross_attention_maps_cache * alpha + current_entity_cross_stack.detach().clone() * (
                        1 - alpha)
            self.self_attention_maps_cache = self.self_attention_maps_cache * alpha + current_entity_soft_stack.detach().clone() * (
                        1 - alpha)

        # -------------------------------------------------
        # 7. 三重防重叠冲突损失 (Entity vs Entity)
        # -------------------------------------------------
        self_cross_attn_loss = 0
        self_cross_attn_loss_1 = 0
        self_cross_attn_loss_2 = 0
        num_pairs = 0
        num_entities = len(ordered_entity_keys)

        for i in range(num_entities):
            curr_cross_i = current_entity_cross_stack[i] / (torch.sum(current_entity_cross_stack[i]) + 1e-4)
            cache_cross_i = self.cross_attention_maps_cache[i] / (torch.sum(self.cross_attention_maps_cache[i]) + 1e-4)
            cache_soft_i = self.self_attention_maps_cache[i] / (torch.sum(self.self_attention_maps_cache[i]) + 1e-4)

            for j in range(num_entities):
                if i == j: continue
                curr_soft_j = current_entity_soft_stack[j] / (torch.sum(current_entity_soft_stack[j]) + 1e-4)

                self_cross_attn_loss += torch.min(curr_soft_j, curr_cross_i).sum()
                self_cross_attn_loss_1 += torch.min(curr_soft_j, cache_cross_i).sum()

                curr_cross_j = current_entity_cross_stack[j] / (torch.sum(current_entity_cross_stack[j]) + 1e-4)
                self_cross_attn_loss_2 += torch.min(cache_soft_i, curr_cross_j).sum()
                num_pairs += 1

        if num_pairs > 0:
            self_cross_attn_loss /= num_pairs
            self_cross_attn_loss_1 /= num_pairs
            self_cross_attn_loss_2 /= num_pairs
        else:
            self_cross_attn_loss = torch.tensor(0.0)
            self_cross_attn_loss_1 = torch.tensor(0.0)
            self_cross_attn_loss_2 = torch.tensor(0.0)

        # -------------
        # 8. 最终损失组合
        # -------------
        total_conflict_loss = self_cross_attn_loss + self_cross_attn_loss_1 + self_cross_attn_loss_2

        if cross_attn_loss > 0.6:
            total_conflict_loss = total_conflict_loss*0.0
            #attribute_binding_loss = attribute_binding_loss*0.0

        effective_cross_attn_loss = cross_attn_loss if cross_attn_loss >= 0.2 else cross_attn_loss * 0.0

        #print("i:",index)
        if index < 4:
            w_attribute = 1.0
        else:
            w_attribute = 1

        if o_attr:
            attribute_binding_loss = attribute_binding_loss*0.0
        if o_self:
            total_conflict_loss = total_conflict_loss*0.0

        joint_loss = (
                effective_cross_attn_loss * 1.0 +
                clean_cross_attention_loss * 0.1 +
                total_conflict_loss +
                attribute_binding_loss * w_attribute
        )

        return joint_loss, cross_attn_loss, total_conflict_loss, attribute_binding_loss

    def fn_compute_loss(
        self,
        indices: List[int],
        smooth_attentions: bool = True,
        K: int = 1,
        attention_res: int = 64,
        from_where: List[int]=None) -> torch.Tensor:

        # -----------------------------
        # cross-attention response loss
        # -----------------------------
        aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
            from_where=from_where, is_cross=True)

        # cross attention map preprocessing
        cross_attention_maps = aggregate_cross_attention_maps[:, :, 1:-1]
        cross_attention_maps = cross_attention_maps * 100
        cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps.float(), dim=-1)

        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]
        number_token = len(indices)
        # clean_cross_attention_loss
        clean_cross_attention_loss = 0.
        cross_attention_map_list = []
        # Extract the maximum values
        topk_average_cross_attn_value_list = []
        otsu_masks = []
        for i in indices:
            cross_attention_map_cur_token = cross_attention_maps[:, :, i]
            if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
            cross_attention_map_list.append(cross_attention_map_cur_token)
            topk_coord_list, topk_value_list = fn_get_topk(cross_attention_map_cur_token, K=K)
            # -----------------------------------
            # clean cross_attention_map_cur_token
            # -----------------------------------
            clean_cross_attention_map_cur_token = cross_attention_map_cur_token
            clean_cross_attention_map_cur_token_mask = fn_get_otsu_mask(clean_cross_attention_map_cur_token)
            otsu_masks.append(clean_cross_attention_map_cur_token_mask)  # don't use over cleaned mask!!
            clean_cross_attention_map_cur_token_mask = fn_clean_mask(clean_cross_attention_map_cur_token_mask,
                                                                     topk_coord_list[0][0], topk_coord_list[0][1])
            clean_cross_attention_map_cur_token_foreground = clean_cross_attention_map_cur_token * clean_cross_attention_map_cur_token_mask + (
                    1 - clean_cross_attention_map_cur_token_mask)
            clean_cross_attention_map_cur_token_background = clean_cross_attention_map_cur_token * (
                    1 - clean_cross_attention_map_cur_token_mask)

            if clean_cross_attention_map_cur_token_background.max() > clean_cross_attention_map_cur_token_foreground.min():
                clean_cross_attention_loss = clean_cross_attention_loss + clean_cross_attention_map_cur_token_background.max()
            else:
                clean_cross_attention_loss = clean_cross_attention_loss + clean_cross_attention_map_cur_token_background.max() * 0

            average_topk_value = torch.mean(torch.stack(topk_value_list))  # average value of top K
            topk_average_cross_attn_value_list.append(average_topk_value)

        cross_attn_loss_list = [max(0 * curr_max, 1.0 - curr_max) for curr_max in topk_average_cross_attn_value_list]
        cross_attn_loss = max(cross_attn_loss_list)

        # ---------------------------------
        # prepare aggregated self attn maps
        # ---------------------------------
        if (len(otsu_masks) != len(cross_attention_map_list)):
            print("numbers of cross attn maps and otsu_masks don't match!")
        self_attention_maps = self.attention_store.aggregate_attention(from_where=from_where, is_cross=False)
        self_attention_map_list = []
        for i in range(len(cross_attention_map_list)):
            cross_attn_map_cur_token = cross_attention_map_list[i]
            self_attn_map_cur_token = torch.zeros_like(cross_attn_map_cur_token)
            mask_cur_token = otsu_masks[i]
            # Vectorized equivalent of the double loop above
            mask_flat2 = mask_cur_token.bool().view(-1)                              # (4096,)
            weights2   = cross_attn_map_cur_token.view(-1)[mask_flat2]               # (N,)
            self_maps2 = self_attention_maps.view(attention_res * attention_res,
                                                  attention_res * attention_res)[mask_flat2]  # (N,4096)
            raw2       = (weights2[:, None] * self_maps2).sum(0)                     # (4096,)
            raw2       = raw2.view(attention_res, attention_res)                     # (64,64)
            if smooth_attentions: raw2 = fn_smoothing_func(raw2)
            self_attention_map_list.append(raw2)

        # ----------------------------------
        # self-cross-attention conflict loss
        # ----------------------------------
        if (len(self_attention_map_list) != len(cross_attention_map_list)):
            print("numbers of cross and self don't match!")
        self_cross_attn_loss = 0
        number_self_cross_loss_pair = 0
        for i in range(number_token):
            cross_attention_map = cross_attention_map_list[i]#.detach().clone()
            cross_attention_map = cross_attention_map / (torch.sum(cross_attention_map)+ 0.0001)
            for j in range(number_token):
                if i == j:
                    continue
                else:
                    self_attention_map = self_attention_map_list[j] / (torch.sum(self_attention_map_list[j])+ 0.0001)
                    self_cross_attn_loss_ij = torch.min(self_attention_map, cross_attention_map).sum()
                    number_self_cross_loss_pair = number_self_cross_loss_pair + 1
                    self_cross_attn_loss = self_cross_attn_loss + self_cross_attn_loss_ij
        if number_self_cross_loss_pair>0: self_cross_attn_loss=self_cross_attn_loss / number_self_cross_loss_pair

        # -------------
        # final losses
        # -------------
        if cross_attn_loss > 0.5:
            self_cross_attn_loss = self_cross_attn_loss * 0
        joint_loss = cross_attn_loss + self_cross_attn_loss + clean_cross_attention_loss * 0.1

        return joint_loss, cross_attn_loss, self_cross_attn_loss

    def fn_augmented_compute_loss(
        self,
        indices: List[int],
        smooth_attentions: bool = True,
        K: int = 1,
        attention_res: int = 64,
        from_where: List[int]=None) -> torch.Tensor:

        # -----------------------------
        # cross-attention response loss
        # -----------------------------
        aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
            from_where=from_where, is_cross=True)

        # cross attention map preprocessing
        cross_attention_maps = aggregate_cross_attention_maps[:, :, 1:-1]
        cross_attention_maps = cross_attention_maps * 100
        cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps.float(), dim=-1)

        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]
        number_token = len(indices)
        # clean_cross_attention_loss
        clean_cross_attention_loss = 0.
        cross_attention_map_list = []
        # Extract the maximum values
        topk_average_cross_attn_value_list = []
        otsu_masks = []
        for i in indices:
            cross_attention_map_cur_token = cross_attention_maps[:, :, i]
            if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
            cross_attention_map_list.append(cross_attention_map_cur_token)
            topk_coord_list, topk_value_list = fn_get_topk(cross_attention_map_cur_token, K=K)
            # -----------------------------------
            # clean cross_attention_map_cur_token
            # -----------------------------------
            clean_cross_attention_map_cur_token = cross_attention_map_cur_token
            clean_cross_attention_map_cur_token_mask = fn_get_otsu_mask(clean_cross_attention_map_cur_token)
            otsu_masks.append(clean_cross_attention_map_cur_token_mask)  # don't use over cleaned mask!!
            clean_cross_attention_map_cur_token_mask = fn_clean_mask(clean_cross_attention_map_cur_token_mask,
                                                                     topk_coord_list[0][0], topk_coord_list[0][1])
            clean_cross_attention_map_cur_token_foreground = clean_cross_attention_map_cur_token * clean_cross_attention_map_cur_token_mask + (
                    1 - clean_cross_attention_map_cur_token_mask)
            clean_cross_attention_map_cur_token_background = clean_cross_attention_map_cur_token * (
                    1 - clean_cross_attention_map_cur_token_mask)

            if clean_cross_attention_map_cur_token_background.max() > clean_cross_attention_map_cur_token_foreground.min():
                clean_cross_attention_loss = clean_cross_attention_loss + clean_cross_attention_map_cur_token_background.max()
            else:
                clean_cross_attention_loss = clean_cross_attention_loss + clean_cross_attention_map_cur_token_background.max() * 0
            # ---------------------------------------------------
            # adaptive top k value cross_attention_map_cur_token
            # ---------------------------------------------------
            average_topk_value = torch.mean(torch.stack(topk_value_list))  # average value of top K
            topk_average_cross_attn_value_list.append(average_topk_value)

        cross_attn_loss_list = [max(0 * curr_max, 1.0 - curr_max) for curr_max in topk_average_cross_attn_value_list]
        cross_attn_loss = max(cross_attn_loss_list)

        # ---------------------------------
        # prepare aggregated self attn maps
        # ---------------------------------
        if (len(otsu_masks) != len(cross_attention_map_list)):
            print("numbers of cross attn maps and otsu_masks don't match!")
        self_attention_maps = self.attention_store.aggregate_attention(from_where=from_where, is_cross=False)
        self_attention_map_list = []
        for i in range(len(cross_attention_map_list)):
            cross_attn_map_cur_token = cross_attention_map_list[i]
            self_attn_map_cur_token = torch.zeros_like(cross_attn_map_cur_token)
            mask_cur_token = otsu_masks[i]
            # Vectorized equivalent of the double loop above
            mask_flat2 = mask_cur_token.bool().view(-1)                              # (4096,)
            weights2   = cross_attn_map_cur_token.view(-1)[mask_flat2]               # (N,)
            self_maps2 = self_attention_maps.view(attention_res * attention_res,
                                                  attention_res * attention_res)[mask_flat2]  # (N,4096)
            raw2       = (weights2[:, None] * self_maps2).sum(0)                     # (4096,)
            raw2       = raw2.view(attention_res, attention_res)                     # (64,64)
            if smooth_attentions: raw2 = fn_smoothing_func(raw2)
            self_attention_map_list.append(raw2)

        # -------------------------------
        # cross attention alignment loss
        # -------------------------------
        if (len(self_attention_map_list) != len(cross_attention_map_list)):
            print("numbers of cross and self don't match!")
        self_attention_maps = torch.stack(self_attention_map_list)
        # print(self_attention_maps.shape)
        cross_attention_maps = torch.stack(cross_attention_map_list)
        # print(cross_attention_maps.shape)
        alpha = 0.9
        if self.cross_attention_maps_cache is None:
            self.cross_attention_maps_cache = cross_attention_maps.detach().clone()
            self.self_attention_maps_cache = self_attention_maps.detach().clone()
        else:
            self.cross_attention_maps_cache = self.cross_attention_maps_cache * alpha + cross_attention_maps.detach().clone() * (1 - alpha)
            self.self_attention_maps_cache = self.self_attention_maps_cache * alpha + self_attention_maps.detach().clone() * (1 - alpha)

        cross_attn_alignment_loss = 0
        for i in range(number_token):
            cross_attention_map_cur_token = cross_attention_maps[i, :, :]
            # if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
            cross_attention_map_cur_token_cache = self.cross_attention_maps_cache[i, :, :]
            # if smooth_attentions: cross_attention_map_cur_token_cache = fn_smoothing_func(cross_attention_map_cur_token_cache)
            cross_attn_alignment_loss = cross_attn_alignment_loss + torch.nn.L1Loss()(cross_attention_map_cur_token,
                                                                                      cross_attention_map_cur_token_cache)

        # -------------------------------------------------
        # self-cross-attention conflict loss with history
        # -------------------------------------------------
        
        self_cross_attn_loss = 0
        number_self_cross_pair = 0
        for i in range(number_token):
            cross_attention_map = cross_attention_maps[i, :, :]  # .detach().clone()
            cross_attention_map = cross_attention_map / (torch.sum(cross_attention_map) + 0.0001)
            for j in range(number_token):
                if i == j:
                    continue
                else:
                    self_attention_map = self_attention_maps[j, :, :] / (torch.sum(self_attention_maps[j, :, :]) + 0.0001)
                    self_cross_attn_loss_ij = torch.min(self_attention_map, cross_attention_map).sum()
                    number_self_cross_pair = number_self_cross_pair + 1
                    self_cross_attn_loss = self_cross_attn_loss + self_cross_attn_loss_ij
        if number_self_cross_pair > 0: self_cross_attn_loss = self_cross_attn_loss / number_self_cross_pair
        
        self_cross_attn_loss_1 = 0
        number_self_cross_pair_1 = 0
        for i in range(number_token):
            cross_attention_map_cache = self.cross_attention_maps_cache[i, :, :]
            cross_attention_map_cache = cross_attention_map_cache / (torch.sum(cross_attention_map_cache) + 0.0001)
            for j in range(number_token):
                if i == j:
                    continue
                else:
                    self_attention_map = self_attention_maps[j, :, :] / (torch.sum(self_attention_maps[j, :, :]) + 0.0001)
                    self_cross_attn_loss_ij_1 = torch.min(self_attention_map, cross_attention_map_cache).sum()
                    number_self_cross_pair_1 = number_self_cross_pair_1 + 1
                    self_cross_attn_loss_1 = self_cross_attn_loss_1 + self_cross_attn_loss_ij_1
        if number_self_cross_pair_1 > 0: self_cross_attn_loss_1 = self_cross_attn_loss_1 / number_self_cross_pair_1

        self_cross_attn_loss_2 = 0
        number_self_cross_pair_2 = 0
        for i in range(number_token):
            self_attention_map_cache = self.self_attention_maps_cache[i, :, :]
            self_attention_map_cache = self_attention_map_cache / (torch.sum(self_attention_map_cache) + 0.0001)
            for j in range(number_token):
                if i == j:
                    continue
                else:
                    cross_attention_map = cross_attention_maps[j, :, :] / (torch.sum(cross_attention_maps[j, :, :]) + 0.0001)
                    self_cross_attn_loss_ij_2 = torch.min(self_attention_map_cache, cross_attention_map).sum()
                    number_self_cross_pair_2 = number_self_cross_pair_2 + 1
                    self_cross_attn_loss_2 = self_cross_attn_loss_2 + self_cross_attn_loss_ij_2
        if number_self_cross_pair_2 > 0: self_cross_attn_loss_2 = self_cross_attn_loss_2 / number_self_cross_pair_2

        # -------------
        # final losses
        # -------------
        self_cross_attn_loss = self_cross_attn_loss + self_cross_attn_loss_1 +self_cross_attn_loss_2
        if cross_attn_loss > 0.5:
            self_cross_attn_loss = self_cross_attn_loss * 0
        joint_loss = cross_attn_loss * 1. + clean_cross_attention_loss * 0.1 + self_cross_attn_loss # + cross_attn_alignment_loss * 0.1 
        # cross_attn_loss = cross_attn_loss + clean_cross_attention_loss * 0.1 + self_cross_attn_loss

        return joint_loss, cross_attn_loss, self_cross_attn_loss

    def fn_calc_kld_loss_func(self, log_var, mu):
        return torch.mean(-0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp()), dim=0)

    def _perform_iterative_refinement_step(
            self,
            latents: torch.Tensor,
            indices: List[List[Union[int,str]]],
            cross_attn_loss: torch.Tensor,
            self_cross_attn_loss: torch.Tensor,
            attribute_loss: torch.Tensor,
            threshold: float,
            text_embeddings: torch.Tensor,
            pooled_text_embeddings: torch.Tensor,
            step_size: float,
            t: int = 1,
            max_refinement_steps: int = 20,
            K: int = 1,
            attention_res: int = 64,
            from_where: List[int] = None,
            index = None
    ):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent code
        according to our loss objective until the given threshold is reached for all tokens.
        """
        iteration = 0
        target_cross_loss = max(0, 1.0 - threshold)
        target_self_loss = 0.3
        target_attribute_loss = 0.5
        while cross_attn_loss > target_cross_loss or self_cross_attn_loss > target_self_loss or attribute_loss > target_attribute_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            timestep = t.expand(latents.shape[0])
            noise_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=text_embeddings,
                pooled_projections=pooled_text_embeddings,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False, )[0]
            self.transformer.zero_grad()
            joint_loss, cross_attn_loss, self_cross_attn_loss, attribute_loss = self.fn_augmented_compute_loss1(indices=indices, K=K,
                                                                                               attention_res=attention_res, from_where=from_where,index=index,
                                                                                               t5_indices=self.t5_token_indices)
            if joint_loss != 0: latents = self._update_latent(latents, joint_loss, step_size)
            print(f"\t Try {iteration}. cross loss: {cross_attn_loss:0.4f}. self loss: {self_cross_attn_loss:0.4f}. attribute loss: {attribute_loss:0.4f}")
            if iteration >= max_refinement_steps:
                print(f"\t Exceeded max number of iterations ({max_refinement_steps})! ")
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        timestep = t.expand(latents.shape[0])
        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=text_embeddings,
            pooled_projections=pooled_text_embeddings,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False, )[0]
        self.transformer.zero_grad()

        joint_loss, cross_attn_loss, self_cross_attn_loss, attribute_loss = self.fn_augmented_compute_loss1(indices=indices, K=K,
                                                                                           attention_res=attention_res, from_where=from_where,index=index,
                                                                                           t5_indices=self.t5_token_indices)
        print(f"\t Finished with loss of: {joint_loss:0.4f}")
        return joint_loss, cross_attn_loss, self_cross_attn_loss, attribute_loss, latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        token_indices: Union[List[int], List[List[Union[int,str]]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        attention_res: int = 64,
        from_where: List[int] = None,
        max_iter_to_alter: int = 14,
        scale_factor: int = 30,
        result_root: str = '',
        seed: int = 0,
        K: int = 1,
        run_sd: bool = False,
        run_initno: bool = False,
        words = None,
        save_attention = False,
        use_t5: bool = False,
        t5_token_indices=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        self.from_where=from_where
        self.use_t5=use_t5
        self.t5_token_indices=t5_token_indices
        self.cross_attention_maps_cache = None
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # print(latents.shape)

        self.attention_store = AttentionStore()
        self.register_attention_control()

        # default config for step size from original repo
        scale_range = np.linspace(1.0, 0.5, len(self.scheduler.timesteps))
        step_size = scale_factor * np.sqrt(scale_range)

        if isinstance(token_indices[0], list):
            token_indices = [token_indices]

        indices = []
        for ind in token_indices:
            indices = indices + [ind] * num_images_per_prompt

        # 8. initno
        if run_initno:
            max_round = 3
            with torch.enable_grad():
                optimized_latents_pool = []
                for round in range(max_round):
                    optimized_latents, optimization_succeed, cross_self_attn_loss = self.fn_initno1(
                        latents=latents,
                        indices=token_indices[0],
                        text_embeddings=prompt_embeds,
                        pooled_text_embeddings=pooled_prompt_embeds,
                        max_step=10,
                        num_inference_steps=num_inference_steps,
                        device=device,
                        guidance_scale=guidance_scale,
                        do_classifier_free_guidance=self.do_classifier_free_guidance,
                        round=round,
                        K=K,
                        attention_res=attention_res,
                        from_where=self.from_where
                    )

                    optimized_latents_pool.append(
                        (cross_self_attn_loss, round, optimized_latents.clone(), latents.clone(), optimization_succeed))
                    if optimization_succeed: break

                    latents = self.prepare_latents(
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height,
                        width,
                        prompt_embeds.dtype,
                        device,
                        generator,
                        latents=None,
                    )

                for score, _round, _optimized_latent, _latent, _optimization_succeed in optimized_latents_pool:
                    print(
                        f'Optimization_succeed: {_optimization_succeed} - Attn score: {score.item():0.4f} - Round: {_round}')
                optimized_latents_pool.sort()
                latents = optimized_latents_pool[0][2] 

        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt:] if self.do_classifier_free_guidance else prompt_embeds
        )
        pooled_text_embeddings = (
            pooled_prompt_embeds[batch_size * num_images_per_prompt:] if self.do_classifier_free_guidance else pooled_prompt_embeds
        )
        # store attention map
        entity_cross_map_numpy_list, self_attention_map_numpy_list, attr_cross_map_numpy_list = [], [], []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Attend and excite process
                with torch.enable_grad():
                    latents = latents.clone().detach().requires_grad_(True)
                    updated_latents = []
                    for latent, index, text_embedding, pooled_text_embedding in zip(latents, indices, text_embeddings, pooled_text_embeddings):
                        # Forward pass of denoising with text conditioning
                        latent = latent.unsqueeze(0)
                        text_embedding = text_embedding.unsqueeze(0)
                        timestep = t.expand(latent.shape[0])

                        noise_pred = self.transformer(
                            hidden_states=latent,
                            timestep=timestep,
                            encoder_hidden_states=text_embedding,
                            pooled_projections=pooled_text_embedding,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False, )[0]
                        self.transformer.zero_grad()

                        joint_loss, cross_attn_loss, self_cross_attn_loss,attribute_loss = self.fn_augmented_compute_loss1(indices=index, K=K, from_where=self.from_where,index=i, t5_indices=self.t5_token_indices)

                        if result_root is not None:
                            # Fetch raw CLIP maps. fn_show_attention_plus_2 does its own
                            # *100 + softmax internally, so we pass raw values here.
                            cross_attention_maps = self.attention_store.aggregate_attention(
                                from_where=self.from_where, is_cross=True)  # (64,64,39) raw
                            self_attention_maps = self.attention_store.aggregate_attention(
                                from_where=self.from_where, is_cross=False)
                            # If use_t5, merge T5 into raw CLIP map in softmax space,
                            # then convert back to raw scale via log so fn_show_attention_plus_2
                            # receives a consistent input (it will re-apply *100 + softmax).
                            if self.use_t5 and self.t5_token_indices is not None:
                                _t5_raw = self.attention_store.aggregate_attention(
                                    from_where=self.from_where, is_cross="t5")  # (64,64,40) raw
                                if _t5_raw is not None:
                                    _clip_sm = torch.nn.functional.softmax((cross_attention_maps * 100).float(), dim=-1)
                                    _t5_sm   = torch.nn.functional.softmax((_t5_raw * 100).float(), dim=-1)
                                    fused_sm = _clip_sm.clone()
                                    for g_i, clip_grp in enumerate(index):
                                        if g_i >= len(self.t5_token_indices): continue
                                        t5_grp = self.t5_token_indices[g_i]
                                        if not isinstance(clip_grp, list): clip_grp = [clip_grp]
                                        if not isinstance(t5_grp, list): t5_grp = [t5_grp]
                                        for pos, c_item in enumerate(clip_grp):
                                            if pos >= len(t5_grp): continue
                                            t5_item = t5_grp[pos]
                                            if isinstance(c_item, str) and ":" in c_item:
                                                cs, ce = map(int, c_item.split(":"))
                                                c_map = _clip_sm[:, :, cs:ce].max(dim=-1)[0]
                                                c_indices = list(range(cs, ce))
                                            else:
                                                c_map = _clip_sm[:, :, int(c_item)]
                                                c_indices = [int(c_item)]
                                            if isinstance(t5_item, str) and ":" in t5_item:
                                                ts, te = map(int, t5_item.split(":"))
                                                t5_map = _t5_sm[:, :, ts:te].max(dim=-1)[0]
                                            else:
                                                t5_map = _t5_sm[:, :, int(t5_item)]
                                            merged = torch.max(c_map, t5_map)
                                            for ci in c_indices:
                                                fused_sm[:, :, ci] = merged
                                    # fn_show_attention_plus_2 does [:,:,0:-1]*100+softmax internally.
                                    # To pass fused_sm (already softmax) through it correctly,
                                    # we recover a raw-equivalent by log(p) / 100, then append a
                                    # dummy last channel so the internal [0:-1] slice removes it.
                                    eps = 1e-8
                                    fused_raw = torch.log(fused_sm.clamp(min=eps)) / 100.0
                                    dummy = torch.zeros(*fused_raw.shape[:2], 1,
                                                        dtype=fused_raw.dtype, device=fused_raw.device)
                                    cross_attention_maps = torch.cat([fused_raw, dummy], dim=-1)
                            cross_attention_maps = cross_attention_maps.cpu()
                            self_attention_maps = self_attention_maps.cpu()
                            
                            entity_cross_maps_numpy, attr_cross_maps_numpy, self_attention_maps_numpy = fn_show_attention_plus_2(
                                cross_attention_maps=cross_attention_maps,
                                self_attention_maps=self_attention_maps,
                                indices=index,
                                K=K,
                                attention_res=attention_res,
                                smooth_attentions=True)  # show average result of top K maps
                            
                            del cross_attention_maps
                            del self_attention_maps
                            # Optional: Clear cache to free memory immediately
                            torch.cuda.empty_cache()

                            entity_cross_map_numpy_list.append(entity_cross_maps_numpy)
                            attr_cross_map_numpy_list.append(attr_cross_maps_numpy)
                            self_attention_map_numpy_list.append(self_attention_maps_numpy)

                         # If this is an iterative refinement step, verify we have reached the desired threshold for all
                        if i < max_iter_to_alter and (i == 4 or i == 10) and (
                                 cross_attn_loss > 0.2 or self_cross_attn_loss > 0.3) and not run_sd:
                             joint_loss, cross_attn_loss, self_cross_attn_loss,attribute_loss, latent = self._perform_iterative_refinement_step(
                                 latents=latent,
                                 indices=index,
                                 cross_attn_loss=cross_attn_loss,
                                #  joint_loss=joint_loss,
                                 self_cross_attn_loss=self_cross_attn_loss,
                                 attribute_loss=attribute_loss,
                                 threshold=0.8,
                                 text_embeddings=text_embedding,
                                 pooled_text_embeddings=pooled_text_embedding,
                                 step_size=step_size[i],
                                 t=t,
                                 K=K,
                                 attention_res=attention_res,
                                 from_where=self.from_where,
                                 index = i
                             )

                        # Perform gradient update
                        if i < max_iter_to_alter and not run_sd:
                            if joint_loss != 0:
                                latent = self._update_latent(
                                    latents=latent,
                                    loss=joint_loss,
                                    step_size=step_size[i],
                                )
                            # logging.info(f"Iteration {i:02d} - cross loss: {cross_attn_loss:0.4f} - self loss: {self_attn_loss:0.4f}")
                        updated_latents.append(latent)
                    latents = torch.cat(updated_latents, dim=0)

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False, )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

            if result_root is not None and save_attention is True:
            # os.makedirs('{:s}/{:s}'.format(result_root,prompt), exist_ok=True)
            # cross_attention_maps_numpy, self_attention_maps_numpy
                entity_indices,attr_indices = [],[]
                for group in words:
                    if not isinstance(group, list):
                        group = [group]
                    ent = group[-1]
                    entity_indices.append(ent)
                    # 列表前面所有的元素作为属性词，批量添加到属性词列表中
                    for attr in group[:-1]:
                        attr_indices.append(attr)
                entity_rows = len(entity_indices)
                attribute_rows = len(attr_indices)
                columns = len(entity_cross_map_numpy_list)
                rows = 2 * entity_rows + attribute_rows # number of lists to be shown
                fig = plt.figure(figsize=(columns * 1.5, rows*1.5))

                for r in range(rows):
                    for c in range(columns):
                        # add_subplot 的 index 是从 1 开始的
                        ax = fig.add_subplot(rows, columns, r * columns + c + 1)

                        # 根据当前行 (r) 判断应该绘制哪一种 attention map
                        if r < entity_rows:
                            img = entity_cross_map_numpy_list[c][r]
                            row_name = f"{entity_indices[r]} Cross"
                        elif r < 2 * entity_rows:
                            k = r - entity_rows
                            img = self_attention_map_numpy_list[c][k]
                            row_name = f"{entity_indices[k]} Self"
                        else:
                            k = r - 2 * entity_rows
                            img = attr_cross_map_numpy_list[c][k]
                            row_name = f"{attr_indices[k]} Cross"

                        ax.imshow(img, cmap='viridis')

                        # 隐藏坐标轴刻度，但保留边框以便后续显示标签
                        ax.set_xticks([])
                        ax.set_yticks([])
                        for spine in ax.spines.values():
                            spine.set_visible(False)

                        # 只在第一行 (r=0) 加上横坐标的时间标签 t_0, t_1...
                        if r == 0:
                            ax.set_title(f"t_{c}", fontsize=14)

                        # 只在第一列 (c=0) 加上纵坐标的 indice 标签
                        if c == 0:
                            ax.set_ylabel(row_name, fontsize=12, rotation=0, labelpad=30, ha='right', va='center')

                    # 使用 tight_layout 自动调整子图参数，防止标签重叠
                plt.tight_layout()
                plt.savefig(f"{result_root}/{seed}.jpg", bbox_inches='tight', pad_inches=0.2)
                plt.close()

        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
