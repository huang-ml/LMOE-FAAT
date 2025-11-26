# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from collections.abc import Callable
from typing import Optional, Union
import copy

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[LlamaConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling


class LoRAExpertMLP(nn.Module):
    def __init__(self, base_mlp, num_experts=8, num_experts_per_token=2, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.base_mlp = base_mlp
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        # Create LoRA experts for each linear layer in the MLP
        self.gate_experts = nn.ModuleList(
            [
                LoRALayer(base_mlp.gate_proj.in_features, base_mlp.gate_proj.out_features, rank, alpha, dropout)
                for _ in range(num_experts)
            ]
        )
        self.up_experts = nn.ModuleList(
            [
                LoRALayer(base_mlp.up_proj.in_features, base_mlp.up_proj.out_features, rank, alpha, dropout)
                for _ in range(num_experts)
            ]
        )
        self.down_experts = nn.ModuleList(
            [
                LoRALayer(base_mlp.down_proj.in_features, base_mlp.down_proj.out_features, rank, alpha, dropout)
                for _ in range(num_experts)
            ]
        )

        # Gating network
        self.gate = nn.Linear(base_mlp.gate_proj.in_features, num_experts, bias=False)

    def _ensure_device_match(self, target_device):
        """Move base MLP and all experts to target device if needed"""
        # Move base MLP projections
        if self.base_mlp.gate_proj.weight.device != target_device:
            self.base_mlp.gate_proj = self.base_mlp.gate_proj.to(target_device)
        if self.base_mlp.up_proj.weight.device != target_device:
            self.base_mlp.up_proj = self.base_mlp.up_proj.to(target_device)
        if self.base_mlp.down_proj.weight.device != target_device:
            self.base_mlp.down_proj = self.base_mlp.down_proj.to(target_device)

        # Move all experts
        for expert in self.gate_experts:
            if expert.lora_A.weight.device != target_device:
                expert.to(target_device)
        for expert in self.up_experts:
            if expert.lora_A.weight.device != target_device:
                expert.to(target_device)
        for expert in self.down_experts:
            if expert.lora_A.weight.device != target_device:
                expert.to(target_device)

    def forward(self, x, return_router_logits=False):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)

        # Ensure base MLP and all experts are on the same device as input
        self._ensure_device_match(x_flat.device)

        gate_base = self.base_mlp.gate_proj(x_flat)
        up_base = self.base_mlp.up_proj(x_flat)
        act_base = self.base_mlp.act_fn(gate_base)
        hidden_base = act_base * up_base
        base_output_flat = self.base_mlp.down_proj(hidden_base)
        base_output = base_output_flat.view(batch_size, seq_len, hidden_dim)
        del hidden_base

        # Ensure gate is on the same device as input
        gate_weight = self.gate.weight.to(x_flat.device)
        router_logits = F.linear(x_flat, gate_weight)
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float32),
            self.num_experts_per_token,
            dim=-1,
            sorted=False,
        )
        routing_weights = routing_weights.to(x.dtype)
        selected_experts = selected_experts.int()

        expert_outputs_flat = torch.zeros_like(base_output_flat)
        flat_selected = selected_experts.view(-1)
        flat_weights = routing_weights.view(-1)
        token_ids = torch.arange(x_flat.size(0), device=x_flat.device).repeat_interleave(self.num_experts_per_token)

        for expert_id, (gate_expert, up_expert, down_expert) in enumerate(
            zip(self.gate_experts, self.up_experts, self.down_experts)
        ):
            expert_mask = flat_selected == expert_id
            if not torch.any(expert_mask):
                continue

            expert_token_ids = token_ids[expert_mask]
            expert_weights = flat_weights[expert_mask].unsqueeze(-1).to(x.dtype)

            expert_input = x_flat.index_select(0, expert_token_ids)
            gate_total = gate_base.index_select(0, expert_token_ids) + gate_expert(expert_input)
            up_total = up_base.index_select(0, expert_token_ids) + up_expert(expert_input)
            hidden_total = self.base_mlp.act_fn(gate_total) * up_total

            down_delta = down_expert(hidden_total)
            expert_outputs_flat.index_add_(0, expert_token_ids, down_delta * expert_weights)

        expert_outputs = expert_outputs_flat.view(batch_size, seq_len, hidden_dim)
        final_output = base_output + expert_outputs

        if return_router_logits:
            return final_output, router_logits.view(batch_size, seq_len, -1)
        return final_output


class LoRAExpertAttention(nn.Module):
    def __init__(self, base_attention, num_experts=4, num_experts_per_token=2, rank=16, alpha=32, dropout=0.1):
        super().__init__()
        self.base_attention = base_attention
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        # Copy necessary attributes from base attention
        self.config = base_attention.config
        self.layer_idx = base_attention.layer_idx
        self.head_dim = base_attention.head_dim
        self.num_key_value_groups = base_attention.num_key_value_groups
        self.scaling = base_attention.scaling
        self.attention_dropout = base_attention.attention_dropout
        self.is_causal = base_attention.is_causal

        # Create LoRA experts - only for output projection to keep it simple and efficient
        self.o_experts = nn.ModuleList(
            [
                LoRALayer(base_attention.o_proj.in_features, base_attention.o_proj.out_features, rank, alpha, dropout)
                for _ in range(num_experts)
            ]
        )

        # Gating network
        self.gate = nn.Linear(base_attention.q_proj.in_features, num_experts, bias=False)

    def _ensure_device_match(self, hidden_states):
        """Ensure all base attention components are on the same device as hidden_states"""
        target_device = hidden_states.device

        # Move all base attention projections to target device if needed
        if self.base_attention.q_proj.weight.device != target_device:
            self.base_attention.q_proj = self.base_attention.q_proj.to(target_device)
        if self.base_attention.k_proj.weight.device != target_device:
            self.base_attention.k_proj = self.base_attention.k_proj.to(target_device)
        if self.base_attention.v_proj.weight.device != target_device:
            self.base_attention.v_proj = self.base_attention.v_proj.to(target_device)
        if self.base_attention.o_proj.weight.device != target_device:
            self.base_attention.o_proj = self.base_attention.o_proj.to(target_device)

        # Move all LoRA experts to target device if needed
        for expert in self.o_experts:
            if expert.lora_A.weight.device != target_device:
                expert.to(target_device)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        return_router_logits=False,
        **kwargs,
    ):
        # Ensure all base attention components and experts are on the same device
        self._ensure_device_match(hidden_states)

        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)

        # Ensure gate is on the same device as input
        gate_weight = self.gate.weight.to(hidden_states_flat.device)
        router_logits = F.linear(hidden_states_flat, gate_weight)
        routing_weights, selected_experts = torch.topk(
            F.softmax(router_logits, dim=-1, dtype=torch.float32),
            self.num_experts_per_token,
            dim=-1,
            sorted=False,
        )
        routing_weights = routing_weights.to(hidden_states.dtype)
        selected_experts = selected_experts.int()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.base_attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.base_attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.base_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self.base_attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        base_output = self.base_attention.o_proj(attn_output)

        attn_output_flat = attn_output.view(-1, attn_output.size(-1))
        base_output_flat = base_output.view(-1, hidden_dim)
        expert_outputs_flat = torch.zeros_like(base_output_flat)

        flat_selected = selected_experts.view(-1)
        flat_weights = routing_weights.view(-1)
        token_ids = torch.arange(attn_output_flat.size(0), device=attn_output_flat.device).repeat_interleave(
            self.num_experts_per_token
        )

        for expert_id, o_expert in enumerate(self.o_experts):
            expert_mask = flat_selected == expert_id
            if not torch.any(expert_mask):
                continue

            expert_token_ids = token_ids[expert_mask]
            expert_weights = flat_weights[expert_mask].unsqueeze(-1).to(attn_output_flat.dtype)
            expert_input = attn_output_flat.index_select(0, expert_token_ids)
            o_delta = o_expert(expert_input)

            expert_outputs_flat.index_add_(0, expert_token_ids, o_delta * expert_weights)

        expert_outputs = expert_outputs_flat.view(batch_size, seq_len, hidden_dim)
        final_output = base_output + expert_outputs

        if return_router_logits:
            return final_output, router_logits.view(batch_size, seq_len, -1)
        return final_output, attn_weights


def LlamaMoeForCausalLMConvert(
    model_name_or_path: str,
    use_attention_experts: bool = False,
    use_ffn_experts: bool = True,
    num_attention_experts: int = 4,
    num_attention_experts_per_token: int = 2,
    num_ffn_experts: int = 8,
    num_ffn_experts_per_token: int = 2,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    use_aux_heads: bool = False,
    num_aux_heads: int = 4,
    aux_loss_alpha: float = 0.8,
    **kwargs
):
    """
    Convert a base Llama model to a MoE model using LoRA experts.

    Args:
        model_name_or_path: Path or name of the base model
        use_attention_experts: Whether to add experts to attention layers
        use_ffn_experts: Whether to add experts to FFN layers
        num_attention_experts: Number of attention experts
        num_attention_experts_per_token: Number of attention experts per token
        num_ffn_experts: Number of FFN experts
        num_ffn_experts_per_token: Number of FFN experts per token
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout rate
        use_aux_heads: Whether to use auxiliary heads for training
        num_aux_heads: Number of auxiliary heads
        aux_loss_alpha: Alpha parameter for auxiliary loss weighting

    Returns:
        Converted MoE model
    """
    # Load base model
    base_model = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)

    # Convert layers to MoE
    for layer_idx, layer in enumerate(base_model.model.layers):
        # Convert attention if requested
        if use_attention_experts:
            layer.self_attn = LoRAExpertAttention(
                layer.self_attn,
                num_experts=num_attention_experts,
                num_experts_per_token=num_attention_experts_per_token,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

        # Convert MLP if requested
        if use_ffn_experts:
            layer.mlp = LoRAExpertMLP(
                layer.mlp,
                num_experts=num_ffn_experts,
                num_experts_per_token=num_ffn_experts_per_token,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

    # Add auxiliary heads if requested
    if use_aux_heads:
        base_model.use_aux_heads = True
        base_model.num_aux_heads = num_aux_heads
        base_model.aux_loss_alpha = aux_loss_alpha

        # Initialize auxiliary LM heads
        base_model.aux_lm_heads = nn.ModuleList()
        for _ in range(num_aux_heads):
            base_model.aux_lm_heads.append(copy.deepcopy(base_model.lm_head))

        # Initialize future fusion projection layers
        base_model.future_fusion_projection = nn.ModuleList()
        for _ in range(num_aux_heads):
            proj = nn.Linear(base_model.config.vocab_size, base_model.config.hidden_size, bias=False)
            nn.init.xavier_uniform_(proj.weight, gain=0.01)
            base_model.future_fusion_projection.append(proj)

        # Initialize fusion gate
        base_model.fusion_gate = nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size, bias=True)
        nn.init.zeros_(base_model.fusion_gate.weight)
        nn.init.zeros_(base_model.fusion_gate.bias)

    # Add conversion metadata
    base_model._conversion_config = {
        "use_attention_experts": use_attention_experts,
        "use_ffn_experts": use_ffn_experts,
        "num_attention_experts": num_attention_experts,
        "num_attention_experts_per_token": num_attention_experts_per_token,
        "num_ffn_experts": num_ffn_experts,
        "num_ffn_experts_per_token": num_ffn_experts_per_token,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "use_aux_heads": use_aux_heads,
        "num_aux_heads": num_aux_heads,
        "aux_loss_alpha": aux_loss_alpha,
        "base_model_name": model_name_or_path,
    }

    return base_model


def LlamaMoeForCausalLMLoad(model_path: str, device_map=None, **kwargs):
    """
    Load a previously converted MoE model.

    Args:
        model_path: Path to the converted model
        device_map: Device map for model placement. Options:
            - None: Load on default device
            - "auto": Automatically distribute across available GPUs
            - dict: Custom device placement mapping
            - str: Single device (e.g., "cuda:0", "cpu")
        **kwargs: Additional arguments for model loading

    Returns:
        Loaded MoE model
    """
    import json
    import os
    from safetensors.torch import load_file

    # Load the conversion config
    config_path = os.path.join(model_path, "conversion_config.json")
    with open(config_path, "r") as f:
        conversion_config = json.load(f)

    # Load the base model first
    base_model_name = conversion_config.get("base_model_name")
    if not base_model_name:
        raise ValueError("Base model name not found in conversion config")

    print(f"Loading base model: {base_model_name}")

    # Remove device_map from kwargs if present to avoid conflicts
    kwargs_for_base = kwargs.copy()
    if "device_map" in kwargs_for_base:
        del kwargs_for_base["device_map"]

    # Determine target device(s) and create custom device map for even distribution
    custom_device_map = None
    if device_map == "auto":
        print("Using automatic device mapping across available GPUs")
        # Check available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Found {num_gpus} GPU(s)")
            if num_gpus > 0:
                device_map_config = "auto"
                # We'll create a custom device map after loading the base model
            else:
                device_map_config = "cpu"
                print("No GPUs available, falling back to CPU")
        else:
            device_map_config = "cpu"
            print("CUDA not available, using CPU")
    elif device_map is None:
        device_map_config = None
    else:
        device_map_config = device_map

    # Load base model with device_map
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map_config,
        **kwargs_for_base,
    )

    # Get the device of the first parameter for reference
    first_param_device = next(base_model.parameters()).device
    print(f"Base model loaded on device: {first_param_device}")

    # Reconstruct the MoE architecture
    print("Reconstructing MoE architecture...")
    use_attention_experts = conversion_config.get("use_attention_experts", False)
    use_ffn_experts = conversion_config.get("use_ffn_experts", True)
    num_attention_experts = conversion_config.get("num_attention_experts", 4)
    num_attention_experts_per_token = conversion_config.get("num_attention_experts_per_token", 2)
    num_ffn_experts = conversion_config.get("num_ffn_experts", 8)
    num_ffn_experts_per_token = conversion_config.get("num_ffn_experts_per_token", 2)
    rank = conversion_config.get("rank", 16)
    alpha = conversion_config.get("alpha", 32)
    dropout = conversion_config.get("dropout", 0.1)

    # Auxiliary heads config
    use_aux_heads = conversion_config.get("use_aux_heads", False)
    num_aux_heads = conversion_config.get("num_aux_heads", 4)
    aux_loss_alpha = conversion_config.get("aux_loss_alpha", 0.8)

    # Create expert placement strategy for even GPU distribution
    num_layers = len(base_model.model.layers)
    if device_map_config == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"\nCreating balanced device map across {num_gpus} GPUs...")

        # Calculate layers per GPU for even distribution
        layers_per_gpu = num_layers // num_gpus
        remainder = num_layers % num_gpus

        # Build layer-to-device mapping
        layer_device_map = {}
        current_layer = 0
        for gpu_id in range(num_gpus):
            # Distribute remainder layers among first GPUs
            num_layers_for_gpu = layers_per_gpu + (1 if gpu_id < remainder else 0)
            for _ in range(num_layers_for_gpu):
                if current_layer < num_layers:
                    layer_device_map[current_layer] = f"cuda:{gpu_id}"
                    current_layer += 1

        print(f"Layer distribution: {dict(sorted(layer_device_map.items()))}")
    else:
        layer_device_map = None

    # Convert layers to MoE architecture with balanced device placement
    for layer_idx, layer in enumerate(base_model.model.layers):
        # Determine target device for this layer
        if layer_device_map and layer_idx in layer_device_map:
            target_device = torch.device(layer_device_map[layer_idx])
            # Move entire layer to target device first
            layer = layer.to(target_device)
            base_model.model.layers[layer_idx] = layer
            layer_device = target_device
        else:
            # Get the device of the current layer
            layer_device = next(layer.parameters()).device

        # Convert attention if requested
        if use_attention_experts:
            base_attn = layer.self_attn
            layer.self_attn = LoRAExpertAttention(
                base_attn,
                num_experts=num_attention_experts,
                num_experts_per_token=num_attention_experts_per_token,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            # Move the new expert module to the same device as the layer
            layer.self_attn = layer.self_attn.to(layer_device)

        # Convert MLP if requested
        if use_ffn_experts:
            base_mlp = layer.mlp
            layer.mlp = LoRAExpertMLP(
                base_mlp,
                num_experts=num_ffn_experts,
                num_experts_per_token=num_ffn_experts_per_token,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            # Move the new expert module to the same device as the layer
            layer.mlp = layer.mlp.to(layer_device)

    # Distribute auxiliary heads across GPUs if using multi-GPU
    if use_aux_heads:
        print("Reconstructing auxiliary heads...")
        base_model.use_aux_heads = True
        base_model.num_aux_heads = num_aux_heads
        base_model.aux_loss_alpha = aux_loss_alpha

        # Determine device strategy for auxiliary heads
        if layer_device_map and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            num_gpus = torch.cuda.device_count()
            # Distribute auxiliary heads across GPUs
            print(f"Distributing {num_aux_heads} auxiliary heads across {num_gpus} GPUs...")

            # Initialize auxiliary LM heads
            base_model.aux_lm_heads = nn.ModuleList()
            for i in range(num_aux_heads):
                aux_head = copy.deepcopy(base_model.lm_head)
                # Distribute auxiliary heads across GPUs
                aux_head_device = f"cuda:{i % num_gpus}"
                base_model.aux_lm_heads.append(aux_head.to(aux_head_device))
                print(f"  Auxiliary head {i} -> {aux_head_device}")

            # Initialize future fusion projection layers
            base_model.future_fusion_projection = nn.ModuleList()
            for i in range(num_aux_heads):
                proj = nn.Linear(base_model.config.vocab_size, base_model.config.hidden_size, bias=False)
                nn.init.xavier_uniform_(proj.weight, gain=0.01)
                # Match projection layer device with corresponding auxiliary head
                proj_device = f"cuda:{i % num_gpus}"
                base_model.future_fusion_projection.append(proj.to(proj_device))

            # Place fusion gate on the device where most layers are
            # Typically the first GPU or where lm_head is
            lm_head_device = base_model.lm_head.weight.device
            base_model.fusion_gate = nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size, bias=True)
            nn.init.zeros_(base_model.fusion_gate.weight)
            nn.init.zeros_(base_model.fusion_gate.bias)
            base_model.fusion_gate = base_model.fusion_gate.to(lm_head_device)
            print(f"  Fusion gate -> {lm_head_device}")
        else:
            # Single GPU or CPU - keep all on same device
            lm_head_device = base_model.lm_head.weight.device

            # Initialize auxiliary LM heads
            base_model.aux_lm_heads = nn.ModuleList()
            for _ in range(num_aux_heads):
                aux_head = copy.deepcopy(base_model.lm_head)
                base_model.aux_lm_heads.append(aux_head.to(lm_head_device))

            # Initialize future fusion projection layers
            base_model.future_fusion_projection = nn.ModuleList()
            for _ in range(num_aux_heads):
                proj = nn.Linear(base_model.config.vocab_size, base_model.config.hidden_size, bias=False)
                nn.init.xavier_uniform_(proj.weight, gain=0.01)
                base_model.future_fusion_projection.append(proj.to(lm_head_device))

            # Initialize fusion gate
            base_model.fusion_gate = nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size, bias=True)
            nn.init.zeros_(base_model.fusion_gate.weight)
            nn.init.zeros_(base_model.fusion_gate.bias)
            base_model.fusion_gate = base_model.fusion_gate.to(lm_head_device)

    # Load the saved state dict
    state_dict_path = os.path.join(model_path, "pytorch_model.bin")
    safetensors_path = os.path.join(model_path, "model.safetensors")

    print("\nLoading model weights...")
    if os.path.exists(safetensors_path):
        print("Loading from safetensors...")
        state_dict = load_file(safetensors_path)
    elif os.path.exists(state_dict_path):
        print("Loading from pytorch_model.bin...")
        state_dict = torch.load(state_dict_path, map_location="cpu")
    else:
        # Try to load sharded model
        index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
        safetensors_index_path = os.path.join(model_path, "model.safetensors.index.json")

        if os.path.exists(safetensors_index_path):
            print("Loading sharded safetensors model...")
            with open(safetensors_index_path, "r") as f:
                index = json.load(f)
            state_dict = {}
            for filename in set(index["weight_map"].values()):
                shard_path = os.path.join(model_path, filename)
                shard_state_dict = load_file(shard_path)
                state_dict.update(shard_state_dict)
        elif os.path.exists(index_path):
            print("Loading sharded pytorch model...")
            with open(index_path, "r") as f:
                index = json.load(f)
            state_dict = {}
            for filename in set(index["weight_map"].values()):
                shard_path = os.path.join(model_path, filename)
                shard_state_dict = torch.load(shard_path, map_location="cpu")
                state_dict.update(shard_state_dict)
        else:
            raise ValueError(f"No model files found in {model_path}")

    # Load the state dict into the model with proper device handling
    # When using device_map, load_state_dict should respect the existing device placement
    missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys in state dict: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state dict: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")

    # Restore conversion config
    base_model._conversion_config = conversion_config

    # Verify device placement and show distribution statistics
    if device_map_config == "auto" and torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Device Placement Verification:")
        print("=" * 60)

        device_param_count = {}
        device_memory_usage = {}

        for name, param in base_model.named_parameters():
            device = str(param.device)
            if device not in device_param_count:
                device_param_count[device] = 0
                device_memory_usage[device] = 0
            device_param_count[device] += 1
            device_memory_usage[device] += param.numel() * param.element_size()

        print(f"\nParameter Distribution:")
        for device in sorted(device_param_count.keys()):
            param_count = device_param_count[device]
            memory_mb = device_memory_usage[device] / (1024 * 1024)
            print(f"  {device}: {param_count} parameters ({memory_mb:.2f} MB)")

        # Show layer distribution
        if layer_device_map:
            print(f"\nLayer-to-GPU Mapping:")
            gpu_layer_count = {}
            for layer_idx, device in layer_device_map.items():
                if device not in gpu_layer_count:
                    gpu_layer_count[device] = []
                gpu_layer_count[device].append(layer_idx)

            for device in sorted(gpu_layer_count.keys()):
                layers = gpu_layer_count[device]
                print(f"  {device}: Layers {min(layers)}-{max(layers)} ({len(layers)} layers)")

        print("=" * 60)

    # Enable gradient checkpointing if using multiple GPUs for memory efficiency
    if device_map_config == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if hasattr(base_model, "gradient_checkpointing_enable"):
            print("\nGradient checkpointing available for memory efficiency")
            print("You can enable it with: model.gradient_checkpointing_enable()")

    print("\nModel loaded successfully!")
    return base_model


def save_moe_model(model, save_path: str):
    """
    Save a converted MoE model with its conversion configuration.

    Args:
        model: The converted MoE model
        save_path: Path to save the model
    """
    import json
    import os

    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the model using the standard transformers method
    model.save_pretrained(save_path)

    # Save the conversion configuration
    if hasattr(model, "_conversion_config"):
        config_path = os.path.join(save_path, "conversion_config.json")
        with open(config_path, "w") as f:
            json.dump(model._conversion_config, f, indent=2)
        print(f"Conversion config saved to {config_path}")
    else:
        print("Warning: Model doesn't have _conversion_config attribute")
