from typing import Callable, Optional, Union
import copy

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import OutputRecorder, check_model_inputs
from configuration_qwen3 import Qwen3Config

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Ensure weight is on the same device as hidden_states
        weight = self.weight.to(hidden_states.device)
        return weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


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


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
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
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Ensure all projections are on the same device as hidden_states
        target_device = hidden_states.device
        
        if self.q_proj.weight.device != target_device:
            self.q_proj = self.q_proj.to(target_device)
        if self.k_proj.weight.device != target_device:
            self.k_proj = self.k_proj.to(target_device)
        if self.v_proj.weight.device != target_device:
            self.v_proj = self.v_proj.to(target_device)
        if self.o_proj.weight.device != target_device:
            self.o_proj = self.o_proj.to(target_device)
        if self.q_norm.weight.device != target_device:
            self.q_norm = self.q_norm.to(target_device)
        if self.k_norm.weight.device != target_device:
            self.k_norm = self.k_norm.to(target_device)
        
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
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
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_router_logits: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
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
        
        router_logits = None
        # Check if MLP is a MoE layer that returns router logits
        if isinstance(self.mlp, (LoRAExpertMLP, LoRAExpertAttention)) and output_router_logits:
            hidden_states, router_logits = self.mlp(hidden_states, return_router_logits=True)
        else:
            hidden_states = self.mlp(hidden_states)
            
        hidden_states = residual + hidden_states
        
        if output_router_logits:
            return hidden_states, router_logits
        return hidden_states


@auto_docstring
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

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


@auto_docstring
class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[BaseModelOutputWithPast, tuple]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Collect router logits if requested
        all_router_logits = [] if output_router_logits else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_output = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_router_logits=output_router_logits,
                **kwargs,
            )
            
            if output_router_logits:
                hidden_states, router_logits = layer_output
                if router_logits is not None:
                    all_router_logits.append(router_logits)
            else:
                hidden_states = layer_output

        hidden_states = self.norm(hidden_states)
        
        # Return router logits as tuple if requested
        if output_router_logits and all_router_logits:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
            ), tuple(all_router_logits)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # MoE-specific attributes
        self.router_aux_loss_coef = getattr(config, 'router_aux_loss_coef', 0.001)
        
        # Auxiliary heads attributes
        self.use_aux_heads = getattr(config, "use_aux_heads", False)
        if self.use_aux_heads:
            self.num_aux_heads = getattr(config, "num_aux_heads", 4)
            self.aux_loss_alpha = getattr(config, "aux_loss_alpha", 0.8)
            
            # Initialize auxiliary LM heads
            self.aux_lm_heads = nn.ModuleList()
            for _ in range(self.num_aux_heads):
                self.aux_lm_heads.append(copy.deepcopy(self.lm_head))
            
            # Initialize future fusion projection layers
            self.future_fusion_projection = nn.ModuleList()
            for _ in range(self.num_aux_heads):
                proj = nn.Linear(config.vocab_size, config.hidden_size, bias=False)
                nn.init.xavier_uniform_(proj.weight, gain=0.01)
                self.future_fusion_projection.append(proj)
            
            # Initialize fusion gate
            self.fusion_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
            nn.init.zeros_(self.fusion_gate.weight)  # Start with gate closed
            nn.init.zeros_(self.fusion_gate.bias)

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
        output_router_logits: Optional[bool] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[CausalLMOutputWithPast, tuple]:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else getattr(self.config, 'output_router_logits', False)
        )
        
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        
        # Extract outputs based on whether router logits were requested
        if output_router_logits and isinstance(model_output, tuple):
            outputs, router_logits = model_output
        else:
            outputs = model_output
            router_logits = None

        hidden_states = outputs.last_hidden_state
        
        loss = None
        aux_loss = None
        total_loss = None
        
        # Handle auxiliary heads if enabled and labels are provided
        if labels is not None and self.use_aux_heads and self.training:
            import math
            loss_fct = nn.CrossEntropyLoss(reduction='sum')
            
            # Get the device where we want to accumulate losses (same as hidden_states)
            target_device = hidden_states.device
            total_aux_loss = torch.tensor(0.0, device=target_device, dtype=hidden_states.dtype)
            fused_future_info = torch.zeros_like(hidden_states)
            
            # Compute auxiliary losses and gather future information
            for i, (aux_head, proj_layer) in enumerate(zip(self.aux_lm_heads, self.future_fusion_projection)):
                prediction_shift = i + 2
                if hidden_states.shape[1] <= prediction_shift:
                    continue
                
                # Get devices for aux head and projection layer
                aux_head_device = next(aux_head.parameters()).device
                proj_device = next(proj_layer.parameters()).device
                
                # Slice the tensors first
                aux_hidden_states_slice = hidden_states[..., :-prediction_shift, :].contiguous()
                aux_labels_slice = labels[..., prediction_shift:].contiguous()
                
                # Create a fresh copy on the auxiliary head device
                aux_hidden_states_input = aux_hidden_states_slice.detach().clone().to(aux_head_device)
                if self.training:
                    aux_hidden_states_input.requires_grad_(True)
                aux_labels = aux_labels_slice.clone().to(aux_head_device)
                
                # Clean up slices
                del aux_hidden_states_slice, aux_labels_slice
                
                # Compute auxiliary logits - bypass Accelerate hooks if present
                # Check if aux_head has _old_forward (Accelerate hook)
                if hasattr(aux_head, '_old_forward'):
                    # Use the original forward method to bypass hooks
                    aux_logits = aux_head._old_forward(aux_hidden_states_input)
                else:
                    aux_logits = aux_head(aux_hidden_states_input)
                
                aux_loss_i = loss_fct(aux_logits.view(-1, self.config.vocab_size), aux_labels.view(-1))
                
                # Apply exponential weighting and move to target device
                weight = self.aux_loss_alpha ** (i + 1)
                total_aux_loss = total_aux_loss + (weight * aux_loss_i.to(target_device))
                
                # Project and accumulate future information
                aux_logits_for_proj = aux_logits.detach().clone().to(proj_device)
                
                # Bypass hooks for projection layer too if present
                if hasattr(proj_layer, '_old_forward'):
                    projected_logits = proj_layer._old_forward(F.softmax(aux_logits_for_proj, dim=-1))
                else:
                    projected_logits = proj_layer(F.softmax(aux_logits_for_proj, dim=-1))
                
                # Move projected logits to target device and accumulate
                projected_logits = projected_logits.to(target_device)
                fused_future_info[..., :-prediction_shift, :] = fused_future_info[..., :-prediction_shift, :] + projected_logits
                
                # Clean up intermediate tensors to save memory
                del aux_hidden_states_input, aux_labels, aux_logits, aux_logits_for_proj, projected_logits
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Get fusion gate device and move hidden states if necessary
            fusion_gate_device = next(self.fusion_gate.parameters()).device
            hidden_states_for_gate = hidden_states.to(fusion_gate_device)
            fused_future_info_for_gate = fused_future_info.to(fusion_gate_device)
            
            # Apply fusion gate - bypass hooks if present
            if hasattr(self.fusion_gate, '_old_forward'):
                gate_values = torch.sigmoid(self.fusion_gate._old_forward(hidden_states_for_gate))
            else:
                gate_values = torch.sigmoid(self.fusion_gate(hidden_states_for_gate))
            
            augmented_hidden_states = hidden_states_for_gate + gate_values * fused_future_info_for_gate
            
            # Get lm_head device
            lm_head_device = self.lm_head.weight.device
            augmented_hidden_states = augmented_hidden_states.to(lm_head_device)
            
            # Compute main loss with augmented hidden states
            main_logits = self.lm_head(augmented_hidden_states)
            
            # Move labels to lm_head device for loss computation
            labels_for_loss = labels.to(lm_head_device)
            shift_logits = main_logits[..., :-1, :].contiguous()
            shift_labels = labels_for_loss[..., 1:].contiguous()
            main_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Compute beta decay factor for auxiliary loss
            beta = 1.0
            if total_steps > 0:
                decay_ratio = min(1.0, current_step / total_steps)
                beta = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            
            # Get num_items_in_batch and ensure it's a scalar or on correct device
            num_items_in_batch = kwargs.get('num_items_in_batch')
            if num_items_in_batch is None:
                num_items_in_batch = 1
            elif isinstance(num_items_in_batch, torch.Tensor):
                num_items_in_batch = num_items_in_batch.to(target_device)
            
            # Combine main and auxiliary losses (ensure they're on the same device)
            main_loss = main_loss.to(target_device)
            total_aux_loss = total_aux_loss.to(target_device)
            total_loss = (main_loss + (0.05 * beta * total_aux_loss)) / num_items_in_batch
            final_logits = main_logits
        else:
            # Standard forward without auxiliary heads
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            final_logits = logits
            
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
                total_loss = loss
        
        # Compute router auxiliary loss if available
        if output_router_logits and router_logits is not None:
            num_experts = router_logits[0].shape[-1] if len(router_logits) > 0 else 0
            
            top_k = 2  # Default
            for layer in self.model.layers:
                if isinstance(layer.mlp, LoRAExpertMLP):
                    top_k = layer.mlp.num_experts_per_token
                    break
                elif isinstance(layer.self_attn, LoRAExpertAttention):
                    top_k = layer.self_attn.num_experts_per_token
                    break
            
            aux_loss = load_balancing_loss_func(
                router_logits,
                num_experts,
                top_k,
                attention_mask,
            )
            
            if total_loss is not None and aux_loss != 0:
                total_loss = total_loss + self.router_aux_loss_coef * aux_loss.to(total_loss.device)

        # Create output
        output_dict = {
            'loss': total_loss,
            'logits': final_logits,
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
        
        if output_router_logits:
            output_dict['aux_loss'] = aux_loss
            output_dict['router_logits'] = router_logits
        
        return CausalLMOutputWithPast(**output_dict)


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

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
        self.gate_experts = nn.ModuleList([
            LoRALayer(base_mlp.gate_proj.in_features, base_mlp.gate_proj.out_features, rank, alpha, dropout)
            for _ in range(num_experts)
        ])
        self.up_experts = nn.ModuleList([
            LoRALayer(base_mlp.up_proj.in_features, base_mlp.up_proj.out_features, rank, alpha, dropout)
            for _ in range(num_experts)
        ])
        self.down_experts = nn.ModuleList([
            LoRALayer(base_mlp.down_proj.in_features, base_mlp.down_proj.out_features, rank, alpha, dropout)
            for _ in range(num_experts)
        ])
        
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
        self.sliding_window = base_attention.sliding_window
        
        # Create LoRA experts - only for output projection to keep it simple and efficient
        self.o_experts = nn.ModuleList([
            LoRALayer(base_attention.o_proj.in_features, base_attention.o_proj.out_features, rank, alpha, dropout)
            for _ in range(num_experts)
        ])
        
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
        
        # Move normalization layers
        if self.base_attention.q_norm.weight.device != target_device:
            self.base_attention.q_norm = self.base_attention.q_norm.to(target_device)
        if self.base_attention.k_norm.weight.device != target_device:
            self.base_attention.k_norm = self.base_attention.k_norm.to(target_device)
        
        # Move all LoRA experts to target device if needed
        for expert in self.o_experts:
            if expert.lora_A.weight.device != target_device:
                expert.to(target_device)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, cache_position=None, return_router_logits=False, **kwargs):
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

        query_states = self.base_attention.q_norm(
            self.base_attention.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.base_attention.k_norm(
            self.base_attention.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
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
            sliding_window=self.sliding_window,
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


def Qwen3MoeForCausalLMConvert(
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
    Convert a base Qwen model to a MoE model using LoRA experts.
    
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
    base_model = Qwen3ForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    
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
                dropout=dropout
            )
        
        # Convert MLP if requested
        if use_ffn_experts:
            layer.mlp = LoRAExpertMLP(
                layer.mlp,
                num_experts=num_ffn_experts,
                num_experts_per_token=num_ffn_experts_per_token,
                rank=rank,
                alpha=alpha,
                dropout=dropout
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
        'use_attention_experts': use_attention_experts,
        'use_ffn_experts': use_ffn_experts,
        'num_attention_experts': num_attention_experts,
        'num_attention_experts_per_token': num_attention_experts_per_token,
        'num_ffn_experts': num_ffn_experts,
        'num_ffn_experts_per_token': num_ffn_experts_per_token,
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'use_aux_heads': use_aux_heads,
        'num_aux_heads': num_aux_heads,
        'aux_loss_alpha': aux_loss_alpha,
        'base_model_name': model_name_or_path
    }
    
    return base_model


def Qwen3MoeForCausalLMLoad(model_path: str, device_map=None, **kwargs):
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
    import os
    import json
    from safetensors.torch import load_file
    
    # Load the conversion config
    config_path = os.path.join(model_path, "conversion_config.json")
    with open(config_path, 'r') as f:
        conversion_config = json.load(f)
    
    # Load the base model first
    base_model_name = conversion_config.get('base_model_name')
    if not base_model_name:
        raise ValueError("Base model name not found in conversion config")
    
    print(f"Loading base model: {base_model_name}")
    
    # Remove device_map from kwargs if present to avoid conflicts
    kwargs_for_base = kwargs.copy()
    if 'device_map' in kwargs_for_base:
        del kwargs_for_base['device_map']
    
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
    base_model = Qwen3ForCausalLM.from_pretrained(
        base_model_name, 
        device_map=device_map_config,
        **kwargs_for_base
    )
    
    # Get the device of the first parameter for reference
    first_param_device = next(base_model.parameters()).device
    print(f"Base model loaded on device: {first_param_device}")
    
    # Reconstruct the MoE architecture
    print("Reconstructing MoE architecture...")
    use_attention_experts = conversion_config.get('use_attention_experts', False)
    use_ffn_experts = conversion_config.get('use_ffn_experts', True)
    num_attention_experts = conversion_config.get('num_attention_experts', 4)
    num_attention_experts_per_token = conversion_config.get('num_attention_experts_per_token', 2)
    num_ffn_experts = conversion_config.get('num_ffn_experts', 8)
    num_ffn_experts_per_token = conversion_config.get('num_ffn_experts_per_token', 2)
    rank = conversion_config.get('rank', 16)
    alpha = conversion_config.get('alpha', 32)
    dropout = conversion_config.get('dropout', 0.1)
    
    # Auxiliary heads config
    use_aux_heads = conversion_config.get('use_aux_heads', False)
    num_aux_heads = conversion_config.get('num_aux_heads', 4)
    aux_loss_alpha = conversion_config.get('aux_loss_alpha', 0.8)
    
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
                dropout=dropout
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
                dropout=dropout
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
        state_dict = torch.load(state_dict_path, map_location='cpu')
    else:
        # Try to load sharded model
        index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
        safetensors_index_path = os.path.join(model_path, "model.safetensors.index.json")
        
        if os.path.exists(safetensors_index_path):
            print("Loading sharded safetensors model...")
            with open(safetensors_index_path, 'r') as f:
                index = json.load(f)
            state_dict = {}
            for filename in set(index['weight_map'].values()):
                shard_path = os.path.join(model_path, filename)
                shard_state_dict = load_file(shard_path)
                state_dict.update(shard_state_dict)
        elif os.path.exists(index_path):
            print("Loading sharded pytorch model...")
            with open(index_path, 'r') as f:
                index = json.load(f)
            state_dict = {}
            for filename in set(index['weight_map'].values()):
                shard_path = os.path.join(model_path, filename)
                shard_state_dict = torch.load(shard_path, map_location='cpu')
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
        print("\n" + "="*60)
        print("Device Placement Verification:")
        print("="*60)
        
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
        
        print("="*60)
    
    # Enable gradient checkpointing if using multiple GPUs for memory efficiency
    if device_map_config == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if hasattr(base_model, 'gradient_checkpointing_enable'):
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
    import os
    import json
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the model using the standard transformers method
    model.save_pretrained(save_path)
    
    # Save the conversion configuration
    if hasattr(model, '_conversion_config'):
        config_path = os.path.join(save_path, "conversion_config.json")
        with open(config_path, 'w') as f:
            json.dump(model._conversion_config, f, indent=2)
        print(f"Conversion config saved to {config_path}")
    else:
        print("Warning: Model doesn't have _conversion_config attribute")
