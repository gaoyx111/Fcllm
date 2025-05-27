import math
import warnings
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import time
import copy
import os
from collections import Counter

import torch
import torch.profiler
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_deepseek import DeepseekConfig, get_Deepseek_config

from weights_download import download_Deepseek_weights
from quantizer import dequantize, quantize
from expert_ARC_cache import ARC_Cache
from utils import memory_cost_deepseek

prefetch_stream = torch.cuda.Stream()
load_stream = torch.cuda.Stream()
evict_expert = torch.cuda.Stream()
quan_expert = torch.cuda.Stream()

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

class DeepseekRMSNorm(nn.Module):
    def __init__(self, hidden_size, device, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.device = device
    
    def init_weights(self, path):
        self.weight.data = torch.load(path, map_location=self.device, weights_only=True)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class DeepseekRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
        self.max_seq_len_cached = None


    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class DeepseekLinearScalingRotaryEmbedding(DeepseekRotaryEmbedding):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class DeepseekDynamicNTKScalingRotaryEmbedding(DeepseekRotaryEmbedding):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepseekMLP(nn.Module):
    def __init__(self, config, layer_idx, is_shared):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_shared = is_shared
        self.quan_bit = self.config.quan_map[layer_idx]

        self.act_fn = ACT2FN[config.hidden_act]
    
    def init_weights(self, path, idx=None, num_in_mem=None):
        # dense layer
        if self.layer_idx == 0:
            path = path + f"/original/model.layers.0.mlp."
            self.gate_proj_path = path + f"gate_proj.weight"
            self.up_proj_path = path + f"up_proj.weight"
            self.down_proj_path = path + f"down_proj.weight"

            self.gate = torch.load(self.gate_proj_path, map_location=self.config.device, weights_only=True)
            self.up = torch.load(self.up_proj_path, map_location=self.config.device, weights_only=True)
            self.down = torch.load(self.down_proj_path, map_location=self.config.device, weights_only=True)
        # shared_expert
        elif idx is None:
            path = path + f"/original/model.layers.{self.layer_idx}.mlp.shared_experts."
            self.gate_proj_path = path + f"gate_proj.weight"
            self.up_proj_path = path + f"up_proj.weight"
            self.down_proj_path = path + f"down_proj.weight"

            self.gate = torch.load(self.gate_proj_path, map_location=self.config.device, weights_only=True)
            self.up = torch.load(self.up_proj_path, map_location=self.config.device, weights_only=True)
            self.down = torch.load(self.down_proj_path, map_location=self.config.device, weights_only=True)
        else:
            self.idx = idx
            init_device = 'cpu'
            self.gate_proj_path = {}
            self.up_proj_path = {}
            self.down_proj_path = {}

            # locate fp16 weights
            path_0 = path + f"/original/model.layers.{self.layer_idx}.mlp.experts.{idx}."
            self.gate_proj_path[0] = path_0 + f"gate_proj.weight"
            self.up_proj_path[0] = path_0 + f"up_proj.weight"
            self.down_proj_path[0] = path_0 + f"down_proj.weight"

            # locate int4/2 weights
            path_4 = path + f"/quantized/int4/model.layers.{self.layer_idx}.mlp.experts.{idx}."
            self.gate_proj_path[4] = path_4 + f"gate_proj.weight"
            self.up_proj_path[4] = path_4 + f"up_proj.weight"
            self.down_proj_path[4] = path_4 + f"down_proj.weight"

            path_2 = path + f"/quantized/int2/model.layers.{self.layer_idx}.mlp.experts.{idx}."
            self.gate_proj_path[2] = path_2 + f"gate_proj.weight"
            self.up_proj_path[2] = path_2 + f"up_proj.weight"
            self.down_proj_path[2] = path_2 + f"down_proj.weight"

            self.gate_cpu = {}
            self.up_cpu = {}
            self.down_cpu = {}

            if self.quan_bit == 0:
                self.gate_cpu[0] = torch.load(self.gate_proj_path[0], map_location=init_device, weights_only=True).pin_memory()
                self.up_cpu[0] = torch.load(self.up_proj_path[0], map_location=init_device, weights_only=True).pin_memory()
                self.down_cpu[0] = torch.load(self.down_proj_path[0], map_location=init_device, weights_only=True).pin_memory()
                self.gate = self.gate_cpu[0].to(self.config.device)
                self.up = self.up_cpu[0].to(self.config.device)
                self.down = self.down_cpu[0].to(self.config.device)
                self.gate_cpu[0] = None
                self.up_cpu[0] = None
                self.down_cpu[0] = None

            self.gate_cpu[4] = torch.load(self.gate_proj_path[4], map_location=init_device, weights_only=True)
            self.up_cpu[4] = torch.load(self.up_proj_path[4], map_location=init_device, weights_only=True)
            self.down_cpu[4] = torch.load(self.down_proj_path[4], map_location=init_device, weights_only=True)

            self.gate_cpu[4] = {k: v.pin_memory() for k, v in self.gate_cpu[4].items()}
            self.up_cpu[4] = {k: v.pin_memory() for k, v in self.up_cpu[4].items()}
            self.down_cpu[4] = {k: v.pin_memory() for k, v in self.down_cpu[4].items()}

            self.gate_cpu[2] = torch.load(self.gate_proj_path[2], map_location=init_device, weights_only=True)
            self.up_cpu[2] = torch.load(self.up_proj_path[2], map_location=init_device, weights_only=True)
            self.down_cpu[2] = torch.load(self.down_proj_path[2], map_location=init_device, weights_only=True)

            self.gate_cpu[2] = {k: v.pin_memory() for k, v in self.gate_cpu[2].items()}
            self.up_cpu[2] = {k: v.pin_memory() for k, v in self.up_cpu[2].items()}
            self.down_cpu[2] = {k: v.pin_memory() for k, v in self.down_cpu[2].items()}

            if num_in_mem is not None and idx < num_in_mem:
                if self.quan_bit != 0:
                    self.gate = {k: v.to(self.config.device) for k, v in self.gate_cpu[self.quan_bit].items()}
                    self.up = {k: v.to(self.config.device) for k, v in self.up_cpu[self.quan_bit].items()}
                    self.down = {k: v.to(self.config.device) for k, v in self.down_cpu[self.quan_bit].items()}
    
    def load_from_cpu(self, weight, is_now):
        non_blocking = False if is_now else True
        return {
            'nbits': weight['nbits'],
            'shape': weight['shape'],
            'W_q': weight['W_q'].to(self.config.device, non_blocking=non_blocking),
            'scale': weight['scale'].to(self.config.device, non_blocking=non_blocking),
            'zero': weight['zero'].to(self.config.device, non_blocking=non_blocking)
        }

    def load_weights(self, is_now=False, nbit=None):
        quan_bit = self.quan_bit if nbit is None else nbit
        if is_now:
            self.gate = self.load_from_cpu(self.gate_cpu[quan_bit], is_now)
            self.up = self.load_from_cpu(self.up_cpu[quan_bit], is_now)
            self.down = self.load_from_cpu(self.down_cpu[quan_bit], is_now)
            # self.gate = {k: v.to(self.config.device) for k, v in self.gate_cpu[self.quan_bit].items()}
            # self.up = {k: v.to(self.config.device) for k, v in self.up_cpu[self.quan_bit].items()}
            # self.down = {k: v.to(self.config.device) for k, v in self.down_cpu[self.quan_bit].items()}
        else:
            prefetch_stream = torch.cuda.Stream()
            with torch.cuda.stream(prefetch_stream):
                self.gate = self.load_from_cpu(self.gate_cpu[quan_bit], is_now)
                self.up = self.load_from_cpu(self.up_cpu[quan_bit], is_now)
                self.down = self.load_from_cpu(self.down_cpu[quan_bit], is_now)

    def dequan_experts(self):
        if not self.is_shared and self.quan_bit != 0:
            self.gate = dequantize(self.gate)
            self.up = dequantize(self.up)
            self.down = dequantize(self.down)
    
    def quan_experts(self):
        if self.quan_bit != 0:
            quan_stream = torch.cuda.Stream()
            is_now = False
            with torch.cuda.stream(quan_stream):
                self.gate = self.load_from_cpu(self.gate_cpu[self.quan_bit], is_now)
                self.up = self.load_from_cpu(self.up_cpu[self.quan_bit], is_now)
                self.down = self.load_from_cpu(self.down_cpu[self.quan_bit], is_now)
            # self.gate = quantize(self.gate, self.quan_bit)
            # self.up = quantize(self.up, self.quan_bit)
            # self.down = quantize(self.down, self.quan_bit)

    def clear(self):
        self.gate = None
        self.up = None
        self.down = None

    def forward(self, x, prefetch_expert_list=None):
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # return down_proj
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down) 


class MoEGate(nn.Module):
    def __init__(self, config, device, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.device = device

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
    
    def init_weights(self, path):
        self.weight.data = torch.load(path, map_location=self.device, weights_only=True)
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape        
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1)
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        return topk_idx, topk_weight


class DeepseekMoE(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.layer_idx = layer_idx
        self.device = config.device
        self.num_in_mem = int(self.n_routed_experts - config.offload_map[layer_idx])
        self.arc_cache = ARC_Cache(self.num_in_mem)

        self.experts = nn.ModuleList([DeepseekMLP(config, layer_idx, False) for _ in range(self.n_routed_experts)])
        self.gate = MoEGate(config, config.device, layer_idx)
        
        self.shared_experts = DeepseekMLP(config, layer_idx, True)

    def init_weights(self, path):
        gate_path = path + f"/original/model.layers.{self.layer_idx}.mlp.gate.weight"
        self.gate.init_weights(gate_path)
        
        for idx in range(self.n_routed_experts):
            self.experts[idx].init_weights(path, idx, self.num_in_mem)
        self.shared_experts.init_weights(path)
    
    def load_weights(self, idx, is_now=False, int2_experts=None):
        # load_stream
        if isinstance(idx, int):
            if self.arc_cache.is_evicted(idx):
                self.experts[idx].load_weights(is_now=is_now, nbit=2)
        # prefetch_stream
        else:
            for i in idx:
                if self.arc_cache.is_evicted(i):
                    nbit = 2 if i in int2_experts else 4
                    self.experts[i].load_weights(is_now=is_now, nbit=nbit)
    
    def post_comp(self, expert_idx):
        if self.num_in_mem == 0:
            self.experts[expert_idx].clear()
        elif self.layer_idx != 0:
            if self.arc_cache.is_evicted(expert_idx):
                self.experts[expert_idx].clear()
            else:
                quan_expert = torch.cuda.Stream()
                with torch.cuda.stream(quan_expert):
                    self.experts[expert_idx].quan_experts()
    
    def forward(self, hidden_states, prefetch_expert_idx):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)

        flat_topk_idx = topk_idx.view(-1)

        load_experts = []
        expert_index = flat_topk_idx.tolist()

        if self.layer_idx == 0 or prefetch_expert_idx is None:
            prefetch_expert_idx = list(set(expert_index))
            evicted_list = []
        else:
            load_stream = torch.cuda.Stream()
            with torch.cuda.stream(load_stream): 
                freq_counter = Counter(expert_index)
                freq_counter = [item[0] for item in sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)]
                for idx in freq_counter:
                    if idx not in prefetch_expert_idx:
                        self.load_weights(idx)
                        load_experts.append(idx)
            
            if self.num_in_mem != 0:
                evicted_list = self.arc_cache.update_list(expert_index)
                for idx in evicted_list:
                    self.experts[idx].clear()

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        flat_expert_weights = topk_weight.view(-1, 1)

        expert_cache = torch.zeros_like(hidden_states)
        idxs = flat_topk_idx.argsort()
        tokens_per_expert = flat_topk_idx.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok

        # experts comp.
        # comp. experts which had been prefetched
        for expert_idx in prefetch_expert_idx:
            if expert_idx not in expert_index:
                if self.arc_cache.is_evicted(expert_idx):
                    self.experts[expert_idx].clear()
                continue
            
            self.experts[expert_idx].dequan_experts()

            start_idx = 0 if expert_idx == 0 else tokens_per_expert[expert_idx-1]
            end_idx = tokens_per_expert[expert_idx]
            if start_idx == end_idx:
                continue
            expert = self.experts[expert_idx]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = hidden_states[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, hidden_states.shape[-1]), expert_out, reduce='sum')
            
            self.post_comp(expert_idx)

        if len(load_experts) > 0:
            for expert_idx in load_experts:
                self.experts[expert_idx].dequan_experts()
                start_idx = 0 if expert_idx == 0 else tokens_per_expert[expert_idx-1]
                end_idx = tokens_per_expert[expert_idx]
                if start_idx == end_idx:
                    continue
                expert = self.experts[expert_idx]
                exp_token_idx = token_idxs[start_idx:end_idx]
                expert_tokens = hidden_states[exp_token_idx]
                expert_out = expert(expert_tokens)
                expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
                expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, hidden_states.shape[-1]), expert_out, reduce='sum')

                self.post_comp(expert_idx)
        
        y = expert_cache.view(*orig_shape)
        # y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y


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


class DeepseekAttention(nn.Module):

    def __init__(self, config: DeepseekConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def init_weights(self, path):
        path = path + f"/original/model.layers.{self.layer_idx}.self_attn."
        self.q_proj.weight.data = torch.load(path+"q_proj.weight", map_location=self.config.device, weights_only=True)
        self.k_proj.weight.data = torch.load(path+"k_proj.weight", map_location=self.config.device, weights_only=True)
        self.v_proj.weight.data = torch.load(path+"v_proj.weight", map_location=self.config.device, weights_only=True)
        self.o_proj.weight.data = torch.load(path+"o_proj.weight", map_location=self.config.device, weights_only=True)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class DeepseekSdpaAttention(DeepseekAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


Deepseek_ATTENTION_CLASSES = {
    "eager": DeepseekAttention,
    "sdpa": DeepseekSdpaAttention,
}

class DeepseekDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Deepseek_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = DeepseekMoE(config, layer_idx) if (layer_idx >= config.first_k_dense_replace) \
                                                    else DeepseekMLP(config, layer_idx, False)
        
        self.input_layernorm = DeepseekRMSNorm(config.hidden_size, self.config.device, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekRMSNorm(config.hidden_size, self.config.device, eps=config.rms_norm_eps)

        # self.next_gate_cpu = MoEGate(config, 'cpu', layer_idx)
        self.next_gate_cpu = MoEGate(config, self.config.device, layer_idx)
    
    def init_weights(self, path):
        self.self_attn.init_weights(path)
        self.mlp.init_weights(path)

        input_ln_path = path + f"/original/model.layers.{self.layer_idx}.input_layernorm.weight"
        post_ln_path = path + f"/original/model.layers.{self.layer_idx}.post_attention_layernorm.weight"
        self.input_layernorm.init_weights(input_ln_path)
        self.post_attention_layernorm.init_weights(post_ln_path)

        if self.layer_idx < self.config.num_hidden_layers - 1:
            gate_path = path + f"/original/model.layers.{self.layer_idx+1}.mlp.gate.weight"
            self.next_gate_cpu.init_weights(gate_path)
            
    def forward(
        self, hidden_states, attention_mask = None, position_ids = None,
        past_key_value = None, prefetch_expert_list = None, next_layer = None
    ):

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # not prefetch only when (not offload) and (not quantize)
        next_prefetch_expert_list = None
        # allocated_memory = torch.cuda.memory_allocated(self.config.device) / (1024**2)
        # print("allocated_memory: ", allocated_memory)
        if self.layer_idx < self.config.num_hidden_layers-1 and self.config.offload_map[self.layer_idx+1] != 0:
            load_stream = torch.cuda.Stream()
            with torch.cuda.stream(load_stream):
                hidden_cpu = hidden_states.clone()
                # hidden_cpu = hidden_states.clone().to('cpu', non_blocking=True)
                next_prefetch_expert_list, _ = self.next_gate_cpu(hidden_cpu)
                next_prefetch_expert_list = next_prefetch_expert_list.view(-1).tolist()
                next_prefetch_expert_list = Counter(next_prefetch_expert_list)
                most_common_items = next_prefetch_expert_list.most_common()
                next_prefetch_expert_list = [item[0] for item in most_common_items]

                # determine which experts to int2
                next_prefetch_expert_dict = dict(most_common_items)
                value_sum = sum(next_prefetch_expert_dict.values())
                target_sum = value_sum * 0.3
                current_sum = 0
                int2_experts = []
                for key, value in reversed(next_prefetch_expert_dict.items()):
                    if current_sum + value > target_sum:
                        break
                    current_sum += value
                    int2_experts.append(key)
                next_layer.mlp.load_weights(next_prefetch_expert_list, int2_experts=int2_experts)
                del hidden_cpu

        hidden_states = self.mlp(hidden_states, prefetch_expert_list)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, next_prefetch_expert_list)

        return outputs


class DeepseekModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self._use_sdpa = config._attn_implementation == "sdpa"

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeepseekDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekRMSNorm(config.hidden_size, self.config.device, eps=config.rms_norm_eps)

    def init_weights(self, path):
        # for i in range(self.config.num_hidden_layers):
        for i in tqdm(range(self.config.num_hidden_layers), desc="Init."):
            self.layers[i].init_weights(path)
        
        ln_path = path + "/original/model.norm.weight"
        embed_path = path + "/original/model.embed_tokens.weight"
        self.norm.init_weights(ln_path)
        self.embed_tokens.weight.data = torch.load(embed_path, map_location=self.config.device, weights_only=True)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Tuple:
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        # use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = None
        next_prefetch_expert_list = None

        # for decoder_layer in self.layers:
        for i in range(self.config.num_hidden_layers):
            layer_outputs = self.layers[i](
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                prefetch_expert_list=next_prefetch_expert_list,
                next_layer=self.layers[i+1] if i<self.config.num_hidden_layers-1 else None
            )

            # layer_outputs = decoder_layer(
            #     hidden_states,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     past_key_value=past_key_values,
            # )

            hidden_states = layer_outputs[0]

            # use_cache:
            next_decoder_cache = layer_outputs[1]
            next_prefetch_expert_list = layer_outputs[2]

        hidden_states = self.norm(hidden_states)

        next_cache = None
        # use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        return tuple(v for v in [hidden_states, next_cache] if v is not None)


class DeepseekForCausalLM(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.config = get_Deepseek_config(args.model)
        self.config.device = args.device
        self.device = args.device
        self.min_length = args.min_length
        self.max_length = args.max_length
        self.early_stopping = args.early_stopping
        self.path = args.path
        self.config._attn_implementation = "sdpa"

        (self.config.offload_map, self.config.quan_map) = memory_cost_deepseek(self.config, args.memory_budget)
        print("offload_map", self.config.offload_map)
        print("quan_map", self.config.quan_map)
        # assert False
        
        self.model = DeepseekModel(self.config)
        self.vocab_size = self.config.vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.init_weights()

    def init_weights(self):
        # expanded_path = os.path.abspath(os.path.expanduser(os.path.join(self.path, "deepseek-moe-16b-base")))
        expanded_path = os.path.abspath(os.path.expanduser(os.path.join(self.path, "deepseek-ai", "deepseek-moe-16b-base")))
        check_path = os.path.join(expanded_path, "original/lm_head.weight")
        if not os.path.exists(check_path):
            print("1111")
            assert False
            download_Deepseek_weights("deepseek-ai/deepseek-moe-16b-base", self.path)

        self.model.init_weights(expanded_path)
        self.lm_head.weight.data = torch.load(check_path, map_location=self.config.device, weights_only=True)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> Tuple:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[0]
 
        logits = self.lm_head(hidden_states)
        # logits = logits.float()

        # return (logits,) + outputs[1:]
        return logits, outputs[1]
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, expriment_mode=None):
        prefill_time = 0
        if expriment_mode == "decoding":
            prefill_start_time = time.time()
        seq_len = input_ids.shape[1]
        past_key_values = DynamicCache()
        if attention_mask is None:
            attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=self.config.device)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=self.config.device).unsqueeze(0)

        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
        #     schedule=torch.profiler.schedule(
        #         wait=0,  # 前1步不采样
        #         warmup=0,  # 第2步作为热身，不计入结果
        #         active=1,  # 采集后面3步的性能数据
        #         repeat=3),  # 重复2轮
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./deepseek_1210log'),  # 保存日志以供 TensorBoard 可视化
        #     record_shapes=True,  # 记录输入张量的形状
        #     profile_memory=True,  # 分析内存分配
        #     with_stack=True  # 记录操作的调用堆栈信息
        # ) as profiler:

        for i in range(1024):
        # for i in tqdm(range(32), desc="Infer."):
            logits, past_key_values = self.forward(input_ids=input_ids, 
                                                attention_mask=attention_mask,
                                                position_ids=position_ids,
                                                past_key_values=past_key_values,
                                                )
            logits = F.softmax(logits, dim=-1)

            # greedy search:
            input_ids = torch.argmax(logits, dim=-1)
            if i == 0:
                output = copy.deepcopy(input_ids[:, -1:])
            else:
                output = torch.cat((output, input_ids), dim=1)

            input_ids = input_ids[:, -1:]
            # if self.early_stopping and i > self.min_length and input_ids.item() == self.config.eos_token_id:
            #     return output
            # print("self.config.eos_token_id: ", self.config.eos_token_id)
            if input_ids.item() == self.config.eos_token_id:
                return (output, prefill_time)
            
            if i == 0:
                if expriment_mode == "prefill":
                    return (output, None)
                elif expriment_mode == "decoding":
                    prefill_time = time.time() - prefill_start_time
            
            # prepare for next decoding.
            position_ids = (position_ids[:, -1] + 1).unsqueeze(-1)
            attention_mask = torch.cat([attention_mask, torch.ones(1, 1, dtype=torch.long, device=self.config.device)], dim=-1)

            # profiler.step()

        return (output, prefill_time)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


