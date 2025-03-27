import math
import struct
import inspect
import time

from .LMConfig import LMConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# Attempt to import Transformer Engine components
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    has_te = True
except ImportError:
    te = None
    # Define dummy classes if TE is not available to prevent import errors
    # These won't be used due to the `has_te` checks, but prevent crashes.
    Format = type('Format', (), {'HYBRID': 'HYBRID', 'E4M3': 'E4M3', 'E5M2': 'E5M2'})
    DelayedScaling = type('DelayedScaling', (), {})
    has_te = False
    print("Transformer Engine not found. FP8 features will be disabled.")


# Keep the original RMSNorm for fallback when TE is not available or not enabled
class RMSNorm(torch.nn.Module):
    """ Standard RMSNorm implementation """
    def __init__(self, dim: int, eps: float = 1e-6): # Default eps
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Compute Root Mean Square and normalize
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Cast to float for stability, normalize, cast back, apply weight
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    """ Precompute Rotary Positional Embeddings (RoPE) """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) # Use default device
    freqs = torch.outer(t, freqs).float()
    # Compute complex numbers in polar form: exp(i * freqs)
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):
    """ Apply Rotary Positional Embeddings to query and key tensors """
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # Ensure sequence length matches pos_cis dimension 0
        assert freqs_cis.shape[0] >= x.shape[1], f"pos_cis seq length {freqs_cis.shape[0]} must be >= input seq length {x.shape[1]}"
        # Ensure head dimension matches pos_cis dimension 1
        assert freqs_cis.shape[1] == x.shape[-1], f"pos_cis head dim {freqs_cis.shape[1]} must match input head dim {x.shape[-1]}"
        # Slice pos_cis to match input sequence length
        freqs_cis = freqs_cis[:x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    # Reshape xq and xk to view the last dimension as complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Reshape pos_cis for broadcasting and slice to match seq len
    pos_cis = reshape_for_broadcast(pos_cis, xq_)
    # Apply rotation in complex plane: x * exp(i*theta)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """ Repeat keys and values for Grouped Query Attention """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # Expand and reshape to repeat KV heads: [bs, slen, n_kv_heads, n_rep, head_dim] -> [bs, slen, n_kv_heads * n_rep, head_dim]
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """ Multi-Head Attention module with optional Grouped Query Attention and FP8 support """
    def __init__(self, args: LMConfig, fp8_enabled: bool = False):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads # Number of query heads
        self.n_local_kv_heads = self.n_kv_heads # Number of key/value heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # Repetition factor for GQA
        self.head_dim = args.dim // args.n_heads
        self.fp8_enabled = fp8_enabled and has_te # Use FP8 only if enabled and TE is available

        q_dim = args.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        # Initialize linear layers: use TE Linear if FP8 enabled, otherwise standard nn.Linear
        linear_cls = te.Linear if self.fp8_enabled else nn.Linear
        self.wq = linear_cls(args.dim, q_dim, bias=False)
        self.wk = linear_cls(args.dim, kv_dim, bias=False)
        self.wv = linear_cls(args.dim, kv_dim, bias=False)
        self.wo = linear_cls(q_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # Check if flash attention is available and enabled
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        # Precompute causal mask for manual attention calculation
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor, # Rotary embeddings
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                is_first_microbatch: Optional[bool] = None): # Optional for TE advanced features
        bsz, seq_len, _ = x.shape

        # Apply QKV projections
        # Pass is_first_microbatch if TE layer requires it (commented out for now)
        # print(f"Attention.forward: Input x shape = {x.shape}, dtype = {x.dtype}, is_contiguous = {x.is_contiguous()}") # Debug
        xq = self.wq(x) #, is_first_microbatch=is_first_microbatch)
        xk = self.wk(x) #, is_first_microbatch=is_first_microbatch)
        xv = self.wv(x) #, is_first_microbatch=is_first_microbatch)

        # Reshape for multi-head attention
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # KV Caching
        if past_key_value is not None:
            # Check for batch size consistency before concatenating
            if past_key_value[0].shape[0] == bsz and past_key_value[1].shape[0] == bsz:
                 xk = torch.cat([past_key_value[0], xk], dim=1)
                 xv = torch.cat([past_key_value[1], xv], dim=1)
            else:
                 # Handle potential shape mismatch, e.g., during generation start
                 print(f"Warning: KV cache batch size mismatch. Expected {bsz}, got {past_key_value[0].shape[0]}. Cache might not be used correctly this step.")
                 # If generating (seq_len=1), we might just use the current xk, xv
                 # If training, this indicates a problem.
                 if seq_len != 1:
                     # Re-attempt concatenation or raise error depending on desired behavior
                     try:
                         xk = torch.cat([past_key_value[0], xk], dim=1)
                         xv = torch.cat([past_key_value[1], xv], dim=1)
                     except RuntimeError as e:
                         print(f"Error concatenating KV cache: {e}")
                         # Decide fallback: error out, or proceed without cache this step?
                         # For now, let's proceed without cache if concat fails after warning
                         pass # xk, xv remain as calculated for this step only

        # Store current keys/values for caching if requested
        present_kv = (xk, xv) if use_cache else None

        # Reshape and transpose for attention calculation
        xq = xq.transpose(1, 2) # [bsz, n_local_heads, seq_len, head_dim]
        # Repeat KV heads if using Grouped Query Attention (GQA)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2) # [bsz, n_local_heads, kv_seq_len, head_dim]
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2) # [bsz, n_local_heads, kv_seq_len, head_dim]

        # Perform attention calculation
        if self.flash and seq_len > 1: # Use Flash Attention if available and seq_len > 1
            dropout_p = self.dropout if self.training else 0.0
            # Use is_causal=True for efficient causal masking in Flash Attention
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None, # Causal mask is handled by is_causal=True
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            # Manual attention calculation
            # Calculate attention scores: (Query @ Key.T) / sqrt(head_dim)
            scores = (xq @ xk.transpose(-2, -1)) * (self.head_dim**-0.5)
            # Apply causal mask
            kv_seq_len = xk.shape[-2] # Get length of key sequence (can be different due to cache)
            current_mask = self.mask[:, :, :seq_len, :kv_seq_len]
            scores = scores + current_mask.to(dtype=scores.dtype) # Ensure mask dtype matches scores
            # Apply softmax and dropout
            scores = F.softmax(scores.float(), dim=-1).type_as(xq) # Softmax in float32 for stability
            scores = self.attn_dropout(scores)
            # Calculate output: AttentionScores @ Value
            output = scores @ xv # [bsz, n_local_heads, seq_len, head_dim]

        # Reshape output and apply final projection
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1) # [bsz, seq_len, dim]
        output = self.resid_dropout(self.wo(output))#, is_first_microbatch=is_first_microbatch)) # Pass if needed

        return output, present_kv


class FeedForward(nn.Module):
    """ Standard FeedForward network (SwiGLU variant) with FP8 support """
    def __init__(self, config: LMConfig, fp8_enabled: bool = False):
        super().__init__()
        # Calculate hidden dimension if not provided
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3) # Reduce dimension following SwiGLU paper
            # Ensure hidden_dim is multiple of 'multiple_of'
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.fp8_enabled = fp8_enabled and has_te # Use FP8 only if enabled and TE is available

        # Initialize linear layers: use TE Linear if FP8 enabled
        linear_cls = te.Linear if self.fp8_enabled else nn.Linear
        self.w1 = linear_cls(config.dim, config.hidden_dim, bias=False) # Gate projection
        self.w3 = linear_cls(config.dim, config.hidden_dim, bias=False) # Up projection
        self.w2 = linear_cls(config.hidden_dim, config.dim, bias=False) # Down projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU activation: silu(w1(x)) * w3(x)
        # TE handles FP8 conversion within fp8_autocast context.
        # Activation (F.silu) runs in the higher precision (e.g., BF16) specified by AMP autocast.
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        hidden = gate * up
        # Apply dropout and down projection
        output = self.dropout(self.w2(hidden))
        return output


class MoEGate(nn.Module):
    """ Gating network for Mixture of Experts """
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok # Number of experts to route each token to
        self.n_routed_experts = config.n_routed_experts # Total number of experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha # Coefficient for auxiliary load balancing loss
        self.seq_aux = config.seq_aux # Whether to calculate aux loss per sequence or per token

        self.norm_topk_prob = config.norm_topk_prob # Normalize probabilities over the selected top-k experts
        self.gating_dim = config.dim
        # Linear layer to compute routing logits
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize gating weights
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h_dim = hidden_states.shape
        tokens = hidden_states.view(-1, h_dim) # Flatten input tokens: [bsz * seq_len, h_dim]

        # Compute routing logits: [bsz * seq_len, n_routed_experts]
        logits = F.linear(tokens.float(), self.weight) # Use float32 for stability in gating

        # Compute routing scores (probabilities)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            # Add other scoring functions if needed (e.g., sigmoid for multi-select)
            raise NotImplementedError(f'Unsupported scoring function for MoE gating: {self.scoring_func}')

        # Select top-k experts for each token
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # Normalize weights among top-k experts if enabled
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20 # Add epsilon for stability
            topk_weight = topk_weight / denominator

        # Calculate auxiliary load balancing loss during training
        aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        if self.training and self.alpha > 0.0:
            if self.seq_aux:
                # Calculate load per expert per sequence
                scores_reshaped = scores.view(bsz, seq_len, self.n_routed_experts)
                # One-hot encode expert choices per token, sum over sequence
                one_hot_choices = F.one_hot(topk_idx, num_classes=self.n_routed_experts).sum(dim=1) # Sum over top_k dim -> [bsz*seq_len, n_experts]
                load_per_token = one_hot_choices.view(bsz, seq_len, self.n_routed_experts).float()
                load_per_seq = load_per_token.sum(dim=1) # Sum over seq_len -> [bsz, n_experts]
                load_fraction = load_per_seq / (seq_len * self.top_k) # Fraction of tokens routed to each expert in seq

                # Calculate importance (average routing probability) per expert per sequence
                importance = scores_reshaped.mean(dim=1) # Average score over sequence length

                # Aux loss = sum(load_fraction * importance) averaged over batch, scaled
                aux_loss = (load_fraction * importance).sum() / bsz * self.alpha * self.n_routed_experts

            else:
                # Calculate load per expert across all tokens (frequency)
                mask_ce = F.one_hot(topk_idx.view(-1), num_classes=self.n_routed_experts)
                load = mask_ce.float().mean(0) # Average load per expert

                # Calculate importance per expert across all tokens (average probability)
                importance = scores.mean(0) # Average score per expert

                # Aux loss = sum(load * importance) * num_experts
                aux_loss = (load * importance).sum() * self.alpha * self.n_routed_experts

        # Return expert indices, weights, and auxiliary loss
        # Ensure weights and loss match the hidden_states dtype
        return topk_idx, topk_weight.type_as(hidden_states), aux_loss.type_as(hidden_states)


class MOEFeedForward(nn.Module):
    """ Mixture of Experts FeedForward network """
    def __init__(self, config: LMConfig, fp8_enabled: bool = False):
        super().__init__()
        self.config = config
        self.fp8_enabled = fp8_enabled and has_te # Use FP8 only if enabled and TE is available

        # Initialize expert networks, passing fp8_enabled flag
        self.experts = nn.ModuleList([
            FeedForward(config, fp8_enabled=self.fp8_enabled)
            for _ in range(config.n_routed_experts)
        ])
        # Initialize gating network
        self.gate = MoEGate(config)
        # Initialize shared expert if configured
        if config.n_shared_experts is not None and config.n_shared_experts > 0:
            self.shared_experts = FeedForward(config, fp8_enabled=self.fp8_enabled)
        else:
            self.shared_experts = None

        self.aux_loss = torch.tensor(0.0) # Initialize aux_loss attribute

    def forward(self, x):
        """
        Forward pass for MoE. Routes tokens to experts based on gating decisions.

        NOTE: This implementation uses a loop and index_add_ for dispatch/combine,
              which is functionally correct but NOT performant on GPUs.
              For high performance, optimized MoE kernels (e.g., using permutation/
              unpermutation like in TE's moe_permute/moe_unpermute, or libraries
              like Tutel) are necessary. This code serves as a conceptual example
              integrating FP8 experts.
        """
        identity = x # Store input for residual connection / shared expert
        orig_shape = x.shape
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim) # Flatten tokens: [bsz * seq_len, dim]

        # Get routing decisions from the gate
        topk_idx, topk_weight, aux_loss = self.gate(identity) # Gate uses original shape input
        self.aux_loss = aux_loss # Store aux loss for retrieval

        # Prepare for dispatch
        flat_topk_idx = topk_idx.view(-1) # Expert indices for each token*top_k: [bsz * seq_len * top_k]
        flat_topk_weight = topk_weight.view(-1) # Weights for each token*top_k: [bsz * seq_len * top_k]
        y = torch.zeros_like(x_flat) # Output buffer

        # Expand input tokens and weights for top-k processing
        # Each token is processed by top_k experts
        expanded_x = x_flat.repeat_interleave(self.config.num_experts_per_tok, dim=0)

        # Loop through experts (Inefficient - see NOTE above)
        for i, expert in enumerate(self.experts):
            # Find which expanded token entries are routed to this expert
            expert_mask = (flat_topk_idx == i)
            if expert_mask.any():
                # Get the indices of the expanded tokens for this expert
                expanded_indices = expert_mask.nonzero(as_tuple=True)[0]
                # Select the corresponding input tokens and weights
                tokens_for_expert = expanded_x[expanded_indices]
                weights_for_expert = flat_topk_weight[expanded_indices]

                # Apply the expert network
                expert_output = expert(tokens_for_expert)

                # Weight the expert output
                weighted_output = expert_output * weights_for_expert.unsqueeze(-1)

                # Combine results: Add weighted output back to the correct original token position
                # Find original token indices corresponding to these expanded indices
                original_token_indices = expanded_indices // self.config.num_experts_per_tok
                # Use index_add_ for safe accumulation if multiple experts route to the same token
                y.index_add_(0, original_token_indices, weighted_output.to(y.dtype)) # Ensure dtype match

        # Reshape output back to original shape
        y = y.view(*orig_shape)

        # Add shared expert output if applicable
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y

    # Inference-specific optimization (can be faster than training loop if implemented efficiently)
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """ More optimized inference path using sorting (still potentially slow without kernels) """
        expert_cache = torch.zeros_like(x) # Output buffer
        # Sort tokens by the expert they are assigned to (for batching expert calls)
        idxs = flat_expert_indices.argsort()
        # Calculate cumulative counts of tokens per expert
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # Get the original token index for each sorted entry
        token_idxs = idxs // self.config.num_experts_per_tok # [num_tokens * top_k]

        last_idx = 0
        for i, end_idx in enumerate(tokens_per_expert):
            if last_idx == end_idx: continue # Skip if no tokens for this expert
            expert = self.experts[i]
            # Get original token indices for this expert's batch
            current_sorted_indices = idxs[last_idx:end_idx] # Indices within the sorted flat list
            exp_token_idx = token_idxs[last_idx:end_idx] # Original token indices for this batch
            # Get the corresponding input tokens
            expert_tokens = x[exp_token_idx]
            # Apply expert
            expert_out = expert(expert_tokens).to(expert_cache.dtype) # Ensure dtype match
            # Apply weights corresponding to this expert's batch
            expert_out.mul_(flat_expert_weights[current_sorted_indices].unsqueeze(-1)) # Use sorted indices to get weights
            # Accumulate results using scatter_add_ for safety
            # Ensure index tensor has the same number of dimensions as the source tensor for broadcasting
            index_for_scatter = exp_token_idx.unsqueeze(-1).expand_as(expert_out)
            expert_cache.scatter_add_(0, index_for_scatter, expert_out)
            last_idx = end_idx

        return expert_cache


class MiniMindBlock(nn.Module):
    """ A single Transformer block with Attention and FeedForward (or MoE) layers """
    def __init__(self, layer_id: int, config: LMConfig, fp8_enabled: bool = False):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.fp8_enabled = fp8_enabled and has_te # Use FP8 only if enabled and TE is available
        self.layer_id = layer_id
        self.config = config # Store config

        # Use TE RMSNorm if FP8 is enabled, otherwise standard RMSNorm
        norm_class = te.RMSNorm if self.fp8_enabled else RMSNorm
        self.attention_norm = norm_class(config.dim, eps=config.norm_eps)
        self.ffn_norm = norm_class(config.dim, eps=config.norm_eps)

        # Initialize Attention layer, passing fp8_enabled flag
        self.attention = Attention(config, fp8_enabled=self.fp8_enabled)

        # Initialize FeedForward or MoE layer, passing fp8_enabled flag
        ff_class = MOEFeedForward if config.use_moe else FeedForward
        self.feed_forward = ff_class(config, fp8_enabled=self.fp8_enabled)

        # Initialize aux_loss attribute for consistent access
        self.aux_loss = torch.tensor(0.0)

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor, # Rotary embeddings
                past_key_value=None,
                use_cache=False,
                is_first_microbatch: Optional[bool] = None): # Pass down if needed

        # --- Self-Attention Block ---
        # Apply pre-normalization
        attn_input = self.attention_norm(x)
        # Perform attention
        h_attn, present_kv = self.attention(
            attn_input,
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache,
            is_first_microbatch=is_first_microbatch # Pass down
        )
        # Apply residual connection
        h = x + h_attn

        # --- FeedForward Block ---
        # Apply pre-normalization
        ffn_input = self.ffn_norm(h)
        # Apply FeedForward or MoE layer
        ffn_output = self.feed_forward(ffn_input)
        # Apply residual connection
        out = h + ffn_output

        # Retrieve auxiliary loss if using MoE
        if self.config.use_moe:
            # MOEFeedForward stores its aux_loss internally
            self.aux_loss = getattr(self.feed_forward, 'aux_loss', torch.tensor(0.0, device=x.device))
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device)

        return out, present_kv


class MiniMindLM(PreTrainedModel):
    """ The main Language Model class """
    config_class = LMConfig # Link to the configuration class

    def __init__(self, params: LMConfig = None, fp8_enabled: bool = False):
        # Use default config if none provided
        if params is None:
            params = LMConfig()
        super().__init__(params) # Initialize PreTrainedModel with config
        self.params = params # Store config
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        # Store fp8_enabled status, ensuring TE is available if True
        self.fp8_enabled = fp8_enabled and has_te

        # --- Model Layers ---
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)

        # Create Transformer blocks, passing the fp8_enabled flag
        self.layers = nn.ModuleList([
            MiniMindBlock(l, params, fp8_enabled=self.fp8_enabled)
            for l in range(self.n_layers)
        ])

        # Final normalization layer: use TE RMSNorm if FP8 enabled
        norm_class = te.RMSNorm if self.fp8_enabled else RMSNorm
        self.norm = norm_class(params.dim, eps=params.norm_eps)

        # Output layer (language model head)
        # Typically kept in higher precision, but could be te.Linear if needed.
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # --- Weight Tying ---
        # Tie input embeddings and output projection weights
        self.tok_embeddings.weight = self.output.weight

        # --- Positional Embeddings ---
        # Precompute Rotary Positional Embeddings (RoPE)
        # Increase buffer size slightly beyond max_seq_len for flexibility if needed
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads,
                                                end=params.max_seq_len * 2, # Buffer size
                                                theta=params.rope_theta),
                             persistent=False) # Not part of state_dict

        # --- FP8 Recipe Initialization ---
        self.fp8_recipe = None
        if self.fp8_enabled:
            # Define a default FP8 recipe. Can be overridden by training script.
            self.fp8_recipe = DelayedScaling(
                margin=0,
                fp8_format=Format.HYBRID, # Common choice: E4M3 fwd, E5M2 bwd
                amax_history_len=16, # Example value
                amax_compute_algo="max" # Example value
            )
            print("Transformer Engine FP8 is enabled with default recipe.")

        # --- Weight Initialization ---
        # Apply initialization after all layers are defined
        self.post_init()


    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False,
                output_attentions: Optional[bool] = None, # Standard HF arg
                output_hidden_states: Optional[bool] = None, # Standard HF arg
                return_dict: Optional[bool] = None, # Standard HF arg
                # Add is_first_microbatch for potential TE use in submodules
                is_first_microbatch: Optional[bool] = None,
                **kwargs): # Use kwargs to capture extra arguments like start_pos

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        start_pos = kwargs.get('start_pos', 0) # Get start_pos for positional embeddings

        # 1. Token Embeddings
        hidden_states = self.tok_embeddings(input_ids)
        hidden_states = self.dropout(hidden_states)

        # Retrieve positional encodings for the current sequence segment
        seq_len = input_ids.shape[1]
        # Ensure pos_cis has enough length for the current start_pos + seq_len
        required_len = start_pos + seq_len
        if required_len > self.pos_cis.shape[0]:
             raise ValueError(f"Input sequence length ({required_len}) exceeds precomputed RoPE buffer size ({self.pos_cis.shape[0]})")
        pos_cis = self.pos_cis[start_pos : required_len]

        # Initialize past_key_values if not provided or if needed
        if past_key_values is None:
            past_key_values = [None] * self.n_layers

        # Lists to store outputs if requested
        present_key_values = [] if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None # Not implemented in Attention yet
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        # 2. Transformer Blocks
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_past = past_key_values[i]

            # Pass arguments to the block's forward method
            # Includes hidden_states, pos_cis, past_kv, use_cache, and is_first_microbatch
            layer_outputs = layer(
                hidden_states,
                pos_cis,
                past_key_value=layer_past,
                use_cache=use_cache,
                is_first_microbatch=is_first_microbatch # Pass down
            )

            hidden_states = layer_outputs[0] # Main output of the block
            if use_cache:
                present_key_values.append(layer_outputs[1]) # Store new KV cache state

            # Accumulate auxiliary loss from MoE layers within the block
            # Assumes the block stores its aux_loss attribute after its forward pass
            total_aux_loss += getattr(layer, 'aux_loss', torch.tensor(0.0, device=hidden_states.device))

            # Store attentions if needed (requires modification in Attention layer to return them)
            # if output_attentions:
            #     all_self_attns = all_self_attns + (layer_outputs[2],) # Assuming attention output is 3rd element

        # 3. Final Norm and Output Head
        hidden_states = self.norm(hidden_states) # Apply final normalization

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,) # Add final hidden state

        logits = self.output(hidden_states) # Compute logits

        # 4. Construct Output Object
        # Return tuple if return_dict is False
        if not return_dict:
            output = (logits,)
            if use_cache:
                output += (present_key_values,)
            if output_hidden_states:
                output += (all_hidden_states,)
            # Add aux_loss to tuple if MoE is used
            if self.params.use_moe:
                output += (total_aux_loss,)
            # if output_attentions: # Add attentions if implemented
            #     output += (all_self_attns,)
            return output

        # Return CausalLMOutputWithPast dictionary if return_dict is True
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=present_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns, # Add attentions if implemented
            # Add aux_loss to the output dictionary if MoE is used.
            # Using 'loss' field is one convention, or add a custom 'aux_loss' key.
            # Let's add a custom key for clarity.
            # loss=total_aux_loss if self.params.use_moe else None # Alternative: Use 'loss' field
            # aux_loss=total_aux_loss if self.params.use_moe else None # Custom key
        )


# --- Generation methods ---
# These methods generally don't need direct FP8 context management,
# as inference often runs in a different precision (like BF16/FP16).
# The forward pass inside will respect the FP8 layers if the model was initialized with fp8_enabled=True.
# Ensure the forward call passes necessary arguments like use_cache and start_pos.

@torch.inference_mode()
def generate(self,
             input_ids: torch.Tensor,
             max_new_tokens: int = 1024,
             temperature: float = 0.75,
             top_p: Optional[float] = 0.90,
             eos_token_id: int = 2,
             pad_token_id: int = 0, # Assuming 0 is pad
             stream: bool = False,
             rp: float = 1.0, # Repetition penalty factor
             use_cache: bool = True,
             **kwargs): # Allow passing extra args like attention_mask if needed

    if stream:
        # Delegate to the streaming generator method
        # Yields tokens one by one
        yield from self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **kwargs)
        return # End generator

    # Non-streaming generation
    bsz = input_ids.size(0)
    # Find the length of the prompt, excluding padding tokens
    prompt_lengths = torch.sum(input_ids != pad_token_id, dim=1)
    min_prompt_len = prompt_lengths.min().item() if bsz > 0 else 0
    max_len = min_prompt_len + max_new_tokens

    # Ensure max_len doesn't exceed model's capacity
    if max_len > self.params.max_seq_len:
         print(f"Warning: Requested max length {max_len} exceeds model max sequence length {self.params.max_seq_len}. Truncating.")
         max_len = self.params.max_seq_len

    # Prepare model inputs for the first step
    model_kwargs = {"use_cache": use_cache, "return_dict": True} # Ensure dict output
    if use_cache:
        model_kwargs["past_key_values"] = None

    # Keep track of generated tokens, starting with the prompt
    generated_ids = input_ids

    # Generation loop
    for step in range(max_new_tokens):
        current_seq_len = generated_ids.shape[1]
        # Prepare inputs for this step
        if use_cache and model_kwargs["past_key_values"] is not None:
            # Use only the last token if cache is enabled and populated
            current_input_ids = generated_ids[:, -1:]
            start_pos = current_seq_len - 1
        else:
            # Use the full sequence for the first step or if cache is disabled
            current_input_ids = generated_ids
            start_pos = 0

        # Forward pass
        outputs = self(
            input_ids=current_input_ids,
            start_pos=start_pos,
            **model_kwargs
        )

        logits = outputs.logits[:, -1, :] # Get logits for the very last token position

        # Apply repetition penalty
        if rp != 1.0 and current_seq_len > 0:
             # Apply penalty only to tokens already generated in each sequence
             for i in range(bsz):
                 # Consider only non-padding tokens for penalty
                 tokens_to_penalize = torch.unique(generated_ids[i][:current_seq_len])
                 # Remove pad token if present
                 tokens_to_penalize = tokens_to_penalize[tokens_to_penalize != pad_token_id]
                 logits[i, tokens_to_penalize] /= rp

        # Apply temperature scaling
        logits.div_(temperature + 1e-9) # Add epsilon for stability

        # Apply top-p (nucleus) sampling
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Create mask for tokens to remove (those outside the nucleus)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the mask: keep the first token whose cumulative probability exceeds top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # Scatter mask back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("inf") # Set logits of removed tokens to -inf

        # Sample the next token from the modified distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append the generated token to the sequence
        generated_ids = torch.cat((generated_ids, next_token), dim=1)

        # Update past_key_values for the next iteration if using cache
        if use_cache:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # Check stopping conditions
        # 1. If all sequences in the batch ended with EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
             break
        # 2. If maximum length is reached
        if generated_ids.shape[1] >= max_len:
             break

    # Return the generated sequences (potentially including prompt and padding)
    return generated_ids


# Streaming generation method (yields tokens one by one)
def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **kwargs):
    bsz = input_ids.size(0)
    if bsz > 1:
        raise NotImplementedError("Streaming generation currently supports batch size 1 only.")

    start_pos = input_ids.shape[1] # Position after the prompt
    past_kvs = None
    current_generated_ids = torch.tensor([], dtype=input_ids.dtype, device=input_ids.device)

    # Process prompt first if use_cache is True to populate KV cache
    if use_cache:
         prompt_outputs = self(input_ids, use_cache=True, return_dict=True, **kwargs)
         past_kvs = prompt_outputs.past_key_values

    last_token_ids = input_ids # Start with the prompt for the loop input

    for i in range(max_new_tokens):
        # Determine inputs for this step
        if i == 0 and not use_cache: # First step without cache: use full prompt
            current_input_ids = last_token_ids
            current_start_pos = 0
        else: # Subsequent steps or first step with cache: use only the last token
            current_input_ids = last_token_ids[:, -1:]
            current_start_pos = start_pos + i -1 if i > 0 else start_pos -1 # Position of the token being generated

        # Forward pass
        outputs = self(
            current_input_ids,
            past_key_values=past_kvs,
            use_cache=use_cache,
            start_pos=current_start_pos,
            return_dict=True,
            **kwargs
        )
        logits = outputs.logits[:, -1, :] # Logits for the last token

        # Update KV cache for next iteration
        if use_cache:
            past_kvs = outputs.past_key_values

        # Apply repetition penalty (only on generated tokens)
        if rp != 1.0 and current_generated_ids.shape[1] > 0:
             unique_tokens = torch.unique(current_generated_ids)
             logits[:, unique_tokens] /= rp

        # Apply temperature
        logits.div_(temperature + 1e-9)

        # Apply top-p sampling
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("inf")

        # Sample next token
        probs = F.softmax(logits, dim=-1)
        input_ids_next = torch.multinomial(probs, num_samples=1)

        # Update the sequence of purely generated tokens
        current_generated_ids = torch.cat((current_generated_ids, input_ids_next), dim=1)
        # Update the input for the next iteration
        last_token_ids = input_ids_next

        # Yield the newly generated token
        yield input_ids_next

        # Check for EOS
        if eos_token_id is not None and input_ids_next.item() == eos_token_id:
            break