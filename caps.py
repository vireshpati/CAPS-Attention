import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from types import SimpleNamespace

class SimpleRoPE(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, pos_scale: float = 1.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.pos_scale = float(pos_scale)

    @staticmethod
    def _rot(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    def _cos_sin(self, L: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(L, device=device, dtype=self.inv_freq.dtype) * self.pos_scale  # (L,)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)                      # (L, d/2)
        emb = torch.cat((freqs, freqs), dim=-1).to(dtype=dtype, device=device) # (L, d)

        cos = emb.cos()[None, :, None, :]  # (1, L, 1, d)
        sin = emb.sin()[None, :, None, :]
        return cos, sin

    def apply_rotary(self, x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
        B, L, h, d = x.shape
        cos, sin = self._cos_sin(L, device=x.device, dtype=x.dtype)
        if transpose:
            sin = -sin  # R^T = R(-θ)
        return x * cos + self._rot(x) * sin

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q_out = self.apply_rotary(q, transpose=False)
        k_out = self.apply_rotary(k, transpose=False)
        return q_out, k_out

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        From: https://github.com/meta-llama/llama/blob/main/llama/model.py
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @classmethod
    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)

    def forward(self, x):
        output = self.norm(x.float()).type_as(x)
        return output * self.weight

NORM = RMSNorm

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CAPSAttention(nn.Module):
    """CAPS attention mechanism.

    Combines:
    - Rotary Position Embedding (RoPE) for SO(2) rotations
    - SPD gating for exponential decay dynamics
    - Clock mechanism for input-dependent time scaling

    Two modes:
    - Multiplicative (use_mult_gate=True): Full block with residual + MLP (like old code)
    - Additive (use_mult_gate=False): Two Q/K pairs combined as qk1 + qk2, no internal MLP
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0,
                 attn_type: str = 'softmax', use_time_decay_gate: bool = False,
                 spd_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout_rate = dropout
        self.attn_type = attn_type
        self.use_time_decay_gate = use_time_decay_gate

        assert d_model % n_head == 0
        self.d_head = d_model // n_head

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # SPD gating projection
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

        # Input projection for scan recurrence
        self.p_proj = nn.Linear(d_model, d_model, bias=False)

        # Clock mechanism
        self.clock = nn.Linear(d_model, n_head, bias=False)

        # Output projection
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.spd_dropout = nn.Dropout(spd_dropout) if spd_dropout > 0 else None

        self.rope = SimpleRoPE(dim=d_model // n_head, base=10000, pos_scale=1.0)

    def forward(self, x: torch.Tensor, use_mult_gate: bool = False, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, D]
            use_mult_gate: True for multiplicative mode, False for additive mode
        """
        return self._forward_add(x)

    def _forward_add(self, x: torch.Tensor) -> torch.Tensor:
        """Additive mode: Two Q/K pairs (p_exp only vs SPD only) combined as qk1 + qk2."""
        B, T, D = x.shape
        times = torch.arange(T, device=x.device)

        # Project Q, K, V
        q_out = self.W_q(x)
        k_out = self.W_k(x)
        v_out = self.W_v(x)

        # Apply time-decay gating if enabled
        # Reshape for multi-head
        q_out = q_out.view(B, T, self.n_head, -1)
        k_out = k_out.view(B, T, self.n_head, -1)
        v = v_out.view(B, T, self.n_head, -1)

        # Clock mechanism
        clock = F.softplus(self.clock(x)).view(B, T, self.n_head, -1) + 1e-6
        log_clock = torch.log(clock)

        # Normalize Q, K
        # q_out = F.normalize(q_out, dim=-1, p=2)
        # k_out = F.normalize(k_out, dim=-1, p=2)
        q_out, k_out = self.rope(q_out, k_out)

        # === First Q/K pair: p_exp scaling only (no SPD) ===
        q1 = q_out.clone()
        k1 = k_out.clone()

        p = self.p_proj(x).view(B, T, self.n_head, -1) + log_clock
        p_max = p.max(dim=1, keepdim=True).values
        p_exp = torch.exp(p - p_max)
        p_exp_cumsum = p_exp.cumsum(dim=1)
        q1 = q1 / (p_exp_cumsum + 1e-8)
        k1 = k1 * p_exp

        # === Second Q/K pair: SPD gating only (no p_exp) ===
        q2 = q_out.clone()
        k2 = k_out.clone()

        gj = -F.softplus(self.gate_proj(x)).view(B, T, self.n_head, -1) 
        gj = gj * clock
        gj_cumsum = gj.cumsum(dim=1).clip(-50, 40)
        gj_cumprod = torch.exp(gj_cumsum)

        q2 = q2 * gj_cumprod
        k2 = k2 / (gj_cumprod + 1e-8)

        clock_cumsum = clock.cumsum(dim=1)
        q3 = q_out / clock_cumsum
        k3 = k_out * clock

        # Attention computation with additive combination
        if self.attn_type == 'softmax':
            output = self._softmax_attention_additive(q1, k1, q2, k2, q3, k3, v)
        else:
            output = self._linear_attention_additive(q1, k1, q2, k2, q3, k3, v)

        # Output projection (no internal norm/residual/MLP for additive mode)
        output = output.reshape(B, T, -1)
        output = self.dropout(self.c_proj(output))

        return output

    def _softmax_attention_additive(self, q1, k1, q2, k2, q3, k3, v):
        """Additive attention: qk1 + qk2 combined before softmax."""

        q_cat = torch.cat([q1, q2, q3], dim=-1)  # head_dim -> 2*head_dim
        k_cat = torch.cat([k1, k2, k3], dim=-1)

        out = F.scaled_dot_product_attention(q_cat.transpose(1, 2), k_cat.transpose(1, 2), v.transpose(1, 2), 
                                             is_causal=False, dropout_p=self.dropout_rate if self.training else 0.0).transpose(1, 2).contiguous()
        return out


    def _linear_attention_additive(self, q1, k1, q2, k2, q3, k3, v, use_fla=False):
        if use_fla:
            from fla.ops.simple_gla import chunk_simple_gla
            o1, _ = chunk_simple_gla(q=q1, k=k1, v=v, output_final_state=False)
            o2, _ = chunk_simple_gla(q=q2, k=k2, v=v, output_final_state=False)
            o3, _ = chunk_simple_gla(q=q3, k=k3, v=v, output_final_state=False)
            out = o1 + o2 + o3
            return out
        else:
            # Original quadratic (faster for small T)
            q_cat = torch.cat([q1, q2, q3], dim=-1)
            k_cat = torch.cat([k1, k2, k3], dim=-1)
            qk = torch.einsum('blhd,bihd->bhli', q_cat, k_cat)
            qk = torch.tril(qk)
            out = torch.einsum('bhli,bihd->blhd', qk, v)
            return out

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layer = config.n_layer
        self.n_layer = n_layer
        self.predictor = config.predictor
        self.attn = nn.ModuleList([CAPSAttention(d_model=config.n_embd, n_head=config.n_head, dropout=config.dropout, attn_type='linear') for i in range(n_layer)])
        self.pns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.lns = nn.ModuleList([NORM(config.n_embd) for i in range(n_layer)])
        self.mlps = nn.ModuleList([MLP(config) for i in range(n_layer)])
        self.n_head = config.n_head
        self.d = config.n_embd // self.n_head

    def forward(self, x):
        for attn, pn, ln, mlp in zip(self.attn, self.pns, self.lns, self.mlps):
            x = x + attn(pn(x))
            x = x + mlp(ln(x))
        return x

def random_ratio_channel_dropout_vectorized(x, training=True):
    if not training:
        return x

    B, C, _ = x.shape
    device = x.device
    r = torch.rand(B, device=device)
    scores = torch.rand(B, C, device=device)
    threshold = r.unsqueeze(1)
    # mask: True = keep
    mask = (scores > threshold).float()
    # (B, C, 1)
    mask = mask.unsqueeze(-1)
    keep_ratio = mask.mean(dim=1, keepdim=True).clamp(min=1e-6)
    mask = mask / keep_ratio
    return x * mask

def nearest_power_of_two_to_sqrt(x: float) -> float:
    if x <= 0:
        raise ValueError("x must be positive")

    sqrt_x = math.sqrt(x)
    n = round(math.log2(sqrt_x))
    n = max(1, n)
    return 2 ** n

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.time_emb_dim = configs.time_emb_dim
        self.enc_in = configs.enc_in + configs.time_emb_dim
        self.number_of_targets = configs.number_of_targets if configs.number_of_targets else configs.enc_in
        n_series = self.number_of_targets
        self.n_series = n_series
        self.d_input = nearest_power_of_two_to_sqrt(self.enc_in) * 4 if not configs.d_model else configs.d_model       # int(np.sqrt(self.enc_in)) * 4

        # Use ARX Tokenization
        self.dropout = nn.Dropout(configs.dropout)
        self.n_head = self.d_input // 8 if not configs.n_heads else configs.n_heads
        self.d = self.d_input // self.n_head
        self.local_attention = getattr(configs, "local_attention", False)   # Disabled
        self.local_attention_window = getattr(configs, "local_attention_window", 8)

        self.seq_emb_dim = nearest_power_of_two_to_sqrt(self.enc_in) * 4
        self.channel_token_mixer_seq = nn.Linear(self.enc_in, self.d_input, bias=False)
        self.value_token_mixer = nn.Parameter(0.02*torch.randn(self.number_of_targets, 1, self.seq_emb_dim))

        hidden_dim = self.d_input + self.seq_emb_dim
        print(f'Current Dimension: {hidden_dim}')
        transformer_config = SimpleNamespace(n_layer=configs.e_layers, n_embd=hidden_dim, n_head=self.n_head, dropout=configs.dropout, 
                                             bias=False, max_len=self.seq_len+self.pred_len, predictor=configs.predictor, decay=True, enc_in=self.number_of_targets,
                                             local_attention=self.local_attention, local_attention_window=self.local_attention_window)
        self.temporal_processor = nn.Sequential(
                                                #NORM(hidden_dim),
                                                Transformer(transformer_config),
                                                #NORM(hidden_dim)
                                                )

        self.pos_emb = nn.Embedding(self.seq_len+self.pred_len, hidden_dim)
        self.channel_emb_2 = nn.Parameter(torch.zeros(1, self.number_of_targets, 1, hidden_dim))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * configs.e_layers))

        self.out_fill = nn.Linear(self.seq_len, self.pred_len, bias=False)

        self.random_channel_dropout = configs.random_channel_dropout

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.00)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # B C L

        # preprocessing
        mean = x[:, :, [-1]] # x[:, :, -self.pred_len:].mean(dim=2, keepdim=True) # # B C 1
        std = torch.ones(x.size(0), x.size(1), 1, device=x.device) # x.std(dim=2, keepdim=True) + 1e-8 # torch.ones_like(x.std(dim=2, keepdim=True)) #                    # B C 1
        x = (x - mean) / std                                       # B C L

        out_fill = self.out_fill(x)  
        x = torch.cat([x, out_fill], dim=2)  # B C L+pred_len
        if x_mark_enc is not None and x_mark_dec is not None and self.time_emb_dim:
            mark = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:]], dim=1).permute(0, 2, 1) # B C L
            mark = (mark - x_mark_enc.permute(0, 2, 1)[:, :, [-1]]) / (x_mark_enc.permute(0, 2, 1).std(dim=-1, keepdim=True) + 1e-8)  
            x = torch.cat([mark, x], dim=1)  # B C L


        x_dropout = random_ratio_channel_dropout_vectorized(x, self.training and self.random_channel_dropout)
        channel_token = self.channel_token_mixer_seq(x_dropout.permute(0, 2, 1))   # B L C -> B L d
        channel_token = channel_token.unsqueeze(1).expand(-1, self.number_of_targets, -1, -1)
        x = x[:, -self.number_of_targets:].unsqueeze(-1) # B C L 1
        x = torch.einsum('bcli,cid->bcld', x, self.value_token_mixer)   # B C L d
        x = torch.cat([channel_token, x], dim=-1)    # B C L 2d

        B, C, N, _ = x.shape
        pos = torch.arange(0, N, dtype=torch.long, device=x.device)
        pos_emb = self.pos_emb(pos)
        x = x + self.channel_emb_2 + pos_emb
        x = x.reshape(B * C, N, -1)                   # B*C L 2d    

        x = self.temporal_processor(x)              # B*C L 2d
        x = x[:, -self.pred_len:].reshape(B, C, self.pred_len, -1)   # B C pred_len 2d
        y = torch.einsum('bcld,cdi->bcli', x[:, :, :, -self.seq_emb_dim:], self.value_token_mixer.transpose(-2, -1)).squeeze(-1)  # B C L_P
        x = y

        # inverse processing
        x = (x*std[:, -self.number_of_targets:] + mean[:, -self.number_of_targets:]).permute(0, 2, 1)                  # B s Ct

        if self.training:
            return x, torch.tensor(0, device=x.device)
        else:
            return x