"""
Linear attention in Based.
https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py
"""
import math

import torch
import torch.nn as nn
from einops import rearrange
try:
    from fla.ops.triton.rebased_fast import parallel_rebased
except ImportError:
    print("FLA is not available")
    parallel_rebased = None
from torch.nn import LayerNorm


class Scaler(nn.Module):
    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.bias = bias
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, hiddens):
        return hiddens * self.gamma + self.beta if self.bias else hiddens * self.gamma


class Normalizer(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        return (x - mean) / (std + self.eps)



class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


def init_feature_map(feature_map: str = 'none', **kwargs: any):
    """
    Initialize query and key mapping for linear attention
    """
    if feature_map in [None, 'none', 'identity']:
        return FeatureMap(**kwargs)
    # Taylor series approximations to exp(x)
    elif feature_map == 'taylor_exp':
        return TaylorExp(**kwargs)
    else:
        raise NotImplementedError(
    f'Sorry "{feature_map}" feature map not implemented.')


class FeatureMap(nn.Module):
    """
    Parent feature map; default is identity function
    """

    def __init__(self,
                 input_dim: int,
                 temp: int = None,
                 head_dim_idx: int = -1,
                 eps: float = 1e-12,
                 **kwargs: any):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim_idx = head_dim_idx
        self.temp = 1. if temp is None else temp
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return x


class TaylorExp(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """

    def __init__(self, input_dim: int, **kwargs: any):
        super().__init__(input_dim, **kwargs)
        self.rd = math.sqrt(self.input_dim)
        self.rrd = math.sqrt(self.rd)

    # Running these in parallel
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)
              ).flatten(start_dim=-2)
        return x2 / self.rd


class ReBased(nn.Module):
    def __init__(
            self,
            d_model: int,
            l_max: int = 2048,
            feature_dim: int = 16,
            num_key_value_heads: int = 12,
            num_heads: int = 12,
            feature_name: str = "taylor_exp",
            eps: float = 1e-12,
            causal: bool = True,
            mode: str = "parallel",
            use_beta: bool = True,
            use_gamma: bool = True,
            normalize: bool = True,
            layer_idx: int = None
    ):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.mode = mode
        assert self.mode in ["fused_chunk", "parallel"]

        # linear attention
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_key_value_heads
        self.causal = causal
        feature_map_kwargs = {
            'input_dim': self.feature_dim,
            'head_dim_idx': -1,
            'temp': 1.,
            'eps': 1e-12
        }
        self.feature_map = init_feature_map(
            feature_map=self.feature_name, **feature_map_kwargs)
        self.proj_q = nn.Linear(
            self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(
            self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(
            self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.proj_o = nn.Linear(
            self.num_heads * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Identity()
        self.beta, self.gamma, self.normalize = use_beta, use_gamma, normalize
        if use_beta and use_gamma and normalize:
            self.ln_q = LayerNorm(self.feature_dim * self.num_heads)
            self.ln_k = LayerNorm(self.feature_dim * self.num_heads)
        elif normalize:
            print("Using Normalizer")
            self.ln_q = Normalizer()
            self.ln_k = Normalizer()
        elif use_gamma and use_beta:
            print("Using Scaler with bias")
            self.ln_q = Scaler(self.feature_dim * self.num_heads)
            self.ln_k = Scaler(self.feature_dim * self.num_heads)
        elif use_gamma:
            print("Using Scaler without bias")
            self.ln_q = Scaler(self.feature_dim * self.num_heads, bias=False)
            self.ln_k = Scaler(self.feature_dim * self.num_heads, bias=False)
            
        if parallel_rebased is None:
            self.feature_map = TaylorExp(self.feature_dim)

        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        if parallel_rebased is None:
            return self.forward_reference(hidden_states)
        mode = self.mode
        b, l, _ = hidden_states.size()
        q, k, v = self.proj_q(hidden_states), self.proj_k(
            hidden_states), self.proj_v(hidden_states)
        if self.beta or self.gamma or self.normalize:
            q, k = self.ln_q(q), self.ln_k(k)

        q, k, v = map(lambda x: rearrange(
            x, "b l (h d) -> b h l d", h=self.num_heads), [q, k, v])
        if mode == "fused_chunk":
            assert q.shape[-1] <= 16
            #o = fused_chunk_based(q, k, v, True, True)
        elif mode == 'parallel':
            assert q.shape[-1] <= 128
            o = parallel_rebased(q, k, v, self.eps, True, True)
        o = rearrange(o, "b h l d -> b l (h d)")
        o = self.proj_o(o)
        o = self.dropout(o)
        return o

    def forward_reference(self, hidden_states: torch.Tensor, filters: torch.Tensor = None, *args, **kwargs):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        # hidden_states = hidden_states.transpose(1, 2)
        b, l, _ = hidden_states.size()
        q, k, v = self.proj_q(hidden_states), self.proj_k(
            hidden_states), self.proj_v(hidden_states)
        if self.beta or self.gamma or self.normalize:
            q, k = self.ln_q(q), self.ln_k(k)
        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads,
                   self.feature_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads,
                   self.head_dim).transpose(1, 2)

        # Linear attention
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

        # Compute attention
        if self.causal:
            y = ((q * (k * v).cumsum(2)).sum(-1) / ((q * k.cumsum(2)).sum(-1) + self.eps))
        else:
            y = ((q * (k * v).sum(2, True)).sum(-1) / ((q * k.sum(2, True)).sum(-1) + self.eps))
        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.proj_o(y.to(hidden_states.dtype))
        y = self.dropout(y)
        return y.to(hidden_states.dtype)


if __name__ == '__main__':
    batch = 4
    seq_len = 1024
    d_model = 1024
    dtype = torch.float32
    x = torch.randn(batch, seq_len, d_model).to(
        dtype).cuda().requires_grad_(True)
    dy = torch.randn(batch, seq_len, d_model).to(
        dtype).cuda()
    model = ReBased(d_model=d_model).to(dtype).cuda()
    y = model(x)
    y.backward(dy, retain_graph=True)
    x_grad, x.grad = x.grad, None
    proj_q_grad, model.proj_q.weight.grad = model.proj_q.weight.grad, None
    proj_k_grad, model.proj_k.weight.grad = model.proj_k.weight.grad, None
    proj_v_grad, model.proj_v.weight.grad = model.proj_v.weight.grad, None
    x.requires_grad_(True)
    y2 = model.forward_reference(x)
    y2.backward(dy)
    print((y - y2).abs().max().item())
    # assert y.allclose(y2, 0, 1e-4)
    print((x_grad - x.grad).abs().max().item())
    # assert x_grad.allclose(x.grad, 0, 1e-4)

    print((proj_q_grad - model.proj_q.weight.grad).abs().max().item())
    print((proj_k_grad - model.proj_k.weight.grad).abs().max().item())
    print((proj_v_grad - model.proj_v.weight.grad).abs().max().item())

    print("All good with rebased fast!")