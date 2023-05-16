import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from local_attention import LocalAttention

# helper function

def exists(val):
    return val is not None

# multi-head attention

class LocalMHA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        prenorm = False,
        exact_windowsize = False,
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.attn_fn = LocalAttention(
            dim = dim_head,
            window_size = window_size,
            causal = causal,
            autopad = True,
            exact_windowsize = exact_windowsize,
            **kwargs
        )

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask = None):
        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        out = self.attn_fn(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# main transformer class

# main transformer class

class LocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim=1,
        max_seq_len,
        dim,
        depth,
        num_classes,
        causal = False,
        local_attn_window_size = 1,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        cls_method = 'mean_pooling', # could also be 'cls_token' or 'concat'

        **kwargs
    ):
        super().__init__()

        self.lin_emb = nn.Linear(input_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.max_seq_len = max_seq_len
        self.cls_method = cls_method


        self.attn_blocks = nn.ModuleList([
            LocalMHA(
                dim = dim,
                window_size = local_attn_window_size,
                dim_head = dim_head,
                heads = heads,
                dropout = attn_dropout,
                causal = causal,
                exact_windowsize = False,
                **kwargs
            )
            for _ in range(depth)
        ])

        self.ff_blocks = nn.ModuleList([
            FeedForward(dim, mult = ff_mult, dropout = ff_dropout)
            for _ in range(depth)
        ])
        if self.cls_method == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.out_proj = nn.Linear(dim, num_classes)  # add this linear layer
        elif self.cls_method == 'concat':
            self.out_proj = nn.Linear(max_seq_len * dim, num_classes)  # add this linear layer
        elif self.cls_method == 'max_pooling':
            self.out_proj = nn.Linear(dim, num_classes)
        else: # mean pooling
            self.out_proj = nn.Linear(dim, num_classes)


    def forward(self, x, mask=None):
        # swap the sequence length and hidden dimension
        x = rearrange(x, 'b d n -> b n d')



        x = self.lin_emb(x)
        if self.cls_method == 'cls_token':
            # get the batch size
            b, n, _ = x.shape
            # Add CLS token and positional encoding
            cls_token = self.cls_token.repeat(b, 1, 1)
            # place the cls token in the middle of the sequence
            x = torch.cat((x[:, :n // 2], cls_token, x[:, n // 2:]), dim=1)
            # get the index of the cls token
            cls_index = n // 2

        for attn, ff in zip(self.attn_blocks, self.ff_blocks):
            x = attn(x, mask = mask) + x
            # x = attn(x, mask=mask)
            x = x + ff(x)

        if self.cls_method == 'cls_token':
            # extract the cls token
            x = x[:, cls_index, :]
        elif self.cls_method == 'mean_pooling':
            # # dimension of x is (batch_size, max_seq_len, dim)
            # we want to average over the sequence dimension
            x = x.mean(dim=1)
            x = x.squeeze(1)
        elif self.cls_method == 'max_pooling':
            x = x.max(dim=1)
            x = x.squeeze(1)
        else: # self.cls_method == 'concat':
            # concatenate the sequence dimension with the hidden dimension
            x = rearrange(x, 'b n d -> b (n d)')

        return self.out_proj(x)

