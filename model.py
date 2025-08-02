import tiktoken
import torch
from torch import nn

# torch.manual_seed(123)
tokenizer = tiktoken.get_encoding('gpt2')

# scale, shift, epsilon
class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x-mean) / torch.sqrt(var + self.eps)
        return self.scale * x_norm + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((x + 0.044715 * torch.pow(x, 3)) * (torch.sqrt(torch.tensor(2.0/torch.pi)))))

# x -> 4x -> x
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), GELU(), nn.Linear(4*embed_dim, embed_dim))

    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, ctx_length, n_heads, dropout, qkv_bias = False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads

        self.Wq = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias = qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(ctx_length, ctx_length), diagonal=1))

    def forward(self, x):
        batch_size, n_tokens, d_in = x.shape

        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)

        # split b, n_tokens, d_out -> b, n_tokens, n_heads, head_dim
        queries = queries.view(batch_size, n_tokens, self.n_heads, self.head_dim)
        keys = keys.view(batch_size, n_tokens, self.n_heads, self.head_dim)
        values = values.view(batch_size, n_tokens, self.n_heads, self.head_dim)

        # transpose b, n_tokens, n_heads, head_dim -> b, n_heads, n_tokens, head_dim
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # identical dims, imagine a transpose with the heads still separate
        attn_scores = queries @ keys.transpose(2, 3)

        # scale mask to input size and make it boolean
        mask_bool = (self.mask.bool()[:n_tokens, :n_tokens]).to(x.device)

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # -1 -> across cols
        attn_weights = torch.softmax(attn_scores/torch.sqrt(torch.tensor(self.d_out ** 0.5)), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # b, n_heads, n_tokens, head_dim -> b, n_tokens, n_heads, head_dim
        ctx_vec = (attn_weights @ values).transpose(1, 2)

        # b, n_tokens, n_heads, head_dim -> b, n_tokens, d_out
        ctx_vec = ctx_vec.contiguous().view(batch_size, n_tokens, self.d_out)
        # optional, can also return directly
        return self.out_proj(ctx_vec)

# ln -> attn -> dropout + shortcut -> ln -> ff -> dropout + shortcut
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = MultiHeadAttention(config["embed_dim"], config["embed_dim"], config["ctx_length"], n_heads=config["n_heads"], dropout=config["drop_rate"], qkv_bias=config["qkv_bias"])
        self.ff = FeedForward(config["embed_dim"])
        self.ln1 = LayerNorm(config["embed_dim"])
        self.ln2 = LayerNorm(config["embed_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.dropout(x) + shortcut
        shortcut = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout(x) + shortcut
        return x

#  dropout -> trfs -> norm
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embed = nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_embed = nn.Embedding(config['ctx_length'], config['embed_dim'])
        self.drop_embed = nn.Dropout(config['drop_rate'])

        self.trf_blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config['n_layers'])))

        self.final_norm = LayerNorm(config['embed_dim'])
        self.out_head = nn.Linear(config['embed_dim'], config['vocab_size'], bias=config["qkv_bias"])


    def forward(self, items):
        batch_size, seq_len = items.shape
        tok_embeds = self.tok_embed(items)
        pos_embeds = self.pos_embed(torch.arange(seq_len, device=items.device))

        x = tok_embeds + pos_embeds
        x = self.drop_embed(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
