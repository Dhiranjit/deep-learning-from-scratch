import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Self-Attention
# =============================================================================

class Head(nn.Module):
    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        scores = q @ k.transpose(-1, -2) * (k.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #type: ignore
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return scores @ self.value(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        head_size    = n_embed // n_head
        self.heads   = nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj    = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        self.sa   = MultiHeadAttention(n_embed, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1  = nn.LayerNorm(n_embed)
        self.ln2  = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# =============================================================================
# Model
# =============================================================================

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layers, block_size, dropout):
        super().__init__()
        self.block_size         = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks  = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        out = []
        context = idx[:, -self.block_size:]
        for _ in range(max_new_tokens):
            logits, _ = self(context)
            probs     = F.softmax(logits[:, -1, :], dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            out.append(idx_next)
            context = torch.cat((context, idx_next), dim=1)[:, -self.block_size:]
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens):
        context = idx[:, -self.block_size:]
        for _ in range(max_new_tokens):
            logits, _ = self(context)
            probs     = F.softmax(logits[:, -1, :], dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            context   = torch.cat((context, idx_next), dim=1)[:, -self.block_size:]
            yield idx_next
