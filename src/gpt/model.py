from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F



class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        assert n_embed % n_head == 0

        self.n_embed   = n_embed
        self.n_head    = n_head
        self.head_size = n_embed // n_head

        # Key, query, value projections for all heads, combined in a single batch
        # (C, hs) for each of k, q, v in a single head -> (C, hs * nh) -> (C, C) for each k, q, v
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)

        # Output projection (C, C)
        self.c_proj = nn.Linear(n_embed, n_embed, bias=False)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal Mask
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    
    def forward(self, x):
        B, T, C = x.shape

        # (B, T, C) -> (B, T, 3 * C)
        qkv = self.c_attn(x) 

        # Shapes: # (B, T, C)
        q, k, v = qkv.split(C, dim=2) 

        # Shapes: (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Causal Attention Math
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # type: ignore
        scores = F.softmax(scores, dim=-1)

        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = scores @ v

        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection (B, T, C) @ (C, C) -> (B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class FeedFoward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        # Layer Normalization
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

        # Core sub-layers
        self.MHA  = CausalSelfAttention(n_embed, n_head, block_size, dropout)
        self.ffwd = FeedFoward(n_embed, dropout)

    def forward(self, x):
        x = x + self.MHA(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size, dropout):
        super().__init__()
        
        # Token + Positional emdeddings
        self.token_embedding  = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding    = nn.Embedding(block_size, n_embed)
        self.drop             = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList(
            [Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        
        # Final LayerNorm and Language Model Head
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # (B, T) -> (B, T, C)
        tok_emb = self.token_embedding(idx)
        # (T,) -> (T, C)
        pos_emb = self.pos_embedding(torch.arange(0, T, device=idx.device))
        
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            return logits, None 
        
        # Flatten Time + Batch for cross entropy
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        return logits, loss

        







