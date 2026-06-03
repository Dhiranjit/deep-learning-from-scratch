import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_embed, embed_dim))
    
    def forward(self, idx): # Index: (B, T)
        return self.W[idx] # (B,T,C)
    

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.lookup_table = Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.lookup_table(idx) # (B,T,C)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss
    
    def generate(self, idx, max_tokens: int):
        for _ in range(max_tokens):
            logits, _ = self(idx) # (B,T,C)
            logits = logits[:, -1, :] # Select last timestep (T) -> (B,C)
            probs = F.softmax(logits, dim=1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T) + (B, 1) -> (B,T+1)
            yield idx_next



