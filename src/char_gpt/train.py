import os
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

from src.char_gpt.model import GPT


# =============================================================================
# Config
# =============================================================================

@dataclass
class GPTConfig:
    # model
    n_embed:    int   = 192
    n_head:     int   = 6
    n_layers:   int   = 6
    block_size: int   = 256
    dropout:    float = 0.2
    # training
    max_steps:  int   = 3000
    batch_size: int   = 64
    eval_iters: int   = 30
    lr:         float = 3e-4


# =============================================================================
# Data
# =============================================================================

@dataclass
class Dataset:
    train: torch.Tensor
    val:   torch.Tensor
    itos:  dict
    vocab_size: int


def load_dataset(path: str | Path):
    with open(path, 'r', encoding='UTF-8') as f:
        text = f.read()

    chars = sorted(set(text))
    stoi  = {s: i for i, s in enumerate(chars)}
    itos  = {i: s for s, i in stoi.items()}
    data  = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n     = int(0.9 * len(data))
    return Dataset(data[:n], data[n:], itos, len(chars))


def get_batch(ds, split, cfg: GPTConfig, device):
    data = ds.train if split == "train" else ds.val
    ix   = torch.randint(0, len(data) - cfg.block_size, (cfg.batch_size,))
    xb   = torch.stack([data[i:i + cfg.block_size]     for i in ix])
    yb   = torch.stack([data[i+1:i + cfg.block_size+1] for i in ix])
    return xb.to(device), yb.to(device)


# =============================================================================
# Eval
# =============================================================================

@torch.no_grad()
def estimate_loss(model, ds, cfg: GPTConfig, device):
    model.eval()
    results = {}
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(ds, split, cfg, device)
            _, loss = model(xb, yb)
            losses[k] = loss.detach()
        results[split] = losses.mean().item()
    model.train()
    return results["train"], results["val"]


# =============================================================================
# Training
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=6000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg    = GPTConfig(max_steps=args.max_steps)

    os.makedirs("models", exist_ok=True)
    model_path = Path(__file__).parents[2] / "data/tiny_shakespeare.txt"
    ds    = load_dataset(model_path)
    model = GPT(
        vocab_size=ds.vocab_size,
        n_embed=cfg.n_embed,
        n_head=cfg.n_head,
        n_layers=cfg.n_layers,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print("=" * 45)
    print(f"  GPT Model")
    print("=" * 45)
    print(f"  vocab_size : {ds.vocab_size}")
    print(f"  n_embed    : {cfg.n_embed}")
    print(f"  n_head     : {cfg.n_head}")
    print(f"  n_layers   : {cfg.n_layers}")
    print(f"  block_size : {cfg.block_size}")
    print(f"  dropout    : {cfg.dropout}")
    print(f"  device     : {device}")
    print("-" * 45)
    print(f"  parameters : {param_count:,}")
    print("=" * 45)
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_steps, eta_min=3e-5)

    pbar = tqdm(range(cfg.max_steps), desc="Training", unit="step")

    for i in pbar:
        xb, yb = get_batch(ds, "train", cfg, device)
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 300 == 0 or i == cfg.max_steps - 1:
            train_loss, val_loss = estimate_loss(model, ds, cfg, device)
            tqdm.write(f"step {i:5d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if i > 0 and i % 1000 == 0:
            torch.save({
                "model": model.state_dict(),
                "itos":  ds.itos,
                "model_config": dict(
                    vocab_size=ds.vocab_size, n_embed=cfg.n_embed, n_head=cfg.n_head,
                    n_layers=cfg.n_layers, block_size=cfg.block_size, dropout=cfg.dropout,
                ),
            }, f"models/gpt_{i}.pt")

    torch.save({
        "model": model.state_dict(),
        "itos":  ds.itos,
        "model_config": dict(
            vocab_size=ds.vocab_size, n_embed=cfg.n_embed, n_head=cfg.n_head,
            n_layers=cfg.n_layers, block_size=cfg.block_size, dropout=cfg.dropout,
        ),
    }, "models/gpt_final.pt")
