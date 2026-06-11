import numpy as np

from pathlib import Path
from src.gpt.tokenizer import BPETokenizer

ROOT       = Path(__file__).resolve().parents[2]
vocab_size = 1200


def main():

    data_path  = ROOT / "data/tiny_shakespeare.txt"
    out_dir    = ROOT / "data/tiny_shakespeare"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPETokenizer()

    tokenizer.train(text, vocab_size, verbose=True)
    tokenizer.save(out_dir / "merges.txt")

    print("Training done!!! Encoding data...")

    ids =  tokenizer.encode(text)
    ids = np.array(ids, dtype=np.uint16)
    ids.tofile(out_dir / "train.bin")

    print(f"Encoding done!!! train.bin saved to {out_dir}")


if __name__ == "__main__":
    main()