import sys
import torch
import argparse
from src.char_gpt.model import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model",  type=str, default="models/gpt_final.pt")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model, weights_only=False, map_location=device)

    itos  = checkpoint["itos"]
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    model = GPT(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.stream:
        for idx_next in model.generate_stream(idx, args.tokens):
            sys.stdout.write(itos[idx_next.item()])
            sys.stdout.flush()
        print()
    else:
        output = model.generate(idx, args.tokens)
        print(''.join(itos[i] for i in output[0].tolist()))

if __name__ == "__main__":
    main()
