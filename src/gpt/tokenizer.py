from collections import Counter
from pathlib import Path
from tqdm import tqdm



# ===========================================================================
# Helper
# ===========================================================================


def count_pairs(ids):
    """Count adjacent-pair frequencies. Returns a Counter."""
    pairs = Counter(zip(ids, ids[1:]))

    return pairs


def merge(ids, pair, new_id):
    """Replace all occurence of a `pair` with `new_id`"""
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1                      # last token is now appended here only.
    return result


# ===========================================================================
# Tokenizer
# ===========================================================================


class BPETokenizer:

    def __init__(self) -> None:
        self.merges = {}
        self.vocab = self._build_vocab()

    
    def _build_vocab(self):
        """Rebuild vocab from base bytes + merges"""
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), new_id in self.merges.items():
            vocab[new_id] = vocab[p0] + vocab[p1]
        return vocab


    def train(self, text: str, vocab_size, verbose=False):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        merges = {}

        original_len = len(ids)

        iterator = tqdm(range(vocab_size - 256), desc="BPE Training", unit="merges", disable=not verbose)

        for i in iterator:
            counts = count_pairs(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get) # type: ignore
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            merges[pair] = new_id

            if i % 10 == 0: 
                iterator.set_postfix(tokens=len(ids))
        
        merged_len = len(ids)
        
        print(f"Compression Ratio: {original_len / merged_len:.2f}X")

        self.merges = merges
        self.vocab = self._build_vocab()
    

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        
        while len(ids) >= 2:
            counts = count_pairs(ids)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        
        return ids
    

    def decode(self, ids):
        text = b"".join([self.vocab[idx] for idx in ids]).decode("utf-8", errors="replace")
        return text
    

    def show_merges(self):
        for (p0, p1), new_id in self.merges.items():
            s0  = self.vocab[p0].decode("utf-8", errors="replace")
            s1  = self.vocab[p1].decode("utf-8", errors="replace")
            out = self.vocab[new_id].decode("utf-8", errors="replace")
            print(f"{new_id}: {s0!r} + {s1!r} -> {out!r}")
    
    
    def save(self, path):
        with open(path, "w") as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
    

    def load(self, path):
        merges = {}
        with open(path, "r") as f:
            for i, line, in enumerate(f):
                p0, p1 = map(int, line.split())
                merges[(p0, p1)] = 256 + i
        
        self.merges = merges
        self.vocab = self._build_vocab()
        return self


    

    
         
    

    

