"""
1. BPETokenizer.train(texts, vocab_size)  — learn merges from a corpus
2. tokenizer.encode(text)                 — text → list[int]
3. tokenizer.decode(ids)                  — list[int] → text
4. tokenizer.save(path) / .load(path)     — persist to JSON
"""

import re
import json
from collections import Counter, defaultdict


_PRETOK_RE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+""")

def _pretokenize(text: str) -> list[str]:
    return _PRETOK_RE.findall(text)


def _word_to_chars(word: str) -> tuple[str, ...]:
    """reepresent a word as a tuple of UTF-8 bytes (expressed as latin-1 chars).
    This guarantees every token is in the 256-entry seed vocab."""
    return tuple(bytes([b]).decode("latin-1") for b in word.encode("utf-8"))

class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[str, str], int] = {}
        self.vocab: dict[str, int] = {}
        self._id2tok: list[str] = []

    def train(self, texts: list[str], vocab_size: int = 1000, verbose: bool = False):
        """Learn BPE merges from *texts* until |vocab| == vocab_size."""
        assert vocab_size >= 256, "vocab_size must be ≥ 256"

        self._id2tok = [bytes([i]).decode("latin-1") for i in range(256)]
        self.vocab = {tok: i for i, tok in enumerate(self._id2tok)}

        word_freq: Counter[tuple[str, ...]] = Counter()
        for text in texts:
            for word in _pretokenize(text):
                word_freq[_word_to_chars(word)] += 1

        pair_counts = self._init_pair_counts(word_freq)

        num_merges = vocab_size - len(self.vocab)
        for step in range(num_merges):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best_pair] < 2:
                break 
            new_tok = best_pair[0] + best_pair[1]
            self.merges[best_pair] = step
            new_id = len(self._id2tok)
            self._id2tok.append(new_tok)
            self.vocab[new_tok] = new_id

            if verbose and step % 100 == 0:
                print(f"  merge {step:4d}: {best_pair!r} → {new_tok!r}  (freq={pair_counts[best_pair]})")

            word_freq, pair_counts = self._apply_merge(best_pair, new_tok, word_freq, pair_counts)

        if verbose:
            print(f"Training complete. vocab_size={len(self.vocab)}, merges={len(self.merges)}")

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in _pretokenize(text):
            ids.extend(self._encode_word(word))
        return ids

    def _encode_word(self, word: str) -> list[int]:
        """Apply learned merges to a single pre-token."""
        tokens = list(_word_to_chars(word))
        while len(tokens) > 1:
            best_idx, best_rank = -1, len(self.merges) + 1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_idx, best_rank = i, rank
            if best_idx == -1:
                break
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]
        return [self.vocab[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        raw = "".join(self._id2tok[i] for i in ids)
        return raw.encode("latin-1").decode("utf-8", errors="replace")

    def save(self, path: str):
        data = {
            "vocab": self._id2tok,
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok._id2tok = data["vocab"]
        tok.vocab = {t: i for i, t in enumerate(tok._id2tok)}
        tok.merges = {(a, b): rank for rank, (a, b) in enumerate(data["merges"])}
        return tok

    @staticmethod
    def _init_pair_counts(
        word_freq: Counter[tuple[str, ...]]
    ) -> defaultdict[tuple[str, str], int]:
        counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        for word, freq in word_freq.items():
            for a, b in zip(word, word[1:]):
                counts[(a, b)] += freq
        return counts

    @staticmethod
    def _apply_merge(
        pair: tuple[str, str],
        new_tok: str,
        word_freq: Counter[tuple[str, ...]],
        pair_counts: defaultdict[tuple[str, str], int],
    ) -> tuple[Counter, defaultdict]:
        """Apply one merge, updating both word_freq and pair_counts in place."""
        a, b = pair
        new_word_freq: Counter[tuple[str, ...]] = Counter()

        for word, freq in word_freq.items():
            if a not in word or b not in word:
                new_word_freq[word] += freq
                continue

            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    if new_word:
                        pair_counts[(new_word[-1], a)] -= freq
                        pair_counts[(new_word[-1], new_tok)] += freq
                    if i + 2 < len(word):
                        pair_counts[(b, word[i + 2])] -= freq
                        pair_counts[(new_tok, word[i + 2])] += freq
                    new_word.append(new_tok)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word_t = tuple(new_word)
            new_word_freq[new_word_t] += freq

        pair_counts[pair] = 0
        return new_word_freq, pair_counts

if __name__ == "__main__":
    true_strings: list[str] = []
    with open("challenge-data/train.txt", "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", maxsplit=1)
            if len(parts) < 2:
                continue
            true_strings.append(parts[0])

    print(f"Loaded {len(true_strings):,} true strings.")

    tokenizer = BPETokenizer()
    tokenizer.train(true_strings, vocab_size=4096, verbose=True)
    tokenizer.save("bpe_tokenizer.json")
    print("Saved to bpe_tokenizer.json")

    sample = true_strings[0]
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    assert decoded == sample, f"Round-trip failed!\n  original: {sample!r}\n  decoded:  {decoded!r}"
    print(f"\nRound-trip OK. Sample: {len(ids)} tokens for {len(sample)} chars.")
    print(f"  ids[:10] = {ids[:10]}")
