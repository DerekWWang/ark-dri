import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence


class ClassificationDataset(Dataset):
    """
    Stores all (token_ids, label) pairs flat.
    Also exposes per-bucket index lists (split by class) for the sampler.

    label 0 = true string
    label 1 = corrupted string
    """

    def __init__(self, true_tokenized: list[list[int]], corrupted_tokenized: list[list[int]],
                 bucket_size: int = 50):
        # flat list of (ids, label)
        self.samples: list[tuple[list[int], int]] = (
            [(ids, 0) for ids in true_tokenized] +
            [(ids, 1) for ids in corrupted_tokenized]
        )

        # bucket_size-wide length buckets → {bucket_ceil: {"true": [...idx], "corrupt": [...idx]}}
        max_len = max(len(ids) for ids, _ in self.samples)
        bounds = list(range(bucket_size, max_len + bucket_size, bucket_size))
        self.buckets: dict[int, dict[str, list[int]]] = {
            b: {"true": [], "corrupt": []} for b in bounds
        }
        for i, (ids, label) in enumerate(self.samples):
            for b in bounds:
                if len(ids) <= b:
                    key = "true" if label == 0 else "corrupt"
                    self.buckets[b][key].append(i)
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, label = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), label


class BucketBatchSampler(Sampler):
    """
    Each batch: B/2 true + B/2 corrupted, all from the same length bucket.
    Iterates over all populated buckets, shuffling within each.
    """

    def __init__(self, dataset: ClassificationDataset, batch_size: int, shuffle: bool = True):
        assert batch_size % 2 == 0, "batch_size must be even"
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.half = batch_size // 2

    def _make_batches(self):
        batches = []
        for b, split in self.dataset.buckets.items():
            true_idx = split["true"][:]
            corrupt_idx = split["corrupt"][:]
            if self.shuffle:
                random.shuffle(true_idx)
                random.shuffle(corrupt_idx)
            # pair up half-batches from each class
            n = min(len(true_idx) // self.half, len(corrupt_idx) // self.half)
            for i in range(n):
                batch = (
                    true_idx[i * self.half : (i + 1) * self.half] +
                    corrupt_idx[i * self.half : (i + 1) * self.half]
                )
                if self.shuffle:
                    random.shuffle(batch)
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._make_batches()

    def __len__(self):
        total = 0
        for split in self.dataset.buckets.values():
            n = min(len(split["true"]) // self.half, len(split["corrupt"]) // self.half)
            total += n
        return total


def collate_fn(batch, pad_id: int):
    """Pad to the longest sequence in the batch and build a boolean mask."""
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)   # (B, T)
    mask = (padded != pad_id).long()                                       # (B, T)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, mask, labels


