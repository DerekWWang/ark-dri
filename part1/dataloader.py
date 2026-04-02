import random
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence


class CausalLMDataset(Dataset):
    """Stores clean tokenized sequences for causal language modeling."""

    def __init__(
        self,
        tokenized_sequences: list[list[int]],
        bucket_size: int = 50,
        max_seq_len: int = 256,
    ):
        self.max_seq_len = max_seq_len

        self.samples: list[list[int]] = [
            ids for ids in tokenized_sequences if 2 <= len(ids) <= max_seq_len
        ]

        if not self.samples:
            raise ValueError("no samples remain after max_seq_len filtering")

        # bucket_size-wide length buckets
        max_len = max(len(ids) for ids in self.samples)
        bounds = list(range(bucket_size, max_len + bucket_size, bucket_size))
        self.buckets: dict[int, list[int]] = {b: [] for b in bounds}
        for i, ids in enumerate(self.samples):
            for b in bounds:
                if len(ids) <= b:
                    self.buckets[b].append(i)
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long)


class BucketBatchSampler(Sampler):
    def __init__(
        self,
        dataset: CausalLMDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def _make_batches(self):
        batches = []
        for _, idxs in self.dataset.buckets.items():
            bucket_indices = idxs[:]
            if self.shuffle:
                random.shuffle(bucket_indices)

            full_batch_count = len(bucket_indices) // self.batch_size
            for i in range(full_batch_count):
                start = i * self.batch_size
                batches.append(bucket_indices[start : start + self.batch_size])

            if not self.drop_last:
                rem = len(bucket_indices) % self.batch_size
                if rem:
                    batches.append(bucket_indices[-rem:])

        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._make_batches()

    def __len__(self):
        total = 0
        for idxs in self.dataset.buckets.values():
            total += len(idxs) // self.batch_size
            if not self.drop_last and len(idxs) % self.batch_size:
                total += 1
        return total

def collate_fn(batch, pad_id: int):
    padded = pad_sequence(batch, batch_first=True, padding_value=pad_id)   # (B, T)
    mask = (padded != pad_id).long()                                        # (B, T)
    return padded, mask


