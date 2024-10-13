from typing import Optional, List, Sized

from torch.utils.data import RandomSampler, Sampler
import random


class GroupRandomSampler(RandomSampler):
    # Only shuffle inside a dataset
    def __init__(self, data_source: Sized, lengths: Optional[List[int]]):
        super().__init__(data_source=data_source)
        self.lengths = lengths
        self.start_idx = [0]
        for length in self.lengths:
            self.start_idx.append(self.start_idx[-1] + length)
        self.samplers = [RandomSampler(list(range(length))) for length in lengths]
    
    def __iter__(self):
        for i, sampler in enumerate(self.samplers):
            for idx in sampler:
                yield self.start_idx[i] + idx


class GlobalGroupRandomSampler(Sampler):
    """
    Ensure all the data within a global batch are from the same group.
    For those groups which can not be divided by global_batchsize, we over sample them  
    """
    def __init__(self, global_batchsize: int, lengths: Optional[List[int]]):
        """
        Initialize the sampler.

        Args:
        - global_batchsize (int): The size of the global batch.
        - lengths (Optional[List[int]]): A list containing the lengths of each group.
        """
        self.global_batchsize = global_batchsize
        self.lengths = lengths
        self.start_idx = [0]
        for length in self.lengths:
            self.start_idx.append(self.start_idx[-1] + length)
        self.indices = []
        self._prepare_indices()
        random.shuffle(self.indices)

    def _prepare_indices(self):
        """
        Prepare the list of indices to sample from, ensuring that each batch contains indices from the same group.
        Oversample groups that do not fit the global batch size perfectly.
        """
        if self.lengths is None:
            raise ValueError("Lengths must be provided.")

        for group_idx, length in enumerate(self.lengths):
            group_indices = list(range(length))
            group_indices = [n + self.start_idx[group_idx] for n in group_indices]
            random.shuffle(group_indices)  # Shuffle within the group for randomness

            # Determine the number of batches needed for this group
            num_batches = (length + self.global_batchsize - 1) // self.global_batchsize

            for _ in range(num_batches):
                batch_indices = group_indices[:self.global_batchsize]

                if len(batch_indices) < self.global_batchsize:
                    # If the batch is smaller than the global batch size, oversample
                    batch_indices += random.choices(group_indices, k=self.global_batchsize - len(batch_indices))

                self.indices.append(batch_indices)
                group_indices = group_indices[self.global_batchsize:]

    def __iter__(self):
        # Flatten the list of batches into a single list of indices
        flat_indices = [idx for batch in self.indices for idx in batch]
        # random.shuffle(flat_indices)  # Shuffle the entire list for randomness
        return iter(flat_indices)

    def __len__(self):
        # Return the total number of indices
        return sum(len(batch) for batch in self.indices)

if __name__ == "__main__":
    sampler = GlobalGroupRandomSampler(6, [4,5,8])
    print(len(sampler))
    for i, idx in enumerate(sampler):
        print(i, ": ", idx)