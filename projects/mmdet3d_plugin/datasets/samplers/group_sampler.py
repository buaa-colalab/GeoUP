# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import math
import itertools
import copy
import torch.distributed as dist
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from .sampler import SAMPLER
import random
from IPython import embed


@SAMPLER.register_module()
class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.
    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, num_replicas = get_dist_info()

    if num_replicas == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

@SAMPLER.register_module()
class InfiniteGroupEachSampleInBatchSampler(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.
    Shuffling is only done for group order, not done within groups.
    """

    def __init__(self, 
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):

        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = sync_random_seed(seed)

        self.size = len(self.dataset)

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = samples_per_gpu * num_replicas
        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)}        

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator 
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx) 
            for local_sample_idx in range(self.batch_size)]
        
        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(self._infinite_group_indices(), 
                                    global_sample_idx, 
                                    None,
                                    self.global_batch_size)

    def __iter__(self):
        while True:
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    self.buffer_per_local_sample[local_sample_idx] = \
                        copy.deepcopy(
                            self.group_idx_to_sample_idxs[new_group_idx])

                curr_batch.append(self.buffer_per_local_sample[local_sample_idx].pop(0))
            
            yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        return self.size
        
    def set_epoch(self, epoch):
        self.epoch = epoch


@SAMPLER.register_module()
class MultiDatasetSeqSampler(Sampler):
    """Sequence-aware sampler for multi-dataset training.

    The sampler works on a ``ConcatDataset`` whose children expose sequence
    ids through ``dataset.flag``. Each child dataset is split into ordered
    sub-sequences first, then those sub-sequences are shuffled globally.

    When ``dataset_ratios`` is provided, the sampler balances datasets by the
    number of samples contributed per epoch instead of letting larger datasets
    dominate naturally. Smaller datasets are re-sampled by cycling through
    their shuffled sub-sequences.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 min_seq_len=20,
                 max_seq_len=40,
                 dataset_ratios=None,
                 total_samples=None):

        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = sync_random_seed(seed)
        
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.dataset_ratios = dataset_ratios

        assert samples_per_gpu == 1, "Currently only supports samples_per_gpu=1 for sequence-based training."
        assert hasattr(self.dataset, 'datasets'), "The dataset should be a ConcatDataset."
        
        self.total_size = len(self.dataset)
        self.cumulative_sizes = self.dataset.cumulative_sizes
        self.dataset_sub_sequences = self._generate_sub_sequences()
        self.sub_sequences = [
            sub_sequence
            for dataset_sub_sequences in self.dataset_sub_sequences
            for sub_sequence in dataset_sub_sequences
        ]
        self.dataset_ratios = self._normalize_dataset_ratios(dataset_ratios)
        self.total_samples = self._resolve_total_samples(total_samples)
        self.samples_per_dataset = self._build_dataset_sample_targets()
        self.approx_samples_per_replica = int(
            np.ceil(self.total_samples / self.num_replicas))

    def _generate_sub_sequences(self):
        """Split each child dataset into ordered sub-sequences."""
        all_sub_sequences = []
        for dataset_idx, sub_dataset in enumerate(self.dataset.datasets):
            if not hasattr(sub_dataset, 'flag'):
                print(
                    f"Warning: Dataset {dataset_idx} has no 'flag' attribute. "
                    "Treating it as a single sequence.")
                flags = np.zeros(len(sub_dataset), dtype=np.int64)
            else:
                flags = sub_dataset.flag

            start_idx = self.cumulative_sizes[dataset_idx] - len(sub_dataset)
            
            group_indices = {}
            for i, flag in enumerate(flags):
                if flag not in group_indices:
                    group_indices[flag] = []
                group_indices[flag].append(start_idx + i)

            dataset_sub_sequences = []
            for seq_indices in group_indices.values():
                g = torch.Generator()
                g.manual_seed(self.seed + dataset_idx + len(dataset_sub_sequences))
                
                start = 0
                while start < len(seq_indices):
                    seq_len = torch.randint(
                        self.min_seq_len,
                        self.max_seq_len + 1,
                        (1,),
                        generator=g).item()
                    end = min(start + seq_len, len(seq_indices))
                    dataset_sub_sequences.append(seq_indices[start:end])
                    start = end

            all_sub_sequences.append(dataset_sub_sequences)
        
        return all_sub_sequences

    def _normalize_dataset_ratios(self, dataset_ratios):
        if dataset_ratios is None:
            return None

        if len(dataset_ratios) != len(self.dataset.datasets):
            raise ValueError(
                'dataset_ratios must have the same length as the number of '
                f'datasets, but got {len(dataset_ratios)} and '
                f'{len(self.dataset.datasets)}.')

        ratios = [float(ratio) for ratio in dataset_ratios]
        if any(ratio < 0 for ratio in ratios):
            raise ValueError('dataset_ratios must be non-negative.')
        if sum(ratios) <= 0:
            raise ValueError(
                'dataset_ratios must contain at least one positive value.')
        return ratios

    def _resolve_total_samples(self, total_samples):
        if total_samples is None:
            return self.total_size
        if total_samples <= 0:
            raise ValueError('total_samples must be a positive integer.')
        return int(total_samples)

    def _build_dataset_sample_targets(self):
        if self.dataset_ratios is None:
            return None

        ratio_sum = sum(self.dataset_ratios)
        raw_targets = [
            self.total_samples * ratio / ratio_sum
            for ratio in self.dataset_ratios
        ]
        sample_targets = [int(np.floor(target)) for target in raw_targets]
        remainder = self.total_samples - sum(sample_targets)

        if remainder > 0:
            order = sorted(
                range(len(raw_targets)),
                key=lambda idx: (raw_targets[idx] - sample_targets[idx]),
                reverse=True)
            for idx in order[:remainder]:
                sample_targets[idx] += 1

        return sample_targets

    def _sample_sub_sequences_for_dataset(self,
                                          dataset_sub_sequences,
                                          target_samples,
                                          generator):
        if target_samples <= 0 or not dataset_sub_sequences:
            return []

        sampled_sub_sequences = []
        sampled_samples = 0
        while sampled_samples < target_samples:
            shuffled_indices = torch.randperm(
                len(dataset_sub_sequences), generator=generator).tolist()
            for sub_sequence_idx in shuffled_indices:
                sampled_sub_sequence = dataset_sub_sequences[sub_sequence_idx]
                sampled_sub_sequences.append(sampled_sub_sequence)
                sampled_samples += len(sampled_sub_sequence)
                if sampled_samples >= target_samples:
                    break
        return sampled_sub_sequences

    def _sample_epoch_sub_sequences(self, generator):
        if self.dataset_ratios is None:
            shuffled_indices = torch.randperm(
                len(self.sub_sequences), generator=generator).tolist()
            return [self.sub_sequences[idx] for idx in shuffled_indices]

        epoch_sub_sequences = []
        for dataset_idx, dataset_sub_sequences in enumerate(self.dataset_sub_sequences):
            epoch_sub_sequences.extend(
                self._sample_sub_sequences_for_dataset(
                    dataset_sub_sequences,
                    self.samples_per_dataset[dataset_idx],
                    generator))

        if not epoch_sub_sequences:
            raise RuntimeError('No sub-sequences were sampled for the current epoch.')

        shuffled_indices = torch.randperm(
            len(epoch_sub_sequences), generator=generator).tolist()
        return [epoch_sub_sequences[idx] for idx in shuffled_indices]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        sub_sequences = self._sample_epoch_sub_sequences(g)

        total_sequence_size = int(
            np.ceil(len(sub_sequences) / self.num_replicas)) * self.num_replicas
        padding_size = total_sequence_size - len(sub_sequences)
        if padding_size > 0:
            sub_sequences.extend(sub_sequences[:padding_size])

        rank_sub_sequences = sub_sequences[self.rank:total_sequence_size:self.num_replicas]

        final_indices = []
        for sub_sequence in rank_sub_sequences:
            final_indices.extend(sub_sequence)

        return iter(final_indices)

    def __len__(self):
        return self.approx_samples_per_replica
        
    def set_epoch(self, epoch):
        self.epoch = epoch


@SAMPLER.register_module()
class InfiniteGroupEachSampleInBatchSamplerForVGGT(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.
    Shuffling is only done for group order, not done within groups.
    """

    def __init__(self, 
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):

        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.queue_length = dataset.queue_length
        self.batch_size = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = sync_random_seed(seed)

        self.size = len(self.dataset)

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = samples_per_gpu * num_replicas
        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)}        

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator 
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx) 
            for local_sample_idx in range(self.batch_size)]
        
        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

    def _infinite_group_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(self._infinite_group_indices(), 
                                    global_sample_idx, 
                                    None,
                                    self.global_batch_size)

    def __iter__(self):
        while True:
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    self.buffer_per_local_sample[local_sample_idx] = \
                        copy.deepcopy(
                            self.group_idx_to_sample_idxs[new_group_idx])
                idx = 0
                while idx < self.queue_length - 1 and len(self.buffer_per_local_sample[local_sample_idx]) > 1:
                    self.buffer_per_local_sample[local_sample_idx].pop(0)
                    idx = idx + 1
                curr_batch.append(self.buffer_per_local_sample[local_sample_idx].pop(0))
            
            yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        return self.size
        
    def set_epoch(self, epoch):
        self.epoch = epoch