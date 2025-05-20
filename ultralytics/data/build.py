# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import random
from pathlib import Path
import math # Added for DistributedWeightedSamplerWrapper

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed, Sampler as TorchSampler # Added Sampler for wrapper base

from ultralytics.data.loaders import (LOADERS, LoadImages, LoadPilAndNumpy, LoadScreenshots, LoadStreams, LoadTensor,
                                      SourceTypes, autocast_list)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr, LOGGER # Added LOGGER
from ultralytics.utils.checks import check_file

from .dataset import YOLODataset, WeightedMultiDatasetSampler # Added WeightedMultiDatasetSampler
from .utils import PIN_MEMORY


class DistributedWeightedSamplerWrapper(TorchSampler):
    """
    Wrapper for a sampler to make it compatible with DDP.
    It takes all indices from the underlying sampler, optionally shuffles them,
    and then distributes them among DDP replicas.
    """
    def __init__(self, underlying_sampler, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0, drop_last: bool = False):
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, num_replicas is {num_replicas}")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer.")

        self.underlying_sampler = underlying_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle  # Whether to shuffle the global list from underlying_sampler
        self.seed = seed
        self.drop_last = drop_last

        total_size_underlying = len(self.underlying_sampler)
        if self.drop_last:
            self.num_samples_per_replica = total_size_underlying // self.num_replicas
            self.effective_total_size = self.num_samples_per_replica * self.num_replicas
        else:
            self.num_samples_per_replica = math.ceil(total_size_underlying / self.num_replicas)
            self.effective_total_size = self.num_samples_per_replica * self.num_replicas
            
    def __iter__(self):
        indices = list(self.underlying_sampler) # Get all indices for the epoch. len(indices) 30471, len(set(indices)) 13859

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ordering = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in ordering]

        # Adjust indices to effective_total_size (pad or truncate)
        if not self.drop_last:
            if len(indices) < self.effective_total_size: # Pad if shorter
                indices += indices[:(self.effective_total_size - len(indices))]
            elif len(indices) > self.effective_total_size: # Truncate if longer
                 indices = indices[:self.effective_total_size]
        else: # drop_last
            indices = indices[:self.effective_total_size] # Truncate to effective_total_size

        # Subsample
        sharded_indices = indices[self.rank::self.num_replicas]
        
        assert len(sharded_indices) == self.num_samples_per_replica, \
            f"DistributedWeightedSamplerWrapper: len(sharded_indices)={len(sharded_indices)} != self.num_samples_per_replica={self.num_samples_per_replica}"

        return iter(sharded_indices)

    def __len__(self):
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if hasattr(self.underlying_sampler, 'set_epoch'):
            self.underlying_sampler.set_epoch(epoch)


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(cfg, img_path, batch, data, mode='train', rect=False, stride=32):
    """Build YOLO Dataset."""
    return YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == 'train',  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        # rect=cfg.rect or rect,  # rectangular batches
        rect=False,  # TODO wjz just temp
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0)


def build_dataloader(dataset, batch_size, workers, shuffle=True, rank=-1):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    # nw = 1  # TODO wjz  for debug
    # Using a more informative print statement for number of workers
    cpu_cores = os.cpu_count()
    print(f"Using {nw} dataloader workers (specified: {workers}, CPU cores: {cpu_cores}, CUDA devices: {nd}, batch_size: {batch_size})")

    sampler = None
    shuffle_dataloader_flag = shuffle  # Default shuffle for dataloader

    # Determine if weighted sampler should be used
    use_weighted_sampler = (
        isinstance(dataset, YOLODataset) and
        hasattr(dataset, 'is_multidataset') and dataset.is_multidataset and
        hasattr(dataset, 'dataset_indices') and hasattr(dataset, 'dataset_lens') and
        hasattr(dataset, 'data') and bool(dataset.dataset_lens)  # Ensure dataset_lens is not empty and evaluate to bool
    )

    if use_weighted_sampler:
        dataset_weights = dataset.data.get('weights')
        num_datasets_from_lens = len(dataset.dataset_lens)

        if dataset_weights is None or len(dataset_weights) != num_datasets_from_lens:
            if dataset_weights is not None:
                LOGGER.warning(
                    f"WARNING ⚠️ build_dataloader: Mismatched 'weights' length ({len(dataset_weights)}) "
                    f"and number of datasets ({num_datasets_from_lens}). Using uniform repeat time of 1.0 for all datasets."
                )
            dataset_weights = [1.0] * num_datasets_from_lens
        
        if all(w == 0 for w in dataset_weights):
            LOGGER.warning(
                "WARNING ⚠️ build_dataloader: All dataset_weights (repeat times) are zero. "
                "The sampler will produce no samples for this epoch."
            )
        elif any(w < 0 for w in dataset_weights):
            LOGGER.error(
                "ERROR ❌ build_dataloader: Negative dataset_weights (repeat times) detected. "
                "WeightedMultiDatasetSampler will raise an error."
            )

        base_weighted_sampler = WeightedMultiDatasetSampler(
            dataset_indices=dataset.dataset_indices,
            dataset_lens=dataset.dataset_lens,
            dataset_weights=dataset_weights,
        )
        
        if rank != -1:  # DDP active
            if not torch.distributed.is_initialized():
                # This case should ideally be handled by the training script before calling build_dataloader in DDP mode
                LOGGER.error("ERROR ❌ build_dataloader: Distributed training requested (rank != -1) but torch.distributed is not initialized.")
                # Fallback or raise error might be needed, for now, proceed and hope it's caught later or works if single node DDP
                num_replicas = 1 # Fallback, will not be correct for multi-node DDP
                effective_rank = 0
            else:
                num_replicas = torch.distributed.get_world_size()
                effective_rank = torch.distributed.get_rank()

            sampler = DistributedWeightedSamplerWrapper(
                underlying_sampler=base_weighted_sampler,
                num_replicas=num_replicas,
                rank=effective_rank, # Use the obtained rank
                shuffle=shuffle,  # Use main shuffle arg to decide if wrapper shuffles global list
                # seed is default 0 for wrapper, uses seed + epoch
                # drop_last can be False by default for samplers usually
            )
        else:  # Not DDP
            sampler = base_weighted_sampler
        
        shuffle_dataloader_flag = False  # Sampler handles ordering

    else:  # Not using weighted sampler (e.g., single dataset)
        if rank != -1:  # DDP active
            # DistributedSampler infers rank and world_size if DDP is initialized
            sampler = distributed.DistributedSampler(dataset, shuffle=shuffle)
            shuffle_dataloader_flag = False
        else:  # Not DDP
            sampler = None
            # shuffle_dataloader_flag remains the original 'shuffle' value

    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK) # RANK is -1 for non-DDP, or process rank
    
    return InfiniteDataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle_dataloader_flag,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://', 'tcp://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower() == 'screen'
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError('Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict')

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, imgsz=640, vid_stride=1, buffer=False):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        imgsz (int, optional): The size of the image for inference. Default is 640.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.
        buffer (bool, optional): Determined whether stream frames will be buffered. Default is False.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, webcam, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(webcam, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif webcam:
        dataset = LoadStreams(source, imgsz=imgsz, vid_stride=vid_stride, buffer=buffer)
    elif screenshot:
        dataset = LoadScreenshots(source, imgsz=imgsz)
    elif from_img:
        dataset = LoadPilAndNumpy(source, imgsz=imgsz)
    else:
        dataset = LoadImages(source, imgsz=imgsz, vid_stride=vid_stride)

    # Attach source types to the dataset
    setattr(dataset, 'source_type', source_type)

    return dataset
