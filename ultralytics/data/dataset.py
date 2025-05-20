# Ultralytics YOLO 🚀, AGPL-3.0 license
import contextlib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
import hashlib

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import Sampler

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable

from .augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label
# from ultralytics.data.augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms
# from ultralytics.data.base import BaseDataset
# from ultralytics.data.utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = '1.0.3'


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.
    Can handle multiple dataset sources specified in the data configuration.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
             Expected format for multi-dataset:
             train:
               - /path/to/dataset1
               - /path/to/dataset2
             val: /path/to/val_dataset
             weights: [0.7, 0.3] # Optional weights for train datasets
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        self.is_multidataset = isinstance(kwargs.get('img_path', ''), list)
        self.dataset_paths = kwargs.get('img_path', []) if self.is_multidataset else [kwargs.get('img_path', '')]
        self.dataset_indices = []
        self.dataset_lens = []
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path('./labels.cache'), im_files_to_cache=None, label_files_to_cache=None):
        """
        Cache dataset labels, check images and read shapes for a specific subset of files.

        Args:
            path (Path): path where to save the cache file.
            im_files_to_cache (list): List of image files for this specific cache.
            label_files_to_cache (list): List of label files for this specific cache.
        Returns:
            (dict): labels cache dictionary.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'

        im_files = im_files_to_cache if im_files_to_cache is not None else self.im_files
        label_files = label_files_to_cache if label_files_to_cache is not None else self.label_files
        total = len(im_files)

        if total == 0:
            x['hash'] = ''
            x['results'] = 0, 0, 0, 0, 0
            x['msgs'] = []
            return x

        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(im_files, label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(label_files + im_files)
        x['results'] = nf, nm, ne, nc, total
        x['msgs'] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training, handling multiple datasets and caching."""
        all_labels = []
        combined_im_files = []
        combined_label_files = []
        self.dataset_indices = [0]
        self.dataset_lens = []

        dataset_paths_to_process = self.dataset_paths

        total_nf, total_nm, total_ne, total_nc, total_n = 0, 0, 0, 0, 0
        all_msgs = []

        im_files_per_dataset = self._group_files_by_source_path(self.im_files, dataset_paths_to_process)

        for i, dataset_path in enumerate(dataset_paths_to_process):
            current_im_files = im_files_per_dataset[i]
            # sort current_im_files by file name to ensure same hash if same dataset
            current_im_files.sort()
            if not current_im_files:
                LOGGER.warning(f"{self.prefix}WARNING ⚠️ No images found for dataset path: {dataset_path}")
                self.dataset_lens.append(0)
                self.dataset_indices.append(self.dataset_indices[-1])
                continue

            current_label_files = img2label_paths(current_im_files)
            path_hash = hashlib.sha256(str(dataset_path).encode()).hexdigest()[:8]
            cache_path = Path(self.data.get('path', '.')) / f'{Path(dataset_path).stem}_{path_hash}.cache'

            try:
                cache, exists = load_dataset_cache_file(cache_path), True
                assert cache['version'] == DATASET_CACHE_VERSION, f"Cache version mismatch: {cache['version']} != {DATASET_CACHE_VERSION}"
                assert cache['hash'] == get_hash(current_label_files + current_im_files), f"Cache hash mismatch: {cache['hash']} != {get_hash(current_label_files + current_im_files)}"
                LOGGER.info(f"{self.prefix}Rank[{LOCAL_RANK}] INFO Cache loaded for {cache_path}. Note that if file content is changed but file name is the same, the cache will be reused, you MUST delete the cache file manually.")
            except (FileNotFoundError, AssertionError, AttributeError, TypeError) as e:
                # if LOCAL_RANK in (-1, 0):
                LOGGER.warning(f"{self.prefix}Rank[{LOCAL_RANK}] WARNING ⚠️ Cache loading failed for {cache_path}, recreating cache. {e}")
                cache, exists = self.cache_labels(cache_path, current_im_files, current_label_files), False  # takes long time: "train: Scanning xxx"

            nf, nm, ne, nc, n = cache.pop('results')
            if exists and LOCAL_RANK in (-1, 0):
                d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
                TQDM(None, desc=self.prefix + d, total=n, initial=n)
                if cache['msgs']:
                    LOGGER.info('\n'.join(cache['msgs']))

            [cache.pop(k) for k in ('hash', 'version', 'msgs')]
            labels = cache['labels']
            if not labels:
                LOGGER.warning(f'WARNING ⚠️ No labels found in {cache_path} for dataset {dataset_path}.')

            all_labels.extend(labels)
            num_labels_added = len(labels)
            self.dataset_lens.append(num_labels_added)
            self.dataset_indices.append(self.dataset_indices[-1] + num_labels_added)

            total_nf += nf
            total_nm += nm
            total_ne += ne
            total_nc += nc
            total_n += n
            all_msgs.extend(cache.get('msgs', []))

        if not all_labels:
            LOGGER.error(f'ERROR ❌ No labels found across all datasets {dataset_paths_to_process}. {HELP_URL}')
            raise ValueError(f"No labels found in any specified dataset paths: {dataset_paths_to_process}")

        self.im_files = [lb['im_file'] for lb in all_labels]
        self.labels = all_labels
        self.ni = len(self.labels)

        if LOCAL_RANK in (-1, 0):
            LOGGER.info(f"{self.prefix}Combined results: {total_nf} images ({total_nm} missing, {total_ne} empty labels, {total_nc} corrupt)")

        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in self.labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Combined dataset: Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either detect or segment datasets, not mixed ones.')
            for lb in self.labels:
                lb['segments'] = []
        if len_cls == 0:
            LOGGER.warning(f'WARNING ⚠️ Combined dataset: No labels found after processing all sources. Training may not work correctly. {HELP_URL}')

        self.dataset_indices.pop()

        assert len(self.im_files) == self.ni, "Mismatch between image file count and label count after processing."
        assert len(self.dataset_indices) == len(self.dataset_lens) == len(dataset_paths_to_process), "Dataset index/length tracking mismatch."
        assert sum(self.dataset_lens) == self.ni, "Sum of dataset lengths does not match total labels."

        return self.labels

    def _group_files_by_source_path(self, all_files, source_paths):
        """Groups a list of file paths based on which source directory they originated from."""
        grouped_files = [[] for _ in source_paths]
        norm_source_paths = [str(Path(p).resolve()) for p in source_paths]

        for file_path in all_files:
            resolved_file_path = str(Path(file_path).resolve())
            found = False
            sorted_indices = sorted(range(len(norm_source_paths)), key=lambda k: len(norm_source_paths[k]), reverse=True)
            for i in sorted_indices:
                if resolved_file_path.startswith(norm_source_paths[i]):
                    grouped_files[i].append(file_path)
                    found = True
                    break
            if not found:
                LOGGER.warning(f"Could not assign file {file_path} to any source path: {source_paths}")

        return grouped_files

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """Custom your label format here."""
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLO Classification Dataset.

    Args:
        root (str): Dataset path.

    Attributes:
        cache_ram (bool): True if images should be cached in RAM, False otherwise.
        cache_disk (bool): True if images should be cached on disk, False otherwise.
        samples (list): List of samples containing file, index, npy, and im.
        torch_transforms (callable): torchvision transforms applied to the dataset.
        album_transforms (callable, optional): Albumentations transforms applied to the dataset if augment is True.
    """

    def __init__(self, root, args, augment=False, cache=False, prefix=''):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Dataset path.
            args (Namespace): Argument parser containing dataset related settings.
            augment (bool, optional): True if dataset should be augmented, False otherwise. Defaults to False.
            cache (bool | str | optional): Cache setting, can be True, False, 'ram' or 'disk'. Defaults to False.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:
            self.samples = self.samples[:round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f'{prefix}: ') if prefix else ''
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = self.verify_images()
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]
        self.torch_transforms = classify_transforms(args.imgsz, rect=args.rect)
        self.album_transforms = classify_albumentations(
            augment=augment,
            size=args.imgsz,
            scale=(1.0 - args.scale, 1.0),
            hflip=args.fliplr,
            vflip=args.flipud,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            auto_aug=False) if augment else None

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:
            im = cv2.imread(f)
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return {'img': sample, 'cls': j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f'{self.prefix}Scanning {self.root}...'
        path = Path(self.root).with_suffix('.cache')

        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            cache = load_dataset_cache_file(path)
            assert cache['version'] == DATASET_CACHE_VERSION
            assert cache['hash'] == get_hash([x[0] for x in self.samples])
            nf, nc, n, samples = cache.pop('results')
            if LOCAL_RANK in (-1, 0):
                d = f'{desc} {nf} images, {nc} corrupt'
                TQDM(None, desc=d, total=n, initial=n)
                if cache['msgs']:
                    LOGGER.info('\n'.join(cache['msgs']))
            return samples

        nf, nc, msgs, samples, x = 0, 0, [], [], {}
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = TQDM(results, desc=desc, total=len(self.samples))
            for sample, nf_f, nc_f, msg in pbar:
                if nf_f:
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f'{desc} {nf} images, {nc} corrupt'
            pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        x['hash'] = get_hash([x[0] for x in self.samples])
        x['results'] = nf, nc, len(samples), samples
        x['msgs'] = msgs
        save_dataset_cache_file(self.prefix, path, x)
        return samples


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc
    gc.disable()
    cache = np.load(str(path) + '.npy', allow_pickle=True).item()  # because np.save() by default appends a .npy extension to the filename
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x['version'] = DATASET_CACHE_VERSION
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # Remove old cache
        try:
            np.save(str(path), x)  # The np.save() by default appends a .npy extension to the filename if it's not already present.
            LOGGER.info(f'{prefix}INFO ✅ Cache saved to {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING ⚠️ Failed to save cache to {path}: {e}')
    else:
        LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


class WeightedMultiDatasetSampler(Sampler):
    """
    Sampler for handling multiple datasets with weighted sampling.

    Picks a dataset based on weights, then samples uniformly from that dataset.
    """
    def __init__(self, dataset_indices, dataset_lens, dataset_weights, num_samples=None, replacement=True):
        """
        Args:
            dataset_indices (list): List of start indices for each dataset in the combined dataset.
            dataset_lens (list): List of lengths (number of samples) for each dataset.
            dataset_weights (list): List of weights corresponding to each dataset.
            num_samples (int, optional): Total number of samples to draw. If None, defaults to sum of all dataset lengths.
            replacement (bool): If True, samples are drawn with replacement. Default is True.
                                (Required for weighted sampling across datasets unless num_samples is carefully chosen)
        """
        if not isinstance(dataset_indices, (list, tuple)) or not isinstance(dataset_lens, (list, tuple)) or not isinstance(dataset_weights, (list, tuple)):
            raise ValueError("dataset_indices, dataset_lens, and dataset_weights must be lists or tuples.")
        if len(dataset_indices) != len(dataset_lens) or len(dataset_lens) != len(dataset_weights):
            raise ValueError("Lengths of dataset_indices, dataset_lens, and dataset_weights must match.")
        if not all(w >= 0 for w in dataset_weights):
            raise ValueError("Dataset weights must be non-negative.")

        self.dataset_indices = dataset_indices
        self.dataset_lens = dataset_lens
        self.dataset_weights = np.array(dataset_weights, dtype=np.float32)
        self.num_datasets = len(dataset_lens)
        self.total_len = sum(dataset_lens)

        if num_samples is None:
            self.num_samples = self.total_len
        else:
            if not isinstance(num_samples, int) or num_samples <= 0:
                raise ValueError("num_samples should be a positive integer.")
            self.num_samples = num_samples

        self.replacement = replacement

        total_weight = self.dataset_weights.sum()
        if total_weight <= 0:
            LOGGER.warning("Total dataset weight is zero or negative. Using uniform weighting.")
            self.dataset_probs = np.ones(self.num_datasets) / self.num_datasets
        else:
            self.dataset_probs = self.dataset_weights / total_weight

    def __iter__(self):
        """Yields indices for one epoch based on weighted sampling."""
        dataset_choices = np.random.choice(self.num_datasets, size=self.num_samples, p=self.dataset_probs)

        indices = []
        for dataset_idx in dataset_choices:
            start_index = self.dataset_indices[dataset_idx]
            length = self.dataset_lens[dataset_idx]
            if length == 0:
                LOGGER.debug(f"Skipping empty dataset index {dataset_idx}")
                continue

            offset = np.random.randint(0, length)  # TODO draw without replacement
            global_index = start_index + offset
            indices.append(global_index)

        return iter(indices)

    def __len__(self):
        """Returns the number of samples drawn in one epoch."""
        return self.num_samples


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # Example Usage (modify paths and YAML structure accordingly)
    import yaml
    import re
    from torch.utils.data import DataLoader

    test_yaml_path = '/mnt/pai-storage-8/tianyuan/face_qrcode_det/yolov8/configs/test.yaml'

    try:
        with open(test_yaml_path, errors='ignore', encoding='utf-8') as f:
            s = f.read()
            if not s.isprintable():
                s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
            data = yaml.safe_load(s) or {}
            data['yaml_file'] = str(test_yaml_path)

        yaml_parent = Path(test_yaml_path).parent
        data_root = Path(data.get('path', yaml_parent)).resolve()

        train_paths = []
        if 'train' in data and isinstance(data['train'], list):
            for p in data['train']:
                path = Path(p)
                if not path.is_absolute():
                    path = (data_root / p).resolve()
                train_paths.append(str(path))
        else:
            LOGGER.error("Multi-dataset training requires 'train' to be a list of paths in the YAML.")
            exit()

        val_path = []
        if 'val' in data and isinstance(data['val'], list):
            for p in data['val']:
                path = Path(p)
                if not path.is_absolute():
                    path = (data_root / p).resolve()
                val_path.append(str(path))

        train_dataset = YOLODataset(
            img_path=train_paths,
            imgsz=640,
            batch_size=16,
            augment=True,
            # hyp=DEFAULT_CFG,
            rect=False,
            cache='ram',
            data=data,
            prefix='Train: '
        )

        train_weights = data.get('weights', [1.0] * len(train_paths))
        if len(train_weights) != len(train_paths):
            LOGGER.warning(f"Number of weights ({len(train_weights)}) does not match number of train datasets ({len(train_paths)}). Using equal weights.")
            train_weights = [1.0] * len(train_paths)

        if not hasattr(train_dataset, 'dataset_indices') or not hasattr(train_dataset, 'dataset_lens'):
            raise RuntimeError("Dataset object missing required attributes for weighted sampling. Check initialization.")
        if len(train_dataset.dataset_indices) != len(train_paths):
            LOGGER.error(f"Mismatch after dataset init: {len(train_dataset.dataset_indices)} indices vs {len(train_paths)} paths.")

        train_sampler = WeightedMultiDatasetSampler(
            dataset_indices=train_dataset.dataset_indices,
            dataset_lens=train_dataset.dataset_lens,
            dataset_weights=train_weights,
            num_samples=len(train_dataset)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=8,
            sampler=train_sampler,
            pin_memory=True,
            collate_fn=YOLODataset.collate_fn
        )

        print(f"Combined Training Dataset Size: {len(train_dataset)}")
        print(f"Sampler will yield {len(train_sampler)} indices per epoch.")
        print(f"Dataset lengths: {train_dataset.dataset_lens}")
        print(f"Dataset start indices: {train_dataset.dataset_indices}")
        print(f"Dataset weights: {train_weights}")
        print(f"Dataset probabilities: {train_sampler.dataset_probs}")

        num_batches_to_show = 5
        for i, batch in enumerate(train_loader):
            if i >= num_batches_to_show:
                break
            print(f"\nBatch {i+1}:")
            print(f"  Image shape: {batch['img'].shape}")
            print(f"  Labels batch_idx unique values: {torch.unique(batch['batch_idx'])}")

        if val_path:
            val_dataset = YOLODataset(
                img_path=val_path,
                imgsz=640,
                batch_size=16,
                augment=False,
                # hyp=DEFAULT_CFG,
                rect=True,
                cache='ram',
                data=data,
                prefix='Val: '
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=16,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                collate_fn=YOLODataset.collate_fn
            )
            print(f"\nValidation Dataset Size: {len(val_dataset)}")

    except FileNotFoundError:
        print(f"Error: YAML file not found at {test_yaml_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

