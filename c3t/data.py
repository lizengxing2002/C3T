"""Folder-convention paired loader for C3T.

Expected layout::

    <root>/<split>/<case>/<MODALITY>.png

with ``split`` in ``{train, val, test}`` and ``MODALITY`` in ``T1/T2/FLAIR``.
Every ordered pair of contrasts present inside a case becomes one record, so a
single observed image supervises all of its target contrasts. Partitions are
literal directory names: this module never builds, remaps, or reshuffles them.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import MODALITIES

EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def read_image(path):
    """Read one 8-bit grayscale slice. Raises instead of silently resizing."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('unreadable image: ' + str(path))
    return image


def discover(root, split, modalities=MODALITIES):
    """Enumerate all ordered contrast pairs under ``<root>/<split>``."""
    base = Path(root) / split
    if not base.is_dir():
        raise ValueError('missing split directory: ' + str(base))
    records = []
    for case in sorted(p for p in base.iterdir() if p.is_dir()):
        present = {}
        for modality in modalities:
            for extension in EXTENSIONS:
                candidate = case / (modality + extension)
                if candidate.is_file():
                    present[modality] = candidate
                    break
        for source in modalities:
            if source not in present:
                continue
            for target in modalities:
                if target == source or target not in present:
                    continue
                records.append(
                    {
                        'source_path': str(present[source]),
                        'target_path': str(present[target]),
                        'source_modality': source,
                        'target_modality': target,
                        'direction': source + '->' + target,
                        'case': case.name,
                    }
                )
    if not records:
        raise ValueError('no complete contrast pairs under ' + str(base))
    return records


class Pairs(Dataset):
    """Yields ``{'x', 'y', 'source', 'target', 'record_index'}`` in [0, 1]."""

    def __init__(
        self, root=None, split='train', modalities=MODALITIES, augment=False, records=None
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError("split must be 'train', 'val' or 'test'")
        self.modalities = tuple(modalities)
        self.records = (
            list(records) if records is not None else discover(root, split, self.modalities)
        )
        if not self.records:
            raise ValueError('empty split: ' + split)
        self.augment = bool(augment and split == 'train')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        x = read_image(record['source_path'])
        y = read_image(record['target_path'])
        if x.shape != y.shape:
            raise ValueError('source/target geometry mismatch: ' + record['case'])
        if self.augment:
            k = random.randrange(4)
            x = np.rot90(x, k)
            y = np.rot90(y, k)
            if random.random() < 0.5:
                x = x[:, ::-1]
                y = y[:, ::-1]
        return {
            'x': torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None] / 255,
            'y': torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None] / 255,
            'source': MODALITIES.index(record['source_modality']),
            'target': MODALITIES.index(record['target_modality']),
            'record_index': index,
        }
