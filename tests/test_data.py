import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from c3t.data import Pairs, discover


def pattern(seed, size=32):
    """Asymmetric deterministic uint8 image: rotation and flip are distinguishable."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size), dtype=np.uint8)


def make_case(root, split, case, modalities=('T1', 'T2', 'FLAIR')):
    directory = Path(root) / split / case
    directory.mkdir(parents=True)
    for index, modality in enumerate(modalities):
        cv2.imwrite(str(directory / (modality + '.png')), pattern(index))
    return directory


def d4(image, index):
    rotated = np.rot90(image, index % 4)
    return rotated[:, ::-1] if index >= 4 else rotated


class PairDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        make_case(self.root, 'train', 'case_a')
        make_case(self.root, 'val', 'case_b', ('T1', 'T2'))

    def test_all_ordered_pairs_enumerated(self):
        records = discover(self.root, 'train')
        self.assertEqual(len(records), 6)
        self.assertEqual(
            {r['direction'] for r in records},
            {'T1->T2', 'T1->FLAIR', 'T2->T1', 'T2->FLAIR', 'FLAIR->T1', 'FLAIR->T2'},
        )

    def test_missing_modality_yields_only_complete_pairs(self):
        records = discover(self.root, 'val')
        self.assertEqual(len(records), 2)
        self.assertEqual({r['direction'] for r in records}, {'T1->T2', 'T2->T1'})

    def test_single_modality_is_rejected(self):
        make_case(self.root, 'test', 'case_c', ('T1',))
        with self.assertRaises(ValueError):
            discover(self.root, 'test')

    def test_unknown_split_rejected(self):
        with self.assertRaises(ValueError):
            Pairs(self.root, 'training')

    def test_absent_split_directory_rejected(self):
        with self.assertRaises(ValueError):
            discover(self.root, 'test')


class PairLoaderTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        make_case(self.root, 'train', 'case_a')
        make_case(self.root, 'val', 'case_b')

    def test_item_shapes_range_and_modality_ids(self):
        dataset = Pairs(self.root, 'train')
        item = dataset[0]
        self.assertEqual(item['x'].shape, (1, 32, 32))
        self.assertTrue(0 <= float(item['x'].min()) and float(item['x'].max()) <= 1)
        self.assertIn(item['source'], (0, 1, 2))
        self.assertIn(item['target'], (0, 1, 2))
        self.assertNotEqual(item['source'], item['target'])

    def test_value_scaling_is_divided_by_255(self):
        dataset = Pairs(self.root, 'train')
        record = dataset.records[0]
        raw = cv2.imread(record['source_path'], cv2.IMREAD_GRAYSCALE)
        item = dataset[0]
        self.assertTrue(np.allclose(item['x'][0].numpy(), raw.astype(np.float32) / 255))

    def test_validation_split_never_augments(self):
        dataset = Pairs(self.root, 'val', augment=True)
        first = dataset[0]['x'].clone()
        for _ in range(8):
            self.assertTrue(np.array_equal(first.numpy(), dataset[0]['x'].numpy()))

    def test_augmentation_applies_one_d4_transform_to_both_views(self):
        dataset = Pairs(self.root, 'train', augment=True)
        record = dataset.records[0]
        source = cv2.imread(record['source_path'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        target = cv2.imread(record['target_path'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        seen = set()
        for _ in range(128):
            item = dataset[0]
            x = item['x'][0].numpy()
            y = item['y'][0].numpy()
            matches = [i for i in range(8) if np.allclose(d4(source, i), x)]
            self.assertEqual(len(matches), 1, 'augmented view is not a single D4 transform')
            index = matches[0]
            self.assertTrue(np.allclose(d4(target, index), y), 'views augmented inconsistently')
            seen.add(index)
        self.assertGreater(len(seen), 1)


if __name__ == '__main__':
    unittest.main()
