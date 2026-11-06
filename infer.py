"""Export synthesized contrasts and their variance maps from a trained checkpoint.

One forward pass per observed contrast yields every missing target at once: the
network reconciles all graph edges from a single anchor, so the predictions for
one source are mutually consistent by construction. Nothing is derived from the
target image, so an export is reproducible from the checkpoint and the input.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from c3t.data import Pairs, read_image
from c3t.model import C3T, MODALITIES, ModelConfig


def group_by_anchor(records):
    """Collapse the ordered-pair listing into one entry per (case, source)."""
    anchors = {}
    for record in records:
        key = (record['case'], record['source_modality'])
        entry = anchors.setdefault(key, {'source_path': record['source_path'], 'targets': {}})
        entry['targets'][record['target_modality']] = record['target_path']
    return anchors


def main():
    parser = argparse.ArgumentParser(description='Export C3T predictions')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True, help='root holding the split folder')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(2)

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = state['config']
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('output directory is not empty')
    output.mkdir(parents=True, exist_ok=True)

    network = C3T(ModelConfig(**config['model']))
    network.load_state_dict(state['ema'], strict=True)
    network.to(device).eval()

    dataset = Pairs(args.data, args.split)
    anchors = group_by_anchor(dataset.records)
    records = []
    groups = {}
    forwards = 0
    with torch.inference_mode():
        for (case, source_modality), entry in anchors.items():
            source_path = Path(entry['source_path'])
            image = read_image(source_path)
            tensor = torch.from_numpy(image.astype(np.float32) / 255)[None, None].to(device)
            source = torch.tensor([MODALITIES.index(source_modality)], device=device)
            result = network(tensor, source)
            forwards += 1

            for target_modality, target_path in entry['targets'].items():
                target = torch.tensor([MODALITIES.index(target_modality)], device=device)
                direction = source_modality + '->' + target_modality
                prediction = C3T.select(result, target)[0, 0].float().cpu().numpy()
                variance = C3T.select(result, target, 'variance')[0, 0].cpu().numpy()
                if not np.isfinite(prediction).all() or not np.isfinite(variance).all():
                    raise FloatingPointError('non-finite prediction for ' + case)

                destination = output / direction
                for folder in ('fake', 'real', 'variance'):
                    (destination / folder).mkdir(parents=True, exist_ok=True)
                stem = case + '_' + target_modality
                fake_path = destination / 'fake' / (stem + '.png')
                real_path = destination / 'real' / (stem + '.png')
                variance_path = destination / 'variance' / (stem + '.npy')
                if not cv2.imwrite(str(fake_path), (prediction.clip(0, 1) * 255).astype(np.uint8)):
                    raise IOError('cannot write ' + str(fake_path))
                if not cv2.imwrite(str(real_path), read_image(target_path)):
                    raise IOError('cannot write ' + str(real_path))
                np.save(variance_path, variance)

                records.append(
                    {
                        'case': case,
                        'direction': direction,
                        'source_modality': source_modality,
                        'target_modality': target_modality,
                        'source_path': str(source_path.resolve()),
                        'fake_path': str(fake_path.resolve()),
                        'real_path': str(real_path.resolve()),
                        'variance_path': str(variance_path.resolve()),
                        'fraction_clipped': float(((prediction < 0) | (prediction > 1)).mean()),
                        'mean_variance': float(variance.mean()),
                    }
                )
                group = groups.setdefault(
                    direction,
                    {
                        'direction': direction,
                        'fake_dir': str(destination / 'fake'),
                        'real_dir': str(destination / 'real'),
                        'count': 0,
                    },
                )
                group['count'] += 1

    export = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'split': args.split,
        'forward_passes': forwards,
        'normalization': 'uint8/255 input; prediction clipped to [0,1] for 8-bit export',
        'uncertainty': 'per-target variance from the precision-weighted graph projection',
        'groups': list(groups.values()),
        'records': records,
    }
    (output / 'export.json').write_text(json.dumps(export, indent=2))
    print('EXPORTED', len(records), 'pairs in', len(groups), 'directions to', output)
    print('FORWARDS', forwards, 'for', len(anchors), 'observed contrasts')


if __name__ == '__main__':
    main()
