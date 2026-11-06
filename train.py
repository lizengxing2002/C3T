"""Train C3T from scratch on paired multi-contrast slices.

Model selection uses only the validation split. Test metrics are never computed
here. Determinism: seeds are set for every RNG; CUDA kernels that lack
deterministic implementations still emit warnings rather than failing.
"""

import argparse
import copy
import datetime
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from c3t.data import Pairs
from c3t.losses import anchor_consistency, objective
from c3t.model import C3T, ModelConfig


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def checkpoint(path, value):
    temporary = Path(path).with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def to_uint8(tensor):
    return (tensor.detach().float().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)


@torch.inference_mode()
def validate(model, dataset, device, batch_size):
    model.eval()
    rows = []
    by_direction = {}
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        x, y, source, target = [batch[k].to(device) for k in ('x', 'y', 'source', 'target')]
        prediction = to_uint8(C3T.select(model(x, source), target))
        reference = np.rint(y.cpu().numpy() * 255).astype(np.uint8)
        for i in range(len(prediction)):
            error = ((prediction[i, 0].astype(float) - reference[i, 0].astype(float)) / 255) ** 2
            row = {
                'mse': float(error.mean()),
                'PSNR': float(
                    peak_signal_noise_ratio(reference[i, 0], prediction[i, 0], data_range=255)
                ),
                'SSIM': float(
                    structural_similarity(reference[i, 0], prediction[i, 0], data_range=255)
                ),
            }
            rows.append(row)
            direction = dataset.records[int(batch['record_index'][i])]['direction']
            by_direction.setdefault(direction, []).append(row)
    model.train()
    keys = ('mse', 'PSNR', 'SSIM')
    return {
        'n': len(rows),
        **{k: float(np.mean([r[k] for r in rows])) for k in keys},
        'directions': {
            d: {k: float(np.mean([r[k] for r in group])) for k in keys}
            for d, group in by_direction.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Train C3T')
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True, help='root holding train/ and val/ folders')
    parser.add_argument('--output', default='runs/c3t', help='run directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, help='override the configured step budget')
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    if args.steps is not None:
        config['steps'] = args.steps
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    torch.set_num_threads(config.get('threads', 2))
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'latest.pt').exists():
        raise FileExistsError('run directory already holds a checkpoint; choose a new --output')

    train = Pairs(args.data, 'train', augment=True)
    valset = Pairs(args.data, 'val')
    model = C3T(ModelConfig(**config['model'])).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0001)
    )
    generator = torch.Generator().manual_seed(seed)
    best = float('inf')
    best_step = 0

    config['parameters'] = sum(p.numel() for p in model.parameters())
    config['selection'] = 'minimum validation MSE'
    config['train_pairs'] = len(train)
    config['val_pairs'] = len(valset)
    atomic_json(run / 'resolved_config.json', config)

    started = time.monotonic()
    initial = validate(ema, valset, device, config.get('validation_batch_size', 2))
    atomic_json(run / 'initial_validation.json', initial)
    print('INITIAL_VALIDATION', json.dumps(initial), flush=True)

    def pack(step):
        return {
            'model': model.state_dict(),
            'ema': ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'best_mse': best,
            'best_step': best_step,
            'config': config,
        }

    amp = bool(config.get('amp', True)) and device.type == 'cuda'
    for step in range(1, config['steps'] + 1):
        ids = torch.randint(len(train), (config['batch_size'],), generator=generator).tolist()
        batch = default_collate([train[i] for i in ids])
        x, y, source, target = [batch[k].to(device) for k in ('x', 'y', 'source', 'target')]
        warmup = min(1.0, step / max(1, config.get('warmup_steps', 500)))
        progress = (step - 1) / max(1, config['steps'])
        floor = config.get('min_lr', 1e-6)
        lr = warmup * (floor + (config['lr'] - floor) * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            result = model(x, source)
            loss, terms = objective(result, x, y, target, config.get('loss_weights'))
            anchored = torch.zeros((), device=device)
            if config.get('anchor_weight', 0) > 0 and step % config.get('anchor_every', 4) == 0:
                reverse = model(y, target)
                anchored = anchor_consistency(result, reverse)
                loss = loss + config['anchor_weight'] * anchored
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite loss at step ' + str(step))
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()

        decay = min(config.get('ema', 0.999), (1 + step) / (10 + step))
        with torch.no_grad():
            for target_parameter, parameter in zip(ema.parameters(), model.parameters()):
                target_parameter.lerp_(parameter, 1 - decay)

        if step % config.get('log_every', 100) == 0 or step == 1:
            row = {
                'step': step,
                'loss': loss.item(),
                'lr': lr,
                'gradient_norm': float(gradient),
                'anchor': float(anchored),
                **{k: float(v.detach()) for k, v in terms.items()},
            }
            with (run / 'train.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            atomic_json(
                run / 'status.json',
                {
                    'state': 'training',
                    'step': step,
                    'steps': config['steps'],
                    'time': now(),
                    'device': str(device),
                    'last_loss': loss.item(),
                    'elapsed_seconds': time.monotonic() - started,
                },
            )
            print('TRAIN', json.dumps(row), flush=True)

        if step % config['validate_every'] == 0 or step == config['steps']:
            metrics = validate(ema, valset, device, config.get('validation_batch_size', 2))
            metrics.update(step=step, time=now())
            with (run / 'validation.jsonl').open('a') as handle:
                handle.write(json.dumps(metrics) + '\n')
            if metrics['mse'] < best:
                best = metrics['mse']
                best_step = step
                checkpoint(run / 'best.pt', pack(step))
                atomic_json(run / 'best_validation.json', metrics)
            checkpoint(run / 'latest.pt', pack(step))
            print('VALIDATION', json.dumps(metrics), flush=True)

    final = json.loads((run / 'best_validation.json').read_text()) if best_step else initial
    summary = {
        'state': 'completed',
        'time': now(),
        'steps': config['steps'],
        'best_step': best_step,
        'best_validation': final,
        'initial_validation': initial,
        'parameters': config['parameters'],
        'elapsed_seconds': time.monotonic() - started,
        'test_evaluated_by_trainer': False,
    }
    atomic_json(run / 'status.json', summary)
    print('COMPLETED', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
