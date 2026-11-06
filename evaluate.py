"""Score an export directory produced by infer.py.

Writes per-image metrics, a per-direction summary, and an overall summary.
Every prediction is compared against the reference slice at its native
resolution; predictions are never rescaled or histogram matched.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from c3t.data import read_image
from c3t.metrics import METRICS, evaluate_pair


def main():
    parser = argparse.ArgumentParser(description='Evaluate C3T exports')
    parser.add_argument('--export', required=True, help='export.json written by infer.py')
    parser.add_argument('--output', default=None, help='defaults to the export directory')
    args = parser.parse_args()

    export_path = Path(args.export)
    export = json.loads(export_path.read_text())
    output = Path(args.output) if args.output else export_path.parent
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in export['records']:
        prediction = read_image(record['fake_path'])
        reference = read_image(record['real_path'])
        if prediction.shape != reference.shape:
            raise ValueError('shape mismatch for ' + record['case'])
        rows.append(
            {
                'case': record['case'],
                'direction': record['direction'],
                **evaluate_pair(reference, prediction),
            }
        )
    if not rows:
        raise ValueError('no exported records to score')

    with (output / 'per_image_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    directions = {}
    for row in rows:
        directions.setdefault(row['direction'], []).append(row)
    summary = {
        'count': len(rows),
        'metrics': {k: float(np.mean([r[k] for r in rows])) for k in METRICS},
        'directions': {
            name: {
                'count': len(group),
                **{k: float(np.mean([r[k] for r in group])) for k in METRICS},
            }
            for name, group in sorted(directions.items())
        },
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
