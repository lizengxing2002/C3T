# C3T: Conservative Cross-Contrast Transport

Reference implementation of C3T for multi-contrast MRI synthesis. From a single observed
contrast of a slice, the network predicts the pairwise contrast changes on a three-node
graph, reconciles the competing routes in closed form, and returns the synthesized images
together with a spatial variance map in one forward pass.

## Environment

Python 3.9 or newer.

```
pip install -r requirements.txt
```

This installs PyTorch, NumPy, SciPy, scikit-image and OpenCV. A CUDA build of PyTorch is
recommended for training; CPU-only runs work as well and are just slower.

## Data

Slices are 8-bit single-channel PNGs, one directory per subject and one file per contrast:

```
<root>/<split>/<subject>/T1.png
<root>/<split>/<subject>/T2.png
<root>/<split>/<subject>/FLAIR.png
```

`<split>` is `train`, `val` or `test`. Every ordered pair of the contrasts present in a
subject is used, so a subject that lacks a contrast simply yields fewer pairs.

## Usage

```bash
# train
python train.py --config configs/default.json --data <root> --output runs/c3t --device cuda

# synthesize every contrast for each test slice
python infer.py --checkpoint runs/c3t/best.pt --data <root> --split test \
                --output exports/c3t --device cuda

# score an export
python evaluate.py --export exports/c3t/export.json
```

Pass `--device cpu` to run without a GPU, and `--steps N` to train.py to shorten a run for a
quick check.

Training writes `best.pt`, `latest.pt`, `train.jsonl`, `validation.jsonl`, `status.json` and
`resolved_config.json` into the run directory. Inference writes
`<output>/<direction>/{fake,real,variance}/` plus `export.json`. Evaluation writes
`per_image_metrics.csv` and `summary.json`.

## Tests

```
python -m pytest tests/ -q
```
