# C3T: Conservative Cross-Contrast Transport

Reference implementation of **C3T**, a source-anchored network for multi-contrast MRI
synthesis. From a single observed slice it predicts every pairwise contrast change on a
three-node contrast graph, reconciles the competing routes in closed form, and returns the
reconstructed images together with an analytical spatial variance map in one forward pass.

```
Zengxing Li, Yu Fu, Jiangyan Liu, Ping Cui, Kaiqiang Xu, Yanyan Huang,
Meng Niu, Lei Fang, Zhiguo Shi, Zhijun Yao, Lequan Yu
"C3T: Conservative Cross-Contrast Transport for Consistent and Uncertainty-Aware
 Multi-Contrast MRI Synthesis", IEEE Transactions on Medical Imaging.
```

## Idea

Treat the contrasts `V = {T1, T2, FLAIR}` as nodes and the transitions
`E = {(T1,T2), (T1,FLAIR), (T2,FLAIR)}` as directed edges. True inter-contrast changes are a
conservative (curl-free) field on the triangle: they sum to zero around the loop and are
integrable into node values. C3T predicts all three edges independently, so its raw estimates
generally violate that constraint, and the loop residual is exactly the disagreement between
the direct route and the two-step detour through the third contrast. Instead of rendering a
target from scratch, C3T estimates the *change* and reconstructs

```
x̂_t = x_s + H⁻¹(d̂_{s→t})
```

so the measured anatomy of the source slice is never re-synthesized.

### Signed wavelet evidence

A four-level U-Net (widths 32/64/128/256, ConvNeXt-style blocks with FiLM conditioning on a
modality embedding, four-head self-attention on an 8×8 pooled bottleneck) encodes `x_s` into
`F ∈ R^{32×M×N}`. Resampled to the two-level orthonormal Haar packet grid (`K = 16` subbands
of size `M/4 × N/4`), per-edge heads emit a residual `r`, a gain logit `g`, and a
block-shared log-variance `ℓ ∈ [-6, 3]`:

```
γ_{e,i} = 2 tanh(g_{e,i}) ∈ (-2, 2)
d_{e,i} = γ_{e,i} [H x_s]_i + r_{e,i}
w_{e,i} = sg[exp(-ℓ_{e,i})]
```

The signed gain copies (`γ = 0`), attenuates, suppresses (`γ = -1`), or inverts (`γ < -1`)
source structures, so contrast inversions are modelled without regenerating anatomy.
Zero-initialized residuals and gains with an `ℓ` bias of `-3` start the network at the
identity `x̂_t = x_s`.

### Precision-weighted conservative projection

Node values are defined up to a common shift, so the gauge `u_T1 = 0` is fixed and the
projection is solved in closed form. With `c_e = +1` for `(T1,T2)` and `(T2,FLAIR)` and
`c_e = -1` for `(T1,FLAIR)`:

```
r_i = curl(d_i)
d̂_{e,i} = d_{e,i} - c_e * (w_{e,i}^{-1} / Σ_{e'} w_{e',i}^{-1}) * r_i
```

The loop residual is distributed in proportion to edge variance, so the least certain edge
absorbs the largest correction. The same projector maps the working covariance onto the
conservative subspace, `Σ̂_i = P_{W_i} W_i⁻¹ P_{W_i}ᵀ`, whose diagonal is the inverse-variance
fusion of the direct route and the two-step route; a spatial variance map is obtained by
averaging over the 16 orthonormal subbands.

### Graph-consistent multiscale refinement

Details are restored through node potentials `φ`, whose edge differences `Bφ` are curl-free by
construction and therefore invariant under the projector:

```
φ = 0.2 S₅(tanh φ_lo) + 0.1 c ⊙ (tanh φ_hi - S₅(tanh φ_hi))
d̂ ← P_W d + H(Bφ)
```

`S₅` is a 5×5 box filter and `c = (1 + σ̄)⁻¹` is a confidence gate derived from the
edge-averaged coarse variance. Inference needs a single refinement pass and no second
projection.

### Training objective

```
L = L_img + 0.1 L_raw + 0.005 L_nll + 0.01 L_curl + 0.05 L_anc
```

* `L_img`: Charbonnier (`ε = 1e-3`) plus `0.5` MSE, weighted by `m = 0.25 + 0.75·1[x_s > τ ∨ y_t > τ]`
  with `τ = 1/255`; plus `0.25 (1 - SSIM)`, `0.1` L1 image gradient, and `0.05` L1 Haar losses.
* `L_raw`: L1 on the unprojected prediction `x̃_t = x_s + H⁻¹(d + H(Bφ))`, supervising each edge
  before projection.
* `L_nll`: Gaussian negative log-likelihood with the stop-gradient squared error.
* `L_curl`: mean squared pre-projection curl.
* `L_anc`: dual-anchor consistency. Every fourth step the same weights also run on `(y_t, t)`,
  and the reconciled changes of the two anchorings are compared with SmoothL1 across all edges.

## Repository layout

```
c3t/
  model.py       contrast graph, wavelet packet transform, U-Net backbone,
                 signed evidence heads, precision-weighted projection, anchored output
  refinement.py  graph-consistent multiscale refinement
  losses.py      training objective: Charbonnier, SSIM, gradient, Haar, NLL, curl, anchor
  data.py        paired slice dataset and D4 augmentation
  metrics.py     PSNR, SSIM, RMSE, NRMSE, VIF
configs/default.json
train.py  infer.py  evaluate.py
tests/
```

## Data

Slices are stored as single-channel PNGs, one directory per subject and one file per contrast:

```
<root>/<split>/<subject>/T1.png
<root>/<split>/<subject>/T2.png
<root>/<split>/<subject>/FLAIR.png
```

`<split>` is `train`, `val`, or `test`. All ordered pairs of present contrasts are enumerated
automatically, so a subject that lacks a contrast simply yields fewer pairs. Images are
converted to float in `[0, 1]`; random D4 rotations and flips are applied to both views of a
pair during training only.

## Installation

```
pip install -r requirements.txt
```

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

Training writes `best.pt`, `latest.pt`, `train.jsonl`, `validation.jsonl`, `status.json`, and
`resolved_config.json` into the run directory; the checkpoint with the lowest validation MSE is
kept. Inference writes `<output>/<direction>/{fake,real,variance}/` plus an `export.json`
describing every record. Evaluation writes `per_image_metrics.csv` and `summary.json` with
overall and per-direction means.

### Configuration

`configs/default.json` reproduces the paper settings: 60 k steps, batch size 8, AdamW with
learning rate `2e-4` and weight decay `1e-4`, cosine decay to `1e-6` with 500 warmup steps,
EMA `0.999`, anchor term every fourth step with weight `0.05`. The network has **2.22 M**
parameters with refinement enabled and **2.21 M** without it.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers the wavelet packet transform and its inverse, the incidence and gauge
operators, conservative projection and covariance propagation, the refinement module, the data
loader and its augmentation, the training objective under autocast, and the metric
implementations.

## Citation

```bibtex
@article{li2026c3t,
  title   = {C3T: Conservative Cross-Contrast Transport for Consistent and
             Uncertainty-Aware Multi-Contrast MRI Synthesis},
  author  = {Li, Zengxing and Fu, Yu and Liu, Jiangyan and Cui, Ping and Xu, Kaiqiang
             and Huang, Yanyan and Niu, Meng and Fang, Lei and Shi, Zhiguo
             and Yao, Zhijun and Yu, Lequan},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2026}
}
```
