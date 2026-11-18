# Model checkpoints

Pretrained weights are not bundled. Train the model using `scripts/train.py`, then export a checkpoint for inference.

Run from the repository root:

```bash
python scripts/import_checkpoint.py --fit-root runs/formal --seed 2026102301 --output checkpoints/kcalresformer.pt
```

Use the exported file with `scripts/infer.py` or `scripts/evaluate.py`. The checkpoint includes the neural weights, calibrated physical parameters and fitted residual scale.

Generated `.pt` files are excluded from Git by default. Model weights can be distributed separately as GitHub release assets.
