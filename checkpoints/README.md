# Checkpoints

The paper's three shared exact-K checkpoints were not found locally. No neural weights are included. Expected source hashes are in `configs/checkpoint_manifest.json`.

Recover `training/<seed>/exact_klmmse_residual/best.pt` from the original formal server output, then run `scripts/import_checkpoint.py` with the corresponding seed. The importer rejects early-development and layout-specialist checkpoints whose hashes differ.

New retraining outputs can be exported using `--fit-root`. Smoke weights remain labeled `SMOKE_ONLY` and are not paper checkpoints. Generated `.pt` files are ignored by Git by default; publish verified weights as a separate release asset if desired.
