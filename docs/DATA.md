# Data and model input contract

The model consumes pilot LS observations and nominal side information. Supervised channels and transmitted bits must remain outside the model input dictionary.

| Field | Shape / meaning |
|---|---|
| `pilot_values` | complex `[B,Pmax]`, LS values `Yp/Xp`; an unbatched `[P]` is also accepted |
| `pilot_coordinates` | `[B,Pmax,2]` or shared `[P,2]`; Hz and seconds |
| `pilot_valid_mask` | Boolean `[B,Pmax]`; required when layouts are padded |
| `target_coordinates` | `[B,1680,2]` or shared `[1680,2]`; ascending frequency-major 120×14 grid |
| `noise_variance` | Scalar or `[B]`; homogeneous complex LS-noise variance per observation |
| `snr_db` | Scalar or `[B]`; nominal SNR in dB |
| `rms_delay_spread_s` | Scalar or `[B]`; nominal RMS delay spread in seconds |
| `max_doppler_hz` | Scalar or `[B]`; nominal maximum Doppler in Hz |

The frontend requires complete Cartesian pilot and target coordinates, homogeneous white pilot noise and the specified separable covariance. Arbitrary missing pilot REs and correlated interference are not supported by this factorization. The neural branch requires 120×14 grids; running another pilot configuration does not establish accuracy under a different channel distribution.

`H_true` with shape `[B,120,14]` is an evaluation label. Optional `Y`, `tx_bits` and the shared `data_mask` enable BER evaluation. `scripts.data_io.select_input` filters these fields out before inference. Optional `layouts` and one-based `channel_ids` retain view identity.

The formal source dataset version is `formal_dmrs_ber_20261023_v1`:

```text
/path/to/formal_dmrs/
├── protocol.json
├── manifest.json
├── calibration/     # 10 shards, 5,000 channels
├── train/           # 200 shards, 100,000 channels
└── validation/      # 20 shards, 10,000 channels
```

Each shard is `shard_NNNN.mat` with a `.mat.complete.json` SHA256 completion marker. Standard shards contain 500 independent channels. Without `--generated-data`, training checks the original manifest checksum recorded in `configs/training.json`. With `--generated-data`, it validates the generator's completed audit and freeze, then binds the newly generated manifest to the new training output. It checks the source recipe, per-shard hashes and actual geometry. Relocating data does not change channel identity. A new MATLAB version or newly written MAT file need not have the original byte checksum.

MAT fields include `H_true`, `X`, `Y`, `Y_noiseless`, `tx_bits`, `noise_variance`, `snr_db`, `delay_spread_ns`, `max_doppler_hz`, `channel_seed`, `data_seed` and `noise_seed`. Geometry is stored separately in `configs/geometry.mat`. Data preparation reads only explicitly selected shards and verifies their completion hashes.

DM-RS symbol indices are zero-based: A `[2]`, B `[2,11]`, C `[2,7,11]`, D `[2,5,8,11]`, with 40/80/120/160 pilots. The first 8,000 validation channels select neural checkpoints; the last 2,000 fit alpha. All fitting uses A/C observations. Paired layouts and real/imaginary components are not separate independent channels.

The protocol uses TDL-A, a 120×14 grid, 15 kHz spacing, perfect timing, nominal delay/Doppler side information and common QPSK payloads. These data are not a separate standards-compliant PDSCH codeword for each layout. Full source/test arrays are not bundled; generate source arrays with `scripts/generate_data.py` using MATLAB and 5G Toolbox. See `DATA_GENERATION.md`. The Python synthetic example is not an NR waveform simulation or a replacement for formal data.

Generated datasets also include `geometry.mat` and a `generation/` directory holding the frozen generator, schedules, replay checks and MATLAB logs. The generated protocol records its mode and validation partition. The small smoke mode uses 2 channels per shard and 6 calibration / 8 train / 10 validation channels; the same loader supports it, and formal training rejects it unless `--smoke` is explicitly selected.
