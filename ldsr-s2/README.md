# LDSR-S2 weights folder

This folder is used by `app.py` when `weights_dir=./ldsr-s2`.

- `load.py` loads `opensr_model.SRLatentDiffusion`.
- On first run it downloads:
  - `config_10m.yaml` from the official ESAOpenSR repo.
  - the checkpoint specified by `ckpt_version` into this same folder.

Expected input for this loader is 4 channels in `R,G,B,NIR` order.
The API auto-selects RGB-NIR from larger Sentinel-2 rasters.
