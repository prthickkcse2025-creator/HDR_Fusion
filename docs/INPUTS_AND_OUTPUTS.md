# Inputs and outputs

## One bracket

Provide three photographs of the same scene: a bright frame, a dark frame, and a normal/middle frame. All three must have matching pixel dimensions after orientation correction and the same bit depth.

```text
scene/
├── over.jpg      # bright bracket; shadow source
├── under.jpg     # dark bracket; highlight source
└── normal.jpg    # base image
```

The bracket files should be captured from a fixed camera position. Slight motion can be handled by registration; substantial camera movement or moving objects can create artifacts.

## CLI

```bash
python scripts/fuse_three_exposures.py \
  --bright scene/over.jpg \
  --dark scene/under.jpg \
  --normal scene/normal.jpg \
  --output output/final.jpg
```

The output directory is created automatically. JPEG output is saved at high quality. A `.tif` or `.tiff` path writes a 16-bit TIFF when the decoded input is 16-bit; install `tifffile` for this option. The JSON response includes image-specific fusion diagnostics. Use `--no-align` only when the brackets are already registered.

For camera RAW, install `python -m pip install -e ".[raw]"`. JPEG, PNG, and TIFF do not need that extra.
