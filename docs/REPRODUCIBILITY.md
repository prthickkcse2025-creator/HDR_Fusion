# Reproducibility and extension

## Environment

- Python 3.10 or newer
- Dependencies declared in `pyproject.toml`
- Optional `rawpy` for camera RAW input
- Optional `tifffile` for 16-bit TIFF output

Install the package in editable mode with `python -m pip install -e .`. The README example uses the committed `samples/` images and can be reproduced with the CLI.

## What to record

Keep the CLI's JSON diagnostics with the output. For comparisons, record source filenames, software revision, image dimensions/bit depth, alignment scores, adaptive thresholds, strengths, and whether overrides or `--no-align` were used. Do not compare only downsampled previews when evaluating fine detail.

## Reference images

The small images in `samples/input/` and `samples/output/` are committed examples for documentation and quick manual review. They are not training data and are not used to tune the algorithm at runtime. Do not add private client interiors or full-resolution datasets to the repository without permission.

## Suggested extension points

- Improve alignment for brackets with parallax while preserving a safe zero-contribution fallback.
- Add metrics and visual comparisons across varied rooms and exposure gaps.
- Add tests for image validation, output bit depth, and the normal-anchor behavior.
- Keep any future learned model separate from this deterministic inference path and identify its training data and license.
