#!/usr/bin/env python3
"""Fuse one bright, one dark, and one normal exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hdrfusion.three_exposure import FusionSettings, fuse_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained three-exposure HDR fusion")
    parser.add_argument("--bright", required=True, help="Bright bracket image (or RAW)")
    parser.add_argument("--dark", required=True, help="Dark bracket image (or RAW)")
    parser.add_argument("--normal", required=True, help="Normal exposure; this is the image anchor")
    parser.add_argument("--output", required=True, help="Output .jpg, .png, or 16-bit .tif")
    parser.add_argument("--no-align", action="store_true", help="Skip exposure registration")
    parser.add_argument("--shadow-strength", type=float, default=None,
                        help="Override the image-adaptive shadow strength")
    parser.add_argument("--highlight-strength", type=float, default=None,
                        help="Override the image-adaptive highlight strength")
    args = parser.parse_args()
    if args.shadow_strength is not None and not 0.0 <= args.shadow_strength <= 0.82:
        parser.error("--shadow-strength must be between 0 and 0.82")
    if args.highlight_strength is not None and not 0.0 <= args.highlight_strength <= 0.82:
        parser.error("--highlight-strength must be between 0 and 0.82")

    settings = FusionSettings(
        shadow_strength=args.shadow_strength,
        highlight_strength=args.highlight_strength,
        align=not args.no_align,
    )
    diagnostics = fuse_files(args.bright, args.dark, args.normal, args.output, settings)
    print(json.dumps({"output": str(Path(args.output).resolve()), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
