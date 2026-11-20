from pathlib import Path
import csv
import json
import time

import numpy as np
import torch
from PIL import Image

import train_new_dataset as training

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raws_new"
MANIFEST = ROOT / "reports" / "unseen_test_manifest.csv"
CHECKPOINT = ROOT / "checkpoints" / "fusion_best.pt"
OUT = ROOT / "outputs" / "unseen_test"
REPORT = ROOT / "reports" / "unseen_test_export.json"


def main():
    for path in (RAW_DIR, MANIFEST, CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
        groups = list(csv.DictReader(f))

    required = {"under", "normal", "over"}
    if not groups or not required.issubset(groups[0].keys()):
        raise ValueError("Manifest must contain under, normal, and over columns.")

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Device: {training.DEVICE}", flush=True)
    print(f"Unseen exposure groups: {len(groups)}", flush=True)
    print(f"Loading checkpoint: {CHECKPOINT}", flush=True)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=training.DEVICE,
        weights_only=False,
    )

    model = training.FusionNet().to(training.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(
        f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')} | "
        f"validation loss: {checkpoint.get('val_loss', float('nan')):.6f}",
        flush=True,
    )

    results = []
    started = time.time()

    for number, group in enumerate(groups, 1):
        names = [group[k].strip() for k in ("under", "normal", "over")]
        paths = [RAW_DIR / name for name in names]

        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Input image missing: {path}")

        under, normal, over = [
            training.load_rgb_uint8(path) for path in paths
        ]

        shapes = [im.shape[:2] for im in (under, normal, over)]
        if len(set(shapes)) != 1:
            raise ValueError(
                f"Dimension mismatch in group {number}: {names} -> {shapes}"
            )

        fused = training.infer_tiled(model, under, normal, over)
        stem = Path(group["normal"]).stem

        jpg_path = OUT / f"{stem}_fusion.jpg"
        tif_path = OUT / f"{stem}_fusion_16bit.tif"

        fused_u8 = np.clip(np.round(fused * 255), 0, 255).astype(np.uint8)
        Image.fromarray(fused_u8).save(
            jpg_path,
            quality=95,
            subsampling=0,
        )

        # Optional 16-bit output derived from the model's JPEG-based inputs.
        try:
            import tifffile
            fused_u16 = np.clip(
                np.round(fused * 65535), 0, 65535
            ).astype(np.uint16)
            tifffile.imwrite(tif_path, fused_u16, photometric="rgb")
            tiff_result = str(tif_path.relative_to(ROOT))
        except ImportError:
            tiff_result = None

        results.append({
            "scene": stem,
            "under": names[0],
            "normal": names[1],
            "over": names[2],
            "jpeg": str(jpg_path.relative_to(ROOT)),
            "tiff16": tiff_result,
            "width": int(fused.shape[1]),
            "height": int(fused.shape[0]),
        })

        print(
            f"[{number:02d}/{len(groups):02d}] Saved {jpg_path.name}",
            flush=True,
        )

        del under, normal, over, fused, fused_u8

    report = {
        "status": "inference_complete",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_validation_loss": float(checkpoint.get("val_loss", float("nan"))),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "input_groups": len(groups),
        "outputs_generated": len(results),
        "elapsed_seconds": round(time.time() - started, 2),
        "outputs": results,
        "notes": [
            "Inference only: no model training was performed.",
            "Exposure order comes from the manifest and was estimated using image brightness.",
            "The 16-bit TIFF is derived from JPEG inputs and does not restore RAW sensor precision.",
            "Visual inspection is required; these images have no reference targets in this test.",
        ],
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nUNSEEN INFERENCE COMPLETE", flush=True)
    print(f"Outputs: {len(results)}", flush=True)
    print(f"Images: {OUT}", flush=True)
    print(f"Report: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
