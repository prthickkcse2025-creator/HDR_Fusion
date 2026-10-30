from pathlib import Path
import csv
import json
import gc

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raws_new"
MANIFEST = ROOT / "reports" / "unseen_test_manifest.csv"
FUSED_DIR = ROOT / "outputs" / "unseen_test"
OUT = ROOT / "outputs" / "shadow_recovery_final_correction"
REPORT = ROOT / "reports" / "shadow_recovery_tiled.json"

OUT.mkdir(parents=True, exist_ok=True)

TILE = 512
HALO = 16
PREVIEW_MAX = 1024


def srgb_to_linear(x):
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4
    )


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055
    )


def luminance(rgb):
    return (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )


def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def preview_rgb(image):
    im = image.copy()
    im.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def scene_parameters(fused_preview, normal_preview, over_preview):
    fused = srgb_to_linear(fused_preview)
    normal = srgb_to_linear(normal_preview)
    over = srgb_to_linear(over_preview)

    yf = luminance(fused)
    yn = luminance(normal)
    yo = luminance(over)

    qdark, qmid, qbright = np.quantile(yn, [0.10, 0.30, 0.82])

    valid = (
        (yn >= qmid) & (yn <= qbright)
        & (yo > 0.015) & (yo < 0.94)
        & (yf > 0.005)
    )

    if valid.sum() < 100:
        return None

    ratio = float(np.median(yf[valid] / np.maximum(yo[valid], 1e-6)))
    aligned = yo * ratio
    gain = np.maximum(aligned - yf, 0) / np.maximum(yf, 0.012)

    gain_scale = float(np.quantile(gain[valid], 0.75))
    if not np.isfinite(gain_scale) or gain_scale <= 1e-6:
        return None

    return {
        "qdark": float(qdark),
        "qmid": float(qmid),
        "qbright": float(qbright),
        "ratio": ratio,
        "gain_scale": gain_scale,
        "over_clip_threshold": float(np.quantile(yo, 0.90)),
    }



def process_tile(fused_u8, normal_u8, over_u8, params):
    fused_srgb = fused_u8.astype(np.float32) / 255.0
    normal_srgb = normal_u8.astype(np.float32) / 255.0
    over_srgb = over_u8.astype(np.float32) / 255.0

    fused = srgb_to_linear(fused_srgb)
    normal = srgb_to_linear(normal_srgb)
    over = srgb_to_linear(over_srgb)

    yf = luminance(fused)
    yn = luminance(normal)
    yo = luminance(over)

    # -------------------------------------------------
    # 1. ADAPTIVE SHADOW RECOVERY
    # -------------------------------------------------
    dark_mask = 1.0 - smoothstep(
        params["qdark"], params["qmid"], yn
    )

    aligned_over = yo * params["ratio"]
    lift = np.maximum(aligned_over - yf, 0.0)

    gain = lift / np.maximum(yf, 0.012)

    confidence = gain / (
        gain + params["gain_scale"]
    )

    highlight_reliability = 1.0 - smoothstep(
        params["over_clip_threshold"], 0.995, yo
    )

    alpha = dark_mask * confidence * highlight_reliability

    # Conservative lift to prevent foggy shadows.
    max_lift = np.minimum(lift, 0.12)
    recovered_y = yf + alpha * max_lift

    recovered = fused * (
        recovered_y / np.maximum(yf, 1e-5)
    )[..., None]

    # -------------------------------------------------
    # 2. NORMAL EXPOSURE COLOUR ANCHOR
    # -------------------------------------------------
    recovered_srgb = linear_to_srgb(recovered)
    recovered_linear = srgb_to_linear(recovered_srgb)

    yr = luminance(recovered_linear)

    recovered_chroma = recovered_linear / np.maximum(
        yr[..., None], 1e-5
    )
    normal_chroma = normal / np.maximum(
        yn[..., None], 1e-5
    )

    chroma = np.where(
        (yn > 0.015)[..., None],
        normal_chroma,
        recovered_chroma
    )

    anchored = chroma * yr[..., None]

    peak = np.max(anchored, axis=2, keepdims=True)
    gamut_scale = np.minimum(
        1.0, 0.99 / np.maximum(peak, 1e-6)
    )
    anchored *= gamut_scale

    # -------------------------------------------------
    # 3. MULTI-SCALE DETAIL FROM NORMAL EXPOSURE
    # -------------------------------------------------
    # Blur is used only to estimate detail.
    # The final image is never replaced by a blurred image.

    normal_gray = Image.fromarray(
        np.round(np.clip(yn, 0, 1) * 255).astype(np.uint8),
        mode="L"
    )

    base_fine = np.asarray(
        normal_gray.filter(
            ImageFilter.GaussianBlur(radius=1.0)
        ),
        dtype=np.float32
    ) / 255.0

    base_coarse = np.asarray(
        normal_gray.filter(
            ImageFilter.GaussianBlur(radius=2.5)
        ),
        dtype=np.float32
    ) / 255.0

    # Fine edges plus restrained medium-scale texture.
    fine_detail = yn - base_fine
    medium_detail = base_fine - base_coarse

    # Noise suppression: ignore tiny fluctuations.
    fine_detail = np.where(
        np.abs(fine_detail) > 0.0015,
        fine_detail,
        0.0
    )

    medium_detail = np.where(
        np.abs(medium_detail) > 0.0025,
        medium_detail,
        0.0
    )

    detail = (
        0.85 * fine_detail
        + 0.18 * medium_detail
    )

    detail = np.clip(detail, -0.035, 0.035)

    # Reduce detail transfer in very dark and clipped regions.
    shadow_protection = smoothstep(0.015, 0.09, yn)
    highlight_protection = 1.0 - smoothstep(
        0.90, 0.99, yn
    )

    detail_strength = (
        0.85 * shadow_protection * highlight_protection
    )

    anchored_y = luminance(anchored)

    restored_y = np.clip(
        anchored_y + detail_strength * detail,
        0.0,
        1.0
    )

    # Equal RGB scaling retains the anchored colour ratios.
    detail_scale = restored_y / np.maximum(
        anchored_y, 1e-5
    )

    result_linear = anchored * detail_scale[..., None]
    result = linear_to_srgb(result_linear)

    return np.round(
        np.clip(result, 0, 1) * 255
    ).astype(np.uint8)

def process_scene(group, index, total):
    stem = Path(group["normal"]).stem
    png_path = OUT / f"{stem}_shadow_recovery.png"
    jpg_path = OUT / f"{stem}_shadow_recovery.jpg"

    if png_path.exists() and jpg_path.exists():
        print(f"[{index:02d}/{total:02d}] {stem}: already complete")
        return {"scene": stem, "status": "previously_completed"}

    fused_path = FUSED_DIR / f"{stem}_fusion.jpg"
    normal_path = RAW / group["normal"]
    over_path = RAW / group["over"]

    for p in (fused_path, normal_path, over_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    with Image.open(fused_path) as f, \
         Image.open(normal_path) as n, \
         Image.open(over_path) as o:

        f = f.convert("RGB")
        n = n.convert("RGB")
        o = o.convert("RGB")

        if not (f.size == n.size == o.size):
            raise ValueError(f"Dimension mismatch: {stem}")

        width, height = f.size

        # Small preview calculates scene-wide adaptive parameters.
        fp = preview_rgb(f)
        npv = preview_rgb(n)
        op = preview_rgb(o)

        params = scene_parameters(fp, npv, op)

        del fp, npv, op
        gc.collect()

        if params is None:
            raise RuntimeError(f"Insufficient exposure signal: {stem}")

        # Full-resolution output canvas.
        output = Image.new("RGB", (width, height))

        for y0 in range(0, height, TILE):
            for x0 in range(0, width, TILE):
                x1 = min(x0 + TILE, width)
                y1 = min(y0 + TILE, height)

                # Read an overlapping tile to avoid edge artefacts.
                left = max(0, x0 - HALO)
                top = max(0, y0 - HALO)
                right = min(width, x1 + HALO)
                bottom = min(height, y1 + HALO)

                ft = np.asarray(
                    f.crop((left, top, right, bottom)),
                    dtype=np.uint8
                )
                nt = np.asarray(
                    n.crop((left, top, right, bottom)),
                    dtype=np.uint8
                )
                ot = np.asarray(
                    o.crop((left, top, right, bottom)),
                    dtype=np.uint8
                )

                tile_result = process_tile(ft, nt, ot, params)

                # Keep only the central tile; discard overlap margins.
                cx0 = x0 - left
                cy0 = y0 - top
                cx1 = cx0 + (x1 - x0)
                cy1 = cy0 + (y1 - y0)

                core = tile_result[cy0:cy1, cx0:cx1]
                output.paste(
                    Image.fromarray(core),
                    (x0, y0)
                )

                del ft, nt, ot, tile_result, core
                gc.collect()

            print(
                f"    {stem}: rows {min(y0 + TILE, height)}/{height}",
                flush=True
            )

        output.save(png_path, format="PNG")
        output.save(
            jpg_path,
            format="JPEG",
            quality=100,
            subsampling=0
        )
        output.close()

        del params
        gc.collect()

    print(f"[{index:02d}/{total:02d}] {stem}: complete")

    return {
        "scene": stem,
        "status": "ok",
        "png": str(png_path.relative_to(ROOT)),
        "jpg": str(jpg_path.relative_to(ROOT)),
        "width": width,
        "height": height,
    }


def main():
    with MANIFEST.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        groups = list(csv.DictReader(f))

    results = []

    for index, group in enumerate(groups, 1):
        result = process_scene(group, index, len(groups))
        results.append(result)
        gc.collect()

    REPORT.write_text(
        json.dumps({
            "status": "experiment_complete",
            "method": "full-resolution tiled adaptive shadow recovery",
            "tile_size": TILE,
            "overlap_halo": HALO,
            "colour_anchor": "normal exposure",
            "detail_restoration": "luminance-only",
            "input_scenes": len(groups),
            "outputs": results,
        }, indent=2),
        encoding="utf-8"
    )

    print(f"\nImages: {OUT}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
