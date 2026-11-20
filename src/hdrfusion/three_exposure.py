"""Constrained three-exposure fusion for bracketed interior photographs.

The middle exposure is the anchor. The bright frame may contribute only to
underexposed regions, and the dark frame may contribute only to clipped or
near-clipped highlights. This is deliberately a deterministic fusion model:
it does not learn a room-specific appearance from the reference image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class FusionSettings:
    """Optional overrides; ``None`` selects per-bracket adaptive values."""

    shadow_strength: float | None = None
    highlight_strength: float | None = None
    shadow_start: float | None = None
    shadow_full: float | None = None
    highlight_start: float | None = None
    highlight_full: float | None = None
    mask_blur_fraction: float = 0.0012
    align: bool = True


RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}


def _read_rgb(path: str | Path, user_wb: list[float] | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError("RAW input needs rawpy; install the project extra with `pip install .[raw]`.") from exc
        with rawpy.imread(str(path)) as raw:
            wb = user_wb
            if wb is None:
                wb = raw.camera_whitebalance
            if not wb or len(wb) < 3:
                raise ValueError(f"Could not read a camera white balance from {path.name}")
            rgb = raw.postprocess(
                user_wb=wb,
                use_camera_wb=False,
                no_auto_bright=True,
                output_bps=16,
                gamma=(2.222, 4.5),
            )
        return np.asarray(rgb, dtype=np.uint16)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode == "I;16" or im.mode == "I;16B" or im.mode == "I;16L":
            gray = np.asarray(im, dtype=np.uint16)
            return np.repeat(gray[..., None], 3, axis=2)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luma_srgb(rgb: np.ndarray) -> np.ndarray:
    # Display-referred luma is used for exposure selection; color fusion below
    # is performed in linear light.
    f = rgb.astype(np.float32) / float(np.iinfo(rgb.dtype).max)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def _registration_view(rgb: np.ndarray, max_side: int = 1200) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / float(np.iinfo(rgb.dtype).max)
    # Log compression makes ECC less sensitive to the exposure difference.
    gray = np.log1p(8.0 * gray) / np.log(9.0)
    h, w = gray.shape
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return gray


def _align_to_normal(source: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """ECC-register a bracket to normal; return aligned RGB, valid mask, score."""
    template = _registration_view(normal)
    moving = _registration_view(source)
    h, w = template.shape
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 120, 1e-6)
    try:
        score, warp_small = cv2.findTransformECC(
            template, moving, warp, cv2.MOTION_EUCLIDEAN, criteria,
            None, 5,
        )
        # Convert the small-view translation back to full-resolution pixels.
        scale = normal.shape[1] / w
        warp_full = warp_small.copy()
        warp_full[:, 2] *= scale
        aligned = cv2.warpAffine(
            source, warp_full, (normal.shape[1], normal.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        valid = cv2.warpAffine(
            np.ones(normal.shape[:2], dtype=np.uint8), warp_full,
            (normal.shape[1], normal.shape[0]),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(np.float32)
        if not np.isfinite(score) or score < 0.50:
            # Uncertain registration must not introduce alternate-frame ghosts.
            return source, np.zeros(normal.shape[:2], dtype=np.float32), float(score)
        return aligned, valid, float(score)
    except cv2.error:
        return source, np.zeros(normal.shape[:2], dtype=np.float32), 0.0


def _blur_mask(mask: np.ndarray, fraction: float) -> np.ndarray:
    radius = max(1, int(round(min(mask.shape) * fraction)))
    # Gaussian blur feathers mask transitions without smoothing image detail.
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=radius, sigmaY=radius)


def _gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    pyramid = [image.astype(np.float32, copy=False)]
    for _ in range(levels):
        h, w = pyramid[-1].shape[:2]
        if min(h, w) < 24:
            break
        down = cv2.pyrDown(pyramid[-1])
        if pyramid[-1].ndim == 3 and down.ndim == 2:
            down = down[..., None]
        pyramid.append(down)
    return pyramid


def _laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    gaussian = _gaussian_pyramid(image, levels)
    laplacian: list[np.ndarray] = []
    for i in range(len(gaussian) - 1):
        h, w = gaussian[i].shape[:2]
        up = cv2.pyrUp(gaussian[i + 1], dstsize=(w, h))
        laplacian.append(gaussian[i] - up)
    laplacian.append(gaussian[-1])
    return laplacian


def _multiscale_blend(
    normal: np.ndarray,
    bright: np.ndarray,
    dark: np.ndarray,
    shadow_weight: np.ndarray,
    highlight_weight: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Blend exposure detail across scales to avoid visible mask transitions."""
    h, w = normal.shape[:2]
    levels = min(4, max(1, int(np.floor(np.log2(min(h, w) / 24.0)))))
    normal_weight = np.clip(1.0 - shadow_weight - highlight_weight, 0.0, 1.0)
    weight_pyramids = [
        _gaussian_pyramid(weight[..., None], levels)
        for weight in (normal_weight, shadow_weight, highlight_weight)
    ]
    image_pyramids = [
        _laplacian_pyramid(image, levels)
        for image in (normal, bright, dark)
    ]

    blended_levels: list[np.ndarray] = []
    for level in range(len(image_pyramids[0])):
        weights = [p[min(level, len(p) - 1)] for p in weight_pyramids]
        total = np.maximum(weights[0] + weights[1] + weights[2], 1e-6)
        weights = [weight / total for weight in weights]
        blended_levels.append(
            image_pyramids[0][level] * weights[0]
            + image_pyramids[1][level] * weights[1]
            + image_pyramids[2][level] * weights[2]
        )

    fused = blended_levels[-1]
    for level in range(len(blended_levels) - 2, -1, -1):
        lh, lw = blended_levels[level].shape[:2]
        fused = cv2.pyrUp(fused, dstsize=(lw, lh)) + blended_levels[level]
    return np.clip(fused, 0.0, 1.0), len(blended_levels) - 1


def fuse_three_exposures(
    bright: np.ndarray,
    dark: np.ndarray,
    normal: np.ndarray,
    settings: FusionSettings | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fuse uint8 RGB exposures in bright, dark, normal order.

    Returns an RGB uint8 image and alignment diagnostics. Correctly exposed
    pixels remain the normal exposure; alternate inputs are confidence-gated.
    """
    settings = settings or FusionSettings()
    for name, image in (("bright", bright), ("dark", dark), ("normal", normal)):
        if image.dtype not in (np.uint8, np.uint16) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} must be an 8-bit or 16-bit RGB image")
        if image.dtype != normal.dtype:
            raise ValueError("All three exposures must have the same bit depth")
        if image.shape != normal.shape:
            raise ValueError(f"{name} dimensions {image.shape} do not match normal {normal.shape}")

    diagnostics: dict[str, float] = {}
    bright_valid = np.ones(normal.shape[:2], dtype=np.float32)
    dark_valid = np.ones(normal.shape[:2], dtype=np.float32)
    if settings.align:
        bright, bright_valid, diagnostics["bright_alignment_score"] = _align_to_normal(bright, normal)
        dark, dark_valid, diagnostics["dark_alignment_score"] = _align_to_normal(dark, normal)
    else:
        diagnostics.update(bright_alignment_score=1.0, dark_alignment_score=1.0)

    y_mid = _luma_srgb(normal)
    y_bright = _luma_srgb(bright)
    y_dark = _luma_srgb(dark)

    # Recompute the tonal gates for every bracket. Percentiles adapt the
    # shadow/highlight zones to this scene's histogram instead of a fixed EV.
    p04, p08, p38, p62, p92, p96 = np.percentile(y_mid, [4, 8, 38, 62, 92, 96])
    scene_range = max(float(p96 - p04), 0.04)

    shadow_low = float(settings.shadow_start if settings.shadow_start is not None else p08)
    shadow_high = float(settings.shadow_full if settings.shadow_full is not None else p38)
    highlight_low = float(settings.highlight_start if settings.highlight_start is not None else p62)
    highlight_high = float(settings.highlight_full if settings.highlight_full is not None else p92)
    min_gate_width = max(scene_range * 0.08, 0.018)
    shadow_high = max(shadow_high, shadow_low + min_gate_width)
    highlight_high = max(highlight_high, highlight_low + min_gate_width)
    shadow_high = min(shadow_high, 0.995)
    highlight_high = min(highlight_high, 0.999)

    # Estimate the exposure separation from this set's median luminance. Bigger
    # brackets can contribute more; flat/narrow-range scenes get gentler blending.
    med_mid = float(np.median(y_mid))
    med_bright = float(np.median(y_bright))
    med_dark = float(np.median(y_dark))
    shadow_ev = max(0.0, float(np.log2((med_bright + 0.01) / (med_mid + 0.01))))
    highlight_ev = max(0.0, float(np.log2((med_mid + 0.01) / (med_dark + 0.01))))
    scene_factor = float(np.clip(scene_range / 0.72, 0.60, 1.05))
    shadow_strength = settings.shadow_strength
    if shadow_strength is None:
        shadow_strength = float(np.clip(0.18 + 0.12 * min(shadow_ev, 3.0), 0.16, 0.58) * scene_factor)
    highlight_strength = settings.highlight_strength
    if highlight_strength is None:
        highlight_strength = float(np.clip(0.18 + 0.12 * min(highlight_ev, 3.0), 0.16, 0.58) * scene_factor)

    # Source-reliability gates are also relative to each alternate frame's
    # signal range. Saturated or nearly black source pixels receive no weight.
    b02, b12, b97 = np.percentile(y_bright, [2, 12, 97])
    d02, d12, d97 = np.percentile(y_dark, [2, 12, 97])
    bright_range = max(float(np.percentile(y_bright, 98) - b02), 0.04)
    dark_range = max(float(np.percentile(y_dark, 98) - d02), 0.04)
    bright_signal = _smoothstep(float(b02), max(float(b12), float(b02) + 0.06 * bright_range), y_bright)
    dark_signal = _smoothstep(float(d02), max(float(d12), float(d02) + 0.06 * dark_range), y_dark)
    bright_clip_start = float(np.clip(max(float(b97), 0.95), 0.95, 0.997))
    dark_clip_start = float(np.clip(max(float(d97), 0.95), 0.95, 0.997))
    bright_not_clipped = 1.0 - _smoothstep(bright_clip_start, min(1.0, bright_clip_start + 0.025), y_bright)
    dark_not_clipped = 1.0 - _smoothstep(dark_clip_start, min(1.0, dark_clip_start + 0.025), y_dark)

    # Bright contributes only to this image's darker tail; dark contributes
    # only to its brighter tail. Normal remains the anchor everywhere else.
    shadow_need = 1.0 - _smoothstep(shadow_low, shadow_high, y_mid)
    shadow_alpha = shadow_strength * shadow_need * bright_signal * bright_not_clipped * bright_valid
    highlight_need = _smoothstep(highlight_low, highlight_high, y_mid)
    highlight_alpha = highlight_strength * highlight_need * dark_signal * dark_not_clipped * dark_valid

    shadow_alpha = np.clip(_blur_mask(shadow_alpha.astype(np.float32), settings.mask_blur_fraction), 0.0, 0.82)
    highlight_alpha = np.clip(_blur_mask(highlight_alpha.astype(np.float32), settings.mask_blur_fraction), 0.0, 0.82)

    # Decode to linear light before blending. Per-pixel exposure weights are
    # then fused with Gaussian/Laplacian pyramids so transitions remain smooth
    # around window frames and other high-contrast edges.
    max_value = float(np.iinfo(normal.dtype).max)
    encoded_lut = np.arange(int(max_value) + 1, dtype=np.float32) / max_value
    lut = np.where(
        encoded_lut <= 0.04045,
        encoded_lut / 12.92,
        ((encoded_lut + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)
    n_linear = lut[normal]
    b_linear = lut[bright]
    d_linear = lut[dark]
    fused_linear, pyramid_levels = _multiscale_blend(
        n_linear, b_linear, d_linear, shadow_alpha, highlight_alpha
    )
    del n_linear, b_linear, d_linear
    encoded = np.where(
        fused_linear <= 0.0031308,
        fused_linear * 12.92,
        1.055 * np.power(fused_linear, 1.0 / 2.4) - 0.055,
    )
    out = np.round(np.clip(encoded, 0.0, 1.0) * max_value).astype(normal.dtype)

    diagnostics["shadow_contribution_mean"] = float(shadow_alpha.mean())
    diagnostics["highlight_contribution_mean"] = float(highlight_alpha.mean())
    diagnostics["normal_preserved_fraction"] = float(np.mean((shadow_alpha + highlight_alpha) < 0.01))
    diagnostics.update(
        shadow_threshold_low=shadow_low,
        shadow_threshold_high=shadow_high,
        highlight_threshold_low=highlight_low,
        highlight_threshold_high=highlight_high,
        shadow_strength=float(shadow_strength),
        highlight_strength=float(highlight_strength),
        shadow_ev=shadow_ev,
        highlight_ev=highlight_ev,
        normal_luma_p08=float(p08),
        normal_luma_p92=float(p92),
        multiscale_levels=float(pyramid_levels),
    )
    return out, diagnostics


def fuse_files(
    bright_path: str | Path,
    dark_path: str | Path,
    normal_path: str | Path,
    output_path: str | Path,
    settings: FusionSettings | None = None,
) -> dict[str, float]:
    """Read three images with shared middle-frame WB for RAW, fuse, and save."""
    normal_path = Path(normal_path)
    shared_wb = None
    if normal_path.suffix.lower() in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError("RAW input needs rawpy; install the project extra with `pip install .[raw]`.") from exc
        with rawpy.imread(str(normal_path)) as raw:
            shared_wb = raw.camera_whitebalance
    bright = _read_rgb(bright_path, shared_wb)
    dark = _read_rgb(dark_path, shared_wb)
    normal = _read_rgb(normal_path, shared_wb)
    result, diagnostics = fuse_three_exposures(bright, dark, normal, settings)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        if result.dtype == np.uint16:
            result = (result.astype(np.uint32) >> 8).astype(np.uint8)
        Image.fromarray(result, mode="RGB").save(destination, quality=96, subsampling=0)
    elif result.dtype == np.uint16:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError("16-bit TIFF output needs tifffile; install it with `pip install tifffile`.") from exc
        tifffile.imwrite(destination, result, photometric="rgb")
    else:
        Image.fromarray(result, mode="RGB").save(destination)
    return diagnostics
