from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
import csv
import json
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# HDR Fusion v2: scene-based train/validation/test, bounded RAM cache,
# AMP, validation-selected checkpoint, and native-resolution tiled output.
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "New_raws"
FINALS = ROOT / "stk"
MANIFEST = ROOT / "reports" / "new_dataset_manifest.csv"
OUT = ROOT / "outputs" / "new_dataset_fusion"
CKPT_DIR = ROOT / "checkpoints" / "new_dataset"
REPORT_DIR = ROOT / "reports"
BEST_CKPT = CKPT_DIR / "fusion_best.pt"
REPORT = REPORT_DIR / "new_dataset_fusion_training.json"

SEED = 42
PATCH = 256
PATCHES_PER_SCENE = 4
EPOCHS = 30
BATCH_SIZE = 4
LR = 2e-4
PATIENCE = 8
TILE = 512
OVERLAP = 128
CACHE_SCENES = 2       # bounded RAM use: cache at most two decoded scenes
JPEG_QUALITY = 95

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print(f"Device: {DEVICE}", flush=True)

def file_index(folder: Path):
    if not folder.exists():
        raise SystemExit(f"Required folder not found: {folder}")
    idx = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            idx[p.name.lower()] = p
            idx[p.stem.lower()] = p
    return idx

def find_column(headers, candidates):
    norm = {h.strip().lower().replace(" ", "_"): h for h in headers}
    for candidate in candidates:
        if candidate in norm:
            return norm[candidate]
    for h in headers:
        key = h.strip().lower().replace(" ", "_")
        if any(key.endswith("_" + candidate) for candidate in candidates):
            return h
    return None

def resolve_file(index, value):
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    name = Path(text).name.lower()
    return index.get(name) or index.get(Path(name).stem)

if not MANIFEST.exists():
    raise SystemExit(f"Manifest missing: {MANIFEST}")

raw_index = file_index(RAW)
target_index = file_index(FINALS)
with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames or []
    rows = list(reader)

columns = {
    "under": find_column(headers, ["under", "underexposed", "under_file", "under_path"]),
    "normal": find_column(headers, ["normal", "middle", "anchor", "normal_file", "middle_file"]),
    "over": find_column(headers, ["over", "overexposed", "over_file", "over_path"]),
    "target": find_column(headers, ["target", "final", "ground_truth", "target_file", "final_file"]),
}
missing = [key for key, value in columns.items() if value is None]
if missing:
    raise SystemExit(f"Cannot map manifest columns {missing}. Headers found: {headers}")

scenes, seen_targets = [], set()
unmatched = 0
for row in rows:
    scene = {
        "under": resolve_file(raw_index, row.get(columns["under"])),
        "normal": resolve_file(raw_index, row.get(columns["normal"])),
        "over": resolve_file(raw_index, row.get(columns["over"])),
        "target": resolve_file(target_index, row.get(columns["target"])),
    }
    if not all(scene.values()):
        unmatched += 1
        continue
    key = scene["target"].name.lower()
    if key not in seen_targets:
        scenes.append(scene)
        seen_targets.add(key)

if len(scenes) < 10:
    raise SystemExit(
        f"Only {len(scenes)} complete scenes matched ({unmatched} unmatched rows). "
        "No files were changed. Check the manifest mapping."
    )

print(
    f"Matched scenes: {len(scenes)} | unique Raw files: {len(set(raw_index.values()))} "
    f"| unique targets: {len(set(target_index.values()))} | unmatched manifest rows: {unmatched}",
    flush=True,
)

def load_rgb_uint8(path: Path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

def to_tensor_patch(array_uint8):
    # Convert only the selected crop to float, not the whole full-resolution image.
    arr = np.ascontiguousarray(array_uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)

def load_scene(scene):
    images = [load_rgb_uint8(scene[key]) for key in ("under", "normal", "over", "target")]
    shapes = [image.shape[:2] for image in images]
    if len(set(shapes)) != 1:
        raise ValueError(
            f"Dimension mismatch for target {scene['target'].name}: {shapes}. "
            "This script refuses to resize or warp the images."
        )
    return images

# Small LRU cache limits RAM use. Images remain uint8 until crops are selected.
SCENE_CACHE = OrderedDict()
def get_cached_scene(scene):
    key = scene["target"].name.lower()
    if key in SCENE_CACHE:
        SCENE_CACHE.move_to_end(key)
        return SCENE_CACHE[key]
    images = load_scene(scene)
    SCENE_CACHE[key] = images
    SCENE_CACHE.move_to_end(key)
    while len(SCENE_CACHE) > CACHE_SCENES:
        SCENE_CACHE.popitem(last=False)
    return images

# Deterministic split by scene prevents patches from one scene leaking across splits.
rng = random.Random(SEED)
indices = list(range(len(scenes)))
rng.shuffle(indices)
n_val = max(2, round(len(scenes) * 0.16))
n_test = max(1, round(len(scenes) * 0.12))
if n_val + n_test >= len(scenes):
    n_val, n_test = 2, 1
val_ids = set(indices[:n_val])
test_ids = set(indices[n_val:n_val + n_test])
train_ids = indices[n_val + n_test:]
train_scenes = [scenes[i] for i in train_ids]
val_scenes = [scenes[i] for i in val_ids]
test_scenes = [scenes[i] for i in indices[n_val:n_val + n_test]]
print(
    f"Split: train={len(train_scenes)}, validation={len(val_scenes)}, "
    f"held-out test={len(test_scenes)}",
    flush=True,
)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return F.relu(x + 0.2 * self.net(x), inplace=True)

class FusionNet(nn.Module):
    """Adaptive per-pixel exposure weights; never invents texture beyond input pixels."""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(12, 48, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResBlock(48), ResBlock(48), ResBlock(48), ResBlock(48),
            nn.Conv2d(48, 3, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        with torch.no_grad():
            self.body[-1].bias.copy_(torch.tensor([-1.25, 2.5, -1.25]))

    def forward(self, under, normal, over):
        def luminance(x):
            return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        features = torch.cat(
            [under, normal, over, luminance(under), luminance(normal), luminance(over)],
            dim=1,
        )
        weights = torch.softmax(self.body(self.head(features)), dim=1)
        stack = torch.stack([under, normal, over], dim=1)
        fused = (stack * weights.unsqueeze(2)).sum(dim=1)
        return fused, weights

def loss_fn(pred, target, normal):
    pixel = F.smooth_l1_loss(pred, target, beta=0.04)
    grad_x = F.l1_loss(
        pred[:, :, :, 1:] - pred[:, :, :, :-1],
        target[:, :, :, 1:] - target[:, :, :, :-1],
    )
    grad_y = F.l1_loss(
        pred[:, :, 1:, :] - pred[:, :, :-1, :],
        target[:, :, 1:, :] - target[:, :, :-1, :],
    )
    mean_color = F.l1_loss(pred.mean(dim=(2, 3)), target.mean(dim=(2, 3)))
    anchor = F.l1_loss(pred, normal)
    return pixel + 0.18 * (grad_x + grad_y) + 0.15 * mean_color + 0.04 * anchor

def random_patch(scene):
    images = get_cached_scene(scene)
    h, w = images[1].shape[:2]
    ph, pw = min(PATCH, h), min(PATCH, w)
    y = random.randint(0, h - ph)
    x = random.randint(0, w - pw)
    # Apply identical geometric augmentation to all bracket images and target.
    if random.random() < 0.5:
        images = [im[:, ::-1, :] for im in images]
    if random.random() < 0.5:
        images = [im[::-1, :, :] for im in images]
    return [to_tensor_patch(im[y:y + ph, x:x + pw]) for im in images]

def make_batches(scene_list, patches_per_scene):
    batch = []
    for scene in scene_list:
        for _ in range(patches_per_scene):
            batch.append(random_patch(scene))
            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []
    if batch:
        yield batch

def batch_to_device(batch):
    groups = [[item[i] for item in batch] for i in range(4)]
    return [
        torch.stack(group).to(DEVICE, non_blocking=True)
        for group in groups
    ]

def validation_loss(model, scene_list):
    model.eval()
    losses = []
    with torch.inference_mode():
        for index, scene in enumerate(scene_list, 1):
            # Load one validation scene at a time and use a repeatable center crop.
            images = load_scene(scene)
            h, w = images[1].shape[:2]
            ph, pw = min(PATCH, h), min(PATCH, w)
            y, x = (h - ph) // 2, (w - pw) // 2
            tensors = [
                to_tensor_patch(im[y:y + ph, x:x + pw]).unsqueeze(0).to(DEVICE)
                for im in images
            ]
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):
                pred, _ = model(tensors[0], tensors[1], tensors[2])
                loss = loss_fn(pred, tensors[3], tensors[1])
            losses.append(float(loss.item()))
            del images, tensors
    return float(np.mean(losses)) if losses else float("inf")

def save_checkpoint(model, epoch, val_loss, history):
    RECOVERY_DIR = ROOT / "checkpoints" / "new_dataset_recovery"
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_loss": float(val_loss),
            "seed": SEED,
            "patch_size": PATCH,
            "architecture": "FusionNet-48ch-4-resblocks",
        },
        RECOVERY_DIR / "fusion_best.pt",
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "status": "training_in_progress",
                "device": str(DEVICE),
                "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
                "matched_scenes": len(scenes),
                "train_scenes": len(train_scenes),
                "validation_scenes": len(val_scenes),
                "held_out_test_scenes": len(test_scenes),
                "patch_size": PATCH,
                "patches_per_scene_per_epoch": PATCHES_PER_SCENE,
                "batch_size": BATCH_SIZE,
                "epoch_limit": EPOCHS,
                "best_epoch": epoch,
                "best_validation_loss": float(val_loss),
                "history": history,
            },
            indent=2,
        )
    )

def train():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # Resume weights from the saved new-dataset Epoch 1 checkpoint.
    RESUME_CKPT = CKPT_DIR / "fusion_best.pt"
    RECOVERY_DIR = ROOT / "checkpoints" / "new_dataset_recovery"
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)

    model = FusionNet().to(DEVICE)
    resume_epoch = 0
    resume_val_loss = float("inf")
    if RESUME_CKPT.exists():
        saved = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved["model"])
        resume_epoch = int(saved.get("epoch", 0))
        resume_val_loss = float(saved.get("val_loss", float("inf")))
        print(
            f"Resuming new-dataset weights from epoch {resume_epoch}; "
            f"validation loss={resume_val_loss:.6f}",
            flush=True,
        )
        baseline_dir = ROOT / "checkpoints" / "new_dataset_recovery"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = baseline_dir / "fusion_best.pt"
        if not baseline_path.exists():
            torch.save(saved, baseline_path)
            print("Saved Epoch 1 baseline to recovery directory.", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    best = resume_val_loss
    stale = 0
    history = []
    if REPORT.exists():
        try:
            previous_report = json.loads(REPORT.read_text())
            previous_history = previous_report.get("history", [])
            if previous_history:
                history = previous_history
        except (OSError, ValueError, TypeError):
            pass
    started = time.time()

    # Epoch 1 history is retained in the backup report; continue from Epoch 2.
    for epoch in range(resume_epoch + 1, EPOCHS + 1):
        model.train()
        train_losses = []
        random.shuffle(train_scenes)
        total_patches = len(train_scenes) * PATCHES_PER_SCENE
        total_batches = (total_patches + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nEpoch {epoch:02d}/{EPOCHS}: {total_batches} batches", flush=True)

        # Stream batches instead of retaining every patch tensor in RAM.
        for batch_index, batch in enumerate(make_batches(train_scenes, PATCHES_PER_SCENE), 1):
            u, n, o, target = batch_to_device(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):
                pred, _ = model(u, n, o)
                loss = loss_fn(pred, target, n)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().item()))

            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                print(
                    f"  batch {batch_index:03d}/{total_batches} "
                    f"| loss={train_losses[-1]:.5f}",
                    flush=True,
                )
            del u, n, o, target, pred, loss, batch

        # Free cached training scenes before validation, keeping RAM predictable.
        SCENE_CACHE.clear()
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = validation_loss(model, val_scenes)
        SCENE_CACHE.clear()
        scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} complete | train={train_loss:.5f} "
            f"| val={val_loss:.5f} | lr={row['lr']:.2g}",
            flush=True,
        )

        if val_loss < best - 1e-5:
            best = val_loss
            stale = 0
            save_checkpoint(model, epoch, best, history)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": float(best),
                    "seed": SEED,
                    "patch_size": PATCH,
                    "architecture": "FusionNet-48ch-4-resblocks",
                },
                RECOVERY_DIR / "fusion_recovery_latest.pt",
            )
            print(f"  Saved best checkpoint (validation loss {best:.5f})", flush=True)
        else:
            stale += 1
            print(f"  No validation improvement: {stale}/{PATIENCE}", flush=True)
            # Save progress report even when checkpoint does not improve.
            REPORT.write_text(
                json.dumps(
                    {
                        "status": "training_in_progress",
                        "device": str(DEVICE),
                        "matched_scenes": len(scenes),
                        "train_scenes": len(train_scenes),
                        "validation_scenes": len(val_scenes),
                        "held_out_test_scenes": len(test_scenes),
                        "history": history,
                        "best_epoch": int(np.argmin([r["val_loss"] for r in history]) + 1),
                        "best_validation_loss": best,
                    },
                    indent=2,
                )
            )
            if stale >= PATIENCE:
                print("Early stopping: validation loss did not improve.", flush=True)
                break

    RECOVERY_BEST = ROOT / "checkpoints" / "new_dataset_recovery" / "fusion_best.pt"
    if not RECOVERY_BEST.exists():
        raise RuntimeError("Training ended without a recovery best checkpoint.")
    print(f"\nTraining finished in {(time.time() - started) / 60:.1f} minutes.", flush=True)
    checkpoint = torch.load(RECOVERY_BEST, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Held-out evaluation uses a fixed center crop; it is reported, not used to tune.
    test_loss = validation_loss(model, test_scenes) if test_scenes else None
    SCENE_CACHE.clear()

    # Full-resolution tiled inference, one scene at a time.
    output_rows = []
    print(f"\nGenerating full-resolution outputs in {OUT}", flush=True)
    for scene_number, scene in enumerate(scenes, 1):
        images = load_scene(scene)
        u8, n8, o8, _target8 = images
        if not (u8.shape == n8.shape == o8.shape):
            print(f"Skipping dimension mismatch: {scene['target'].name}", flush=True)
            continue
        fused = infer_tiled(model, u8, n8, o8)
        stem = scene["target"].stem
        jpg_path = OUT / f"{stem}_fusion.jpg"
        tif_path = OUT / f"{stem}_fusion_16bit.tif"
        Image.fromarray(np.round(fused * 255).astype(np.uint8), mode="RGB").save(
            jpg_path, quality=JPEG_QUALITY, subsampling=0
        )
        # 16-bit TIFF container from JPEG-derived values; not true RAW precision.
        import tifffile
        tifffile.imwrite(
            tif_path,
            np.round(fused * 65535).astype(np.uint16),
            photometric="rgb",
        )
        output_rows.append(
            {
                "scene": scene["target"].name,
                "height": int(fused.shape[0]),
                "width": int(fused.shape[1]),
                "jpeg": str(jpg_path.relative_to(ROOT)),
                "tiff16": str(tif_path.relative_to(ROOT)),
            }
        )
        print(f"  [{scene_number:02d}/{len(scenes)}] {stem}", flush=True)
        del images, fused

    final_report = {
        "status": "complete",
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "matched_scenes": len(scenes),
        "train_scenes": len(train_scenes),
        "validation_scenes": len(val_scenes),
        "held_out_test_scenes": len(test_scenes),
        "best_checkpoint": str(RECOVERY_BEST.relative_to(ROOT)),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "best_validation_loss": float(checkpoint.get("val_loss", best)),
        "held_out_test_center_crop_loss": test_loss,
        "history": history,
        "outputs_generated": len(output_rows),
        "outputs": output_rows,
        "notes": [
            "Training and validation used scene-level splits.",
            "Test metric is a fixed center-crop loss, not a full-image benchmark.",
            "TIFF16 files are converted from JPEG-derived values and do not restore sensor RAW precision.",
            "Visual review is still required before calling images production-ready.",
        ],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    REPORT.write_text(json.dumps(final_report, indent=2))
    print("\nTRAINING AND INFERENCE COMPLETE", flush=True)
    print(f"Best checkpoint: {BEST_CKPT}", flush=True)
    print(f"Outputs generated: {len(output_rows)}", flush=True)
    print(f"Training report: {REPORT}", flush=True)

def infer_tiled(model, under_u8, normal_u8, over_u8):
    h, w = normal_u8.shape[:2]
    accum = np.zeros((h, w, 3), dtype=np.float32)
    counts = np.zeros((h, w, 1), dtype=np.float32)
    stride = max(1, TILE - OVERLAP)
    ys = list(range(0, max(h - TILE, 0) + 1, stride))
    xs = list(range(0, max(w - TILE, 0) + 1, stride))
    if not ys or ys[-1] != max(h - TILE, 0):
        ys.append(max(h - TILE, 0))
    if not xs or xs[-1] != max(w - TILE, 0):
        xs.append(max(w - TILE, 0))

    model.eval()
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                y2, x2 = min(y + TILE, h), min(x + TILE, w)
                tensors = []
                for arr in (under_u8, normal_u8, over_u8):
                    crop = to_tensor_patch(arr[y:y2, x:x2]).unsqueeze(0).to(DEVICE)
                    tensors.append(crop)
                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=torch.float16,
                    enabled=(DEVICE.type == "cuda"),
                ):
                    pred, _ = model(*tensors)
                tile = pred[0].float().permute(1, 2, 0).cpu().numpy()
                accum[y:y2, x:x2] += tile
                counts[y:y2, x:x2] += 1.0
                del tensors, pred, tile
    return np.clip(accum / np.maximum(counts, 1e-6), 0.0, 1.0)

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nInterrupted by user. No claim of completion; any existing best checkpoint was left intact.", flush=True)
        raise
