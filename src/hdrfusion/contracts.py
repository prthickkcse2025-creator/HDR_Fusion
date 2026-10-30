"""Safety contracts; this is not the full fusion pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class SceneFiles:
    scene_id: str
    under_raw: Path
    middle_raw: Path
    over_raw: Path
    target_jpeg: Path

def validate_scene_files(scene: SceneFiles) -> None:
    for field in ("under_raw", "middle_raw", "over_raw", "target_jpeg"):
        p = Path(getattr(scene, field))
        if not p.is_file():
            raise FileNotFoundError(f"{scene.scene_id}: missing {field}: {p}")

def assert_native_resolution(output_shape: Tuple[int, ...],
                             middle_shape: Tuple[int, ...]) -> None:
    if tuple(output_shape[:2]) != tuple(middle_shape[:2]):
        raise ValueError(f"Resolution mismatch: {output_shape[:2]} != {middle_shape[:2]}")

def assert_finite(image, label="image") -> None:
    if not np.isfinite(np.asarray(image)).all():
        raise ValueError(f"{label} contains NaN or Inf")
