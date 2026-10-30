"""Native-coordinate overlap-add utilities; never resize the input or output."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Tile:
    y0: int
    y1: int
    x0: int
    x1: int

def make_tiles(height: int, width: int, tile_size: int = 512, overlap: int = 128):
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > overlap >= 0")
    stride = tile_size - overlap
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not ys or ys[-1] + tile_size < height: ys.append(max(0, height - tile_size))
    if not xs or xs[-1] + tile_size < width: xs.append(max(0, width - tile_size))
    ys, xs = list(dict.fromkeys(ys)), list(dict.fromkeys(xs))
    return [Tile(y, min(y + tile_size, height), x, min(x + tile_size, width))
            for y in ys for x in xs]

def smooth_window(h, w, floor=1e-3):
    def axis(n):
        return np.ones(1, np.float32) if n <= 1 else np.maximum(
            np.sin(np.linspace(0, np.pi, n, dtype=np.float32)), floor)
    return axis(h)[:, None] * axis(w)[None, :]

def overlap_add(tile_outputs, tiles, output_shape):
    h, w = output_shape[:2]
    is_color = len(output_shape) == 3
    accum = np.zeros(output_shape, np.float32)
    weights = np.zeros((h, w), np.float32)
    for tile_image, tile in zip(tile_outputs, tiles):
        expected = (tile.y1 - tile.y0, tile.x1 - tile.x0)
        if tuple(tile_image.shape[:2]) != expected:
            raise ValueError(f"Unexpected tile shape {tile_image.shape[:2]}, expected {expected}")
        win = smooth_window(*expected)
        if is_color:
            accum[tile.y0:tile.y1, tile.x0:tile.x1] += tile_image.astype(np.float32) * win[..., None]
        else:
            accum[tile.y0:tile.y1, tile.x0:tile.x1] += tile_image.astype(np.float32) * win
        weights[tile.y0:tile.y1, tile.x0:tile.x1] += win
    if np.any(weights <= 0):
        raise ValueError("At least one output pixel has no tile coverage")
    return accum / (weights[..., None] if is_color else weights)
