# Algorithm notes

## Objective

Produce a natural-looking result anchored to the normal exposure. Use the brighter image only to reveal shadow detail and the darker image only to preserve highlight detail. Fusion decisions are calculated independently for each image triplet. There is no training phase, room classifier, or learned per-room parameter.

## Processing stages

1. **Read and validate.** Decode RGB images (including optional RAW support), apply EXIF orientation, and require matching dimensions and bit depth. For RAW brackets, render with the normal frame's camera white balance.
2. **Register.** Estimate a small Euclidean warp from each alternate exposure to the normal image using exposure-compressed grayscale. If the registration confidence is too low, the corresponding alternate exposure contributes zero weight.
3. **Estimate scene controls.** Luminance percentiles in the normal frame set the shadow and highlight gates. Median luminance differences estimate bracket separation. Source luminance percentiles suppress near-black or clipped source pixels. Blend strengths are therefore image adaptive; optional CLI overrides exist for manual control.
4. **Build contribution masks.** A smooth shadow mask gates the bright frame; a smooth highlight mask gates the dark frame. Both are softened spatially to avoid hard seams. The remaining weight belongs to the normal frame.
5. **Fuse.** Convert to linear light, build Gaussian weight pyramids and Laplacian image pyramids, blend at each scale, and reconstruct the output.
6. **Write diagnostics.** Return registration scores, adaptive gates, blend strengths, mean alternate-frame contributions, and normal-frame preserved fraction.

## Confidence and limits

The normal image is the anchor throughout. Confidence-gated registration reduces, but does not eliminate, ghosting. No method can restore texture missing from all exposures. Large parallax, moving content, extreme exposure gaps, and colored or flickering light can still cause errors.
