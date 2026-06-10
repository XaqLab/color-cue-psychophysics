"""Render trial stimuli as base64-encoded PNG images (no display required)."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from color_cue.stimulus import get_cmap, get_theta_array


def render_trial_image(
    theta_left: float,
    theta_right: float,
    stimulus_kwargs: dict,
    rng: np.random.Generator,
    border_pixels: int = 14,
    gap_pixels: int = 28,
    scale: int = 3,
) -> str:
    """Render a two-panel trial image and return it as a base64 PNG string.

    Args:
        theta_left: Latent angle for the left stimulus.
        theta_right: Latent angle for the right stimulus.
        stimulus_kwargs: Keyword arguments forwarded to ``get_theta_array``.
        rng: NumPy random generator advanced in-place by the stimulus textures.
        border_pixels: White border width around each panel.
        gap_pixels: Width of the blank separator between panels.
        scale: Integer upscale factor applied with nearest-neighbour resampling.

    Returns:
        Base64-encoded PNG string suitable for embedding in an HTML ``<img>``
        ``src`` attribute as ``data:image/png;base64,<returned_value>``.
    """
    cmap = get_cmap()
    left_arr = get_theta_array(theta_left, rng=rng, **stimulus_kwargs)
    right_arr = get_theta_array(theta_right, rng=rng, **stimulus_kwargs)

    left_rgb = cmap(left_arr)[..., :3]
    right_rgb = cmap(right_arr)[..., :3]

    panel_h, panel_w = left_rgb.shape[:2]
    canvas_h = panel_h + 2 * border_pixels
    canvas_w = 2 * panel_w + 4 * border_pixels + gap_pixels
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=float)

    y0, y1 = border_pixels, border_pixels + panel_h
    lx0, lx1 = border_pixels, border_pixels + panel_w
    rx0 = lx1 + border_pixels + gap_pixels + border_pixels
    rx1 = rx0 + panel_w

    canvas[y0:y1, lx0:lx1] = left_rgb
    canvas[y0:y1, rx0:rx1] = right_rgb

    img = Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
