import csv
from importlib import resources
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from numpy.fft import fftfreq, ifft2
from numpy.typing import ArrayLike

CMAP_PTH = resources.files("color_cue").joinpath("data/cmap.csv")


def cue_to_theta(
    cue: float | ArrayLike,
    theta_min: float = -np.pi / 2,
    theta_max: float = 0,
) -> float | ArrayLike:
    """Map normalized cue values to latent stimulus angles.

    Args:
        cue: Scalar or array of cue values in ``[0, 1]``.
        theta_min: Lower endpoint of the hue-angle arc.
        theta_max: Upper endpoint of the hue-angle arc.

    Returns:
        The latent angle or angles corresponding to ``cue``. A Python ``float``
        is returned for scalar inputs; otherwise a NumPy array is returned.
    """
    cue = np.asarray(cue)
    theta = cue * (theta_max - theta_min) + theta_min
    if theta.ndim == 0:
        return float(theta)
    return theta


def theta_to_cue(
    theta: float | ArrayLike,
    theta_min: float = -np.pi / 2,
    theta_max: float = 0,
    clip: bool = True,
) -> float | ArrayLike:
    """Map latent stimulus angles back to normalized cue values.

    Args:
        theta: Scalar or array of latent hue angles.
        theta_min: Lower endpoint of the hue-angle arc.
        theta_max: Upper endpoint of the hue-angle arc.
        clip: Whether to clip ``theta`` into ``[theta_min, theta_max]`` before
            applying the inverse mapping.

    Returns:
        The normalized cue value or values in ``[0, 1]``. A Python ``float`` is
        returned for scalar inputs; otherwise a NumPy array is returned.
    """
    theta = np.asarray(theta)
    if clip:
        theta = np.clip(theta, theta_min, theta_max)
    cue = (theta - theta_min) / (theta_max - theta_min)
    if cue.ndim == 0:
        return float(cue)
    return cue


def get_cmap() -> ListedColormap:
    """Load the custom color map used for the cue stimulus.

    Returns:
        A ``ListedColormap`` built from the RGB rows stored in ``cmap.csv``.
        The resulting colormap maps values in ``[0, 1]`` to RGBA colors.
    """
    colors = []
    with resources.as_file(CMAP_PTH) as cmap_path:
        with open(cmap_path, newline="") as f:
            reader = csv.reader(f)
            for line in reader:
                colors.append([int(v) for v in line])
    cmap = ListedColormap(np.array(colors).astype(float) / 255)
    return cmap


def get_cue_movie(
    cue: float,
    num_steps: int,
    size: tuple[int, int] = (128, 96),
    kappa: float = 0.1,
    alpha: float = 1.0,
    beta: float = -2.0,
    theta_min: float = -np.pi / 2,
    theta_max: float = 0,
    rng: np.random.Generator | int | None = None,
) -> ArrayLike:
    """Generate a temporally correlated color-cue movie from a cue value.

    The movie is built by sampling a spatially correlated complex noise field in
    the Fourier domain, adding a deterministic signal vector of magnitude
    ``kappa`` at the latent hue angle implied by ``cue``, then taking the pixel
    angle at each frame.

    Args:
        cue: Normalized cue value in ``[0, 1]``. Values nearer ``theta_max`` are
            redder on the current color arc, and values nearer ``theta_min`` are
            bluer.
        num_steps: Number of frames to generate.
        size: ``(height, width)`` of each movie frame.
        kappa: Signal magnitude in the complex plane. Larger values produce a
            stronger global hue drive relative to the texture noise.
        alpha: Temporal-correlation parameter. Larger values make high-frequency
            Fourier components decorrelate more quickly across frames.
        beta: Spatial power-spectrum exponent for the Fourier-domain noise.
        theta_min: Lower endpoint of the hue-angle arc.
        theta_max: Upper endpoint of the hue-angle arc.
        rng: Optional NumPy random generator or seed.

    Returns:
        A NumPy array of shape ``(num_steps, height, width)`` whose values lie in
        ``[0, 1)`` and index into the custom cue colormap.
    """
    rng = np.random.default_rng(rng)

    fx, fy = np.meshgrid(fftfreq(size[1]), fftfreq(size[0]))
    f = (fx**2 + fy**2) ** 0.5
    gamma = np.exp(-alpha * f)
    eps = 1e-5 / max(size)
    S_f = (f + eps) ** beta
    S_f[0, 0] = 0.0

    theta = cue * (theta_max - theta_min) + theta_min
    noise, values = None, []
    for _ in range(num_steps):
        _noise = (rng.normal(size=size) + 1j * rng.normal(size=size)) * S_f**0.5
        if noise is None:
            noise = _noise
        else:
            noise = gamma * noise + (1 - gamma) * _noise
        x_comp = ifft2(noise) + kappa * np.exp(1j * theta)
        x_circ = np.angle(x_comp)
        x_circ = 0.5 * x_circ / np.pi + (x_circ <= 0).astype(float)
        values.append(x_circ)
    values = np.stack(values)
    return values


def get_theta_movie(
    theta: float,
    num_steps: int,
    theta_min: float = -np.pi / 2,
    theta_max: float = 0,
    **kwargs,
) -> ArrayLike:
    """Generate a color-cue movie directly from a latent hue angle.

    Args:
        theta: Latent hue angle to render.
        num_steps: Number of frames to generate.
        theta_min: Lower endpoint of the hue-angle arc.
        theta_max: Upper endpoint of the hue-angle arc.
        **kwargs: Additional keyword arguments forwarded to ``get_cue_movie``.

    Returns:
        A NumPy array of shape ``(num_steps, height, width)`` containing the cue
        movie frames.
    """
    cue = theta_to_cue(theta, theta_min=theta_min, theta_max=theta_max, clip=True)
    return get_cue_movie(
        cue,
        num_steps,
        theta_min=theta_min,
        theta_max=theta_max,
        **kwargs,
    )


def get_cue_array(
    cue: float,
    **kwargs,
) -> ArrayLike:
    """Generate a single-frame color-cue image from a normalized cue value.

    Args:
        cue: Normalized cue value in ``[0, 1]``.
        **kwargs: Additional keyword arguments forwarded to ``get_cue_movie``.

    Returns:
        A NumPy array of shape ``(height, width)`` corresponding to the first
        frame of the generated cue movie.
    """
    values = get_cue_movie(cue, 1, **kwargs)[0]
    return values


def get_theta_array(
    theta: float,
    theta_min: float = -np.pi / 2,
    theta_max: float = 0,
    **kwargs,
) -> ArrayLike:
    """Generate a single-frame color-cue image directly from a latent angle.

    Args:
        theta: Latent hue angle to render.
        theta_min: Lower endpoint of the hue-angle arc.
        theta_max: Upper endpoint of the hue-angle arc.
        **kwargs: Additional keyword arguments forwarded to ``get_theta_movie``.

    Returns:
        A NumPy array of shape ``(height, width)`` containing one rendered cue
        image.
    """
    return get_theta_movie(
        theta,
        1,
        theta_min=theta_min,
        theta_max=theta_max,
        **kwargs,
    )[0]
