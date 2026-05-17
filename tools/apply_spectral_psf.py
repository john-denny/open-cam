#!/usr/bin/env python3
"""Apply a spatial PSF to a pbrt SpectralFilm EXR (all float channels).

Intended as a post-render optics step (measured MTF / defocus approximation) after
PBRT.  Uses separable Gaussian blur (numpy only).

Two modes
---------
gaussian
    A single ``sigma_pixels`` is applied identically to every channel.  Legacy mode,
    kept for backward compatibility.

chromatic_gaussian
    Wavelength-dependent blur: for every spectral bucket (S0.*nm channels) the PSF
    width is computed as the quadrature sum of a constant geometric-aberration term and
    a diffraction-limited term that scales linearly with wavelength::

        σ_diff(λ) = 0.42 × N × λ_nm / (1000 × pixel_pitch_um)   [pixels]
        σ(λ)      = sqrt( σ_diff(λ)² + σ_geom² )

    where N is the f-number and ``σ_geom = sigma_geometric_pixels`` from the config.
    This models longitudinal chromatic aberration and the wavelength dependence of the
    Airy disk size.  Non-spectral (R/G/B) channels each receive a σ computed at a
    representative wavelength (620/540/460 nm respectively).

    ``f_number`` and ``pixel_pitch_um`` are read from the camera model
    (``sensor.f_number``, ``sensor.pixel_pitch_um``).  They may also be overridden
    directly in the ``post_psf`` config block.

See config/sensor_models/default.yaml (lens.post_psf) and README (optics section).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from camera_model import load_camera_model
from exr_multispectral import (
    parse_s0_wavelength_nm,
    read_separate_exr_channels,
    write_separate_channels_exr,
)

# Representative center wavelengths (nm) for broadband R/G/B channels.
_RGB_CENTER_NM: dict[str, float] = {"R": 620.0, "G": 540.0, "B": 460.0}


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.ones(1, dtype=np.float64)
    r = int(max(1, np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def separable_gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """HxW float — reflect-pad separable Gaussian."""
    if sigma <= 0:
        return np.asarray(img, dtype=np.float32, copy=False)
    k = _gaussian_kernel_1d(sigma)
    if k.size == 1:
        return np.asarray(img, dtype=np.float32, copy=False)
    pad = k.size // 2
    acc = np.asarray(img, dtype=np.float64)
    # horizontal
    x = np.pad(acc, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.empty_like(acc)
    for i in range(acc.shape[0]):
        tmp[i, :] = np.convolve(x[i, :], k, mode="valid")
    # vertical
    y = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(acc)
    for j in range(acc.shape[1]):
        out[:, j] = np.convolve(y[:, j], k, mode="valid")
    return out.astype(np.float32, copy=False)


def psf_sigma_chromatic(
    wavelength_nm: float,
    f_number: float,
    pixel_pitch_um: float,
    sigma_geometric_px: float,
) -> float:
    """Effective PSF σ (pixels) combining diffraction and geometric aberration.

    Airy disk first-zero radius converted to equivalent Gaussian σ:

        σ_diff(λ) = 0.42 × N × λ_nm / (1000 × pixel_pitch_um)

    Total PSF σ in quadrature (both components modelled as Gaussian):

        σ(λ) = sqrt( σ_diff(λ)² + σ_geom² )

    Parameters
    ----------
    wavelength_nm:
        Wavelength in nanometres.
    f_number:
        Lens f-number (N = f/D).
    pixel_pitch_um:
        Pixel pitch in micrometres.
    sigma_geometric_px:
        Constant geometric-aberration blur radius (defocus, coma …) in pixels.
    """
    sigma_diff = 0.42 * f_number * wavelength_nm / (1000.0 * pixel_pitch_um)
    return float(np.sqrt(sigma_diff**2 + sigma_geometric_px**2))


def apply_stray_light(arr: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply simple veiling glare + broad halo stray-light model."""
    if not bool(cfg.get("enabled", False)):
        return np.asarray(arr, dtype=np.float32, copy=False)

    out = np.asarray(arr, dtype=np.float32, copy=False)
    veiling = float(cfg.get("veiling_glare_fraction", 0.0))
    veiling = float(np.clip(veiling, 0.0, 1.0))
    if veiling > 0.0:
        mean_l = float(np.mean(out))
        out = (1.0 - veiling) * out + veiling * mean_l

    halo_sigma = float(cfg.get("halo_sigma_pixels", 0.0))
    halo_strength = float(cfg.get("halo_strength", 0.0))
    halo_strength = float(np.clip(halo_strength, 0.0, 1.0))
    if halo_sigma > 0.0 and halo_strength > 0.0:
        halo = separable_gaussian_blur_2d(out, halo_sigma)
        # Renormalize to avoid creating net scene energy.
        out = (out + halo_strength * halo) / (1.0 + halo_strength)

    return out.astype(np.float32, copy=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_default)
    ap.add_argument("--exr-in", type=Path, required=True, help="Input multispectral EXR.")
    ap.add_argument("--exr-out", type=Path, default=None, help="Output EXR (default: overwrite --exr-in).")
    ap.add_argument("--camera-model-config", type=Path, default=None, help="Camera model YAML path (preferred).")
    ap.add_argument("--config", type=Path, default=None, help="config/optics.yaml")
    args = ap.parse_args()

    repo = args.repo_root.resolve()

    camera_model: dict = {}
    if args.camera_model_config is not None:
        cfg_path = args.camera_model_config.resolve()
        if not cfg_path.is_file():
            print(f"error: missing {cfg_path}", file=sys.stderr)
            sys.exit(1)
        camera_model = load_camera_model(cfg_path)
        psf = (camera_model.get("lens", {}) or {}).get("post_psf", {}) or {}
    else:
        cfg_path = (args.config or (repo / "config" / "optics.yaml")).resolve()
        if not cfg_path.is_file():
            print(f"error: missing {cfg_path}", file=sys.stderr)
            sys.exit(1)
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        psf = cfg.get("post_psf", {}) or {}

    if not bool(psf.get("enabled", False)):
        print("post_psf.enabled is false; nothing to do.", file=sys.stderr)
        sys.exit(0)

    mode = str(psf.get("mode", "gaussian")).lower()
    if mode == "none":
        print('post_psf.mode is "none"; nothing to do.', file=sys.stderr)
        sys.exit(0)
    if mode not in ("gaussian", "chromatic_gaussian"):
        raise ValueError(
            f'post_psf.mode must be "gaussian", "chromatic_gaussian", or "none", got {mode!r}'
        )

    stray = psf.get("stray_light", {}) or {}

    exr_in = args.exr_in if args.exr_in.is_absolute() else (repo / args.exr_in).resolve()
    exr_out = (args.exr_out or exr_in).resolve()
    if not exr_in.is_file():
        raise FileNotFoundError(exr_in)

    chans = read_separate_exr_channels(exr_in)
    out: dict[str, np.ndarray] = {}

    if mode == "gaussian":
        sigma = float(psf.get("sigma_pixels", 0.0))
        if sigma < 0:
            raise ValueError("sigma_pixels must be >= 0")
        for name, arr in chans.items():
            blurred = separable_gaussian_blur_2d(arr, sigma)
            out[name] = apply_stray_light(blurred, stray)
        print(
            f"wrote {exr_out} (gaussian sigma_px={sigma}, "
            f"stray_light={'on' if bool(stray.get('enabled', False)) else 'off'})"
        )

    else:  # chromatic_gaussian
        sensor_cfg = camera_model.get("sensor", {}) or {}

        # f_number: psf config overrides sensor model.
        _fn_raw = psf.get("f_number", None)
        if _fn_raw is not None:
            f_number = float(_fn_raw)
        elif "f_number" in sensor_cfg:
            f_number = float(sensor_cfg["f_number"])
        else:
            raise KeyError(
                "chromatic_gaussian mode requires f_number. Set sensor.f_number in the "
                "camera model or add f_number to the lens.post_psf block."
            )

        # pixel_pitch_um: psf config overrides sensor model.
        _pp_raw = psf.get("pixel_pitch_um", None)
        if _pp_raw is not None:
            pixel_pitch_um = float(_pp_raw)
        elif "pixel_pitch_um" in sensor_cfg:
            pixel_pitch_um = float(sensor_cfg["pixel_pitch_um"])
        else:
            raise KeyError(
                "chromatic_gaussian mode requires pixel_pitch_um. Set sensor.pixel_pitch_um "
                "in the camera model or add pixel_pitch_um to the lens.post_psf block."
            )

        sigma_geom = float(psf.get("sigma_geometric_pixels", 0.0))
        if sigma_geom < 0:
            raise ValueError("sigma_geometric_pixels must be >= 0")

        sigmas_applied: list[tuple[str, float]] = []
        for name, arr in chans.items():
            lam_nm = parse_s0_wavelength_nm(name)
            if lam_nm is None:
                # Broadband R/G/B channel — use channel-typical center wavelength.
                lam_nm = _RGB_CENTER_NM.get(name.upper(), 550.0)
            sigma = psf_sigma_chromatic(lam_nm, f_number, pixel_pitch_um, sigma_geom)
            blurred = separable_gaussian_blur_2d(arr, sigma)
            out[name] = apply_stray_light(blurred, stray)
            sigmas_applied.append((name, sigma))

        # Summarize the σ(λ) range for spectral channels.
        spectral_sigmas = [s for n, s in sigmas_applied if parse_s0_wavelength_nm(n) is not None]
        if spectral_sigmas:
            print(
                f"wrote {exr_out} (chromatic_gaussian: f/{f_number}, "
                f"pitch={pixel_pitch_um}µm, σ_geom={sigma_geom}px, "
                f"σ(λ) range {min(spectral_sigmas):.3f}–{max(spectral_sigmas):.3f}px, "
                f"stray_light={'on' if bool(stray.get('enabled', False)) else 'off'})"
            )
        else:
            print(f"wrote {exr_out} (chromatic_gaussian: no spectral channels found, applied RGB mode)")

    exr_out.parent.mkdir(parents=True, exist_ok=True)
    write_separate_channels_exr(exr_out, out)


if __name__ == "__main__":
    main()
