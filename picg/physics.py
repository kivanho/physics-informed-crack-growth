"""Synthetic short-crack physics model.

This is a public stand-in for the proprietary physics model used in the
original study. It combines two textbook ingredients:

* an El Haddad-type short-crack threshold that rises with crack extension
  ``delta`` towards the long-crack threshold,

      dK_th(delta) = dK_th,LC * sqrt((delta + a_i) / (delta + a_i + a_EH)),

* a Paris-type growth law driven by the excess of the applied stress-intensity
  range over that threshold, integrated separately for the depth front ``a``
  and the surface front ``c`` of a semi-elliptical (or embedded) defect.

A specimen "transitions" when either front's threshold reaches a fixed
fraction of the long-crack threshold. The cycle count at that moment is the
physics-model life prediction, and the threshold-vs-cycle curves are the
reference shapes the hybrid decoder is regularised towards.

All constants are generic illustrative values, not material data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticMaterial:
    """Illustrative constants for the synthetic short-crack model."""

    long_crack_threshold: float = 5.0  # dK_th,LC  [MPa sqrt(m)]
    el_haddad_length_m: float = 100e-6  # a_EH
    intrinsic_length_m: float = 5e-6  # a_i, keeps dK_th(0) > 0
    paris_c: float = 1.0e-11  # [m/cycle / (MPa sqrt(m))^m]
    paris_m: float = 3.0
    stress_ratio_exponent: float = 0.3  # Walker-type mean-stress correction
    transition_fraction: float = 0.95  # transition at 95 % of dK_th,LC


DEFAULT_MATERIAL = SyntheticMaterial()


def transition_threshold(material: SyntheticMaterial = DEFAULT_MATERIAL) -> float:
    """Threshold value that defines the short-to-long crack transition."""

    return float(material.transition_fraction * material.long_crack_threshold)


def short_crack_threshold(extension_m: np.ndarray, material: SyntheticMaterial = DEFAULT_MATERIAL) -> np.ndarray:
    """El Haddad-type threshold as a function of crack extension."""

    extension = np.maximum(np.asarray(extension_m, dtype=float), 0.0)
    a_i = material.intrinsic_length_m
    a_eh = material.el_haddad_length_m
    return material.long_crack_threshold * np.sqrt((extension + a_i) / (extension + a_i + a_eh))


def invert_short_crack_threshold(
    threshold: np.ndarray,
    material: SyntheticMaterial = DEFAULT_MATERIAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the extension that produces ``threshold`` and a validity mask."""

    ratio = np.square(np.asarray(threshold, dtype=float) / material.long_crack_threshold)
    valid = np.isfinite(ratio) & (ratio < 1.0)
    a_i = material.intrinsic_length_m
    a_eh = material.el_haddad_length_m
    with np.errstate(divide="ignore", invalid="ignore"):
        extension = (ratio * (a_i + a_eh) - a_i) / (1.0 - ratio)
    return np.where(valid, np.maximum(extension, 0.0), np.nan), valid


def transition_extension(material: SyntheticMaterial = DEFAULT_MATERIAL) -> float:
    """Crack extension at which the threshold reaches the transition value."""

    extension, _ = invert_short_crack_threshold(np.array([transition_threshold(material)]), material)
    return float(extension[0])


def _stress_intensity_ranges(
    a_m: float,
    c_m: float,
    delta_sigma: float,
    width_m: float,
    thickness_m: float,
    embedded: bool,
) -> tuple[float, float]:
    """Simplified Newman-Raju-style ranges at the depth and surface points."""

    aspect = min(a_m / c_m, c_m / a_m)
    shape_q = 1.0 + 1.464 * aspect**1.65
    base = delta_sigma * np.sqrt(np.pi * a_m / shape_q)
    finite_t = 1.0 / np.sqrt(np.cos(min(np.pi * a_m / (2.0 * thickness_m), 1.3)))
    finite_w = 1.0 / np.sqrt(np.cos(min(np.pi * c_m / width_m, 1.3)))
    depth_factor = 1.0 if embedded else 1.12
    surface_factor = 0.9 if embedded else 1.0
    dk_a = depth_factor * base * finite_t
    dk_c = surface_factor * np.sqrt(a_m / c_m) * base * finite_w
    return float(dk_a), float(dk_c)


def simulate_specimen(
    a0_mm: float,
    c0_mm: float,
    sigma_max: float,
    sigma_min: float,
    w_mm: float,
    t_mm: float,
    defect_type: str,
    *,
    material: SyntheticMaterial = DEFAULT_MATERIAL,
    post_threshold_cycles: float = 10_000.0,
    post_event_s: float = 0.2,
    step_extension_m: float = 2e-6,
    max_steps: int = 50_000,
    max_cycles: float = 1e8,
) -> dict[str, np.ndarray | float] | None:
    """Integrate both crack fronts until shortly after the transition.

    Returns ``None`` for non-propagating defects (no transition within
    ``max_cycles``).
    """

    a = a0_mm * 1e-3
    c = c0_mm * 1e-3
    width = w_mm * 1e-3
    thickness = t_mm * 1e-3
    delta_sigma = sigma_max - sigma_min
    stress_ratio = sigma_min / sigma_max
    mean_stress = (1.0 - stress_ratio) ** (-material.stress_ratio_exponent)
    embedded = defect_type == "embedded"
    k_transition = transition_threshold(material)

    cycle, ext_a, ext_c = 0.0, 0.0, 0.0
    rows = [(cycle, ext_a, ext_c)]
    crossing_cycle = np.nan
    stop_cycle = np.inf
    for _ in range(max_steps):
        dk_a, dk_c = _stress_intensity_ranges(a + ext_a, c + ext_c, delta_sigma, width, thickness, embedded)
        th_a, th_c = short_crack_threshold(np.array([ext_a, ext_c]), material)
        rate_a = material.paris_c * max(dk_a * mean_stress - th_a, 0.0) ** material.paris_m
        rate_c = material.paris_c * max(dk_c * mean_stress - th_c, 0.0) ** material.paris_m
        fastest = max(rate_a, rate_c)
        if fastest <= 0.0:
            return None
        d_cycle = step_extension_m / fastest
        cycle += d_cycle
        ext_a += rate_a * d_cycle
        ext_c += rate_c * d_cycle
        rows.append((cycle, ext_a, ext_c))
        if np.isnan(crossing_cycle) and max(short_crack_threshold(np.array([ext_a, ext_c]), material)) >= k_transition:
            crossing_cycle = cycle
            stop_cycle = cycle + 1.05 * post_event_s * post_threshold_cycles
        if cycle >= stop_cycle:
            break
        if cycle > max_cycles:
            return None
    if not np.isfinite(crossing_cycle) or cycle < stop_cycle:
        return None
    data = np.asarray(rows, dtype=float)
    return {
        "cycle": data[:, 0],
        "extension_a_m": data[:, 1],
        "extension_c_m": data[:, 2],
        "crossing_cycle": float(crossing_cycle),
    }
