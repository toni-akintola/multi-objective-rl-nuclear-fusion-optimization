"""
Simple vertical displacement (VDE) guard helpers.

The ITER Hybrid environment does not expose a direct "z" coordinate, so we
derive a proxy from the asymmetry between the upper and lower plasma
triangularity (delta). A large difference corresponds to the plasma column
leaning upward or downward. This heuristic is sufficient for visualization
and reward shaping purposes.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

Z_MAX = 15.0  # centimetres – acceptable vertical excursion
MAX_DZ = 2.5  # centimetres per step – acceptable vertical velocity


def _extract_vertical_proxy(state: Dict[str, Any]) -> float:
    """
    Compute a proxy for the plasma vertical position (cm).

    We map the asymmetry between upper and lower triangularity to a vertical
    offset. Values are scaled to centimetres to match the historical visual
    encoding in the demo scripts.
    """
    profiles = state.get("profiles") or {}

    # Triangularity at the upper and lower separatrix. Defaults keep the plasma
    # centred if the values are missing.
    delta_upper = float(np.asarray(profiles.get("delta_upper", [0.0])).flatten()[0])
    delta_lower = float(np.asarray(profiles.get("delta_lower", [0.0])).flatten()[0])

    # Difference acts as a signed displacement. The scale factor is chosen so
    # that modest asymmetries produce several centimetres of motion, which
    # works well for visual feedback.
    z_cm = (delta_upper - delta_lower) * 20.0
    z_cm = float(np.clip(z_cm, -50.0, 50.0))  # avoid runaway numbers

    return z_cm


def vertical_violation(prev_state: Dict[str, Any] | None, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate whether the plasma vertical position is within a safe band.

    Returns a dictionary compatible with the previous optimisation utilities:
        - z: current vertical proxy (cm)
        - dz: change since previous step (cm)
        - in_band: bool, |z| <= Z_MAX
        - smooth: bool, |dz| <= MAX_DZ
        - ok: in_band and smooth
        - severity: combined magnitude of position and velocity violations
    """
    if state is None:
        return {
            "z": 0.0,
            "dz": 0.0,
            "in_band": True,
            "smooth": True,
            "ok": True,
            "severity": 0.0,
        }

    z_cm = _extract_vertical_proxy(state)

    if prev_state is None:
        prev_z = z_cm
    else:
        prev_z = _extract_vertical_proxy(prev_state)

    dz = z_cm - prev_z

    in_band = abs(z_cm) <= Z_MAX
    smooth = abs(dz) <= MAX_DZ

    # Severity accumulates position and velocity violations (normalised >= 0).
    pos_severity = max(abs(z_cm) - Z_MAX, 0.0) / max(Z_MAX, 1e-6)
    vel_severity = max(abs(dz) - MAX_DZ, 0.0) / max(MAX_DZ, 1e-6)
    severity = pos_severity + vel_severity

    return {
        "z": z_cm,
        "dz": dz,
        "in_band": in_band,
        "smooth": smooth,
        "ok": in_band and smooth,
        "severity": severity,
    }


__all__ = ["vertical_violation", "Z_MAX", "MAX_DZ"]

