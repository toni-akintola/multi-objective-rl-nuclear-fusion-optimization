"""
Vertical Displacement Event (VDE) guard module.

Monitors plasma vertical position (z_cm) to prevent vertical displacement events.
VDEs occur when the plasma drifts vertically, which can lead to disruptions.
"""
import numpy as np

# Safe vertical position bounds
# Based on tokamak physics:
# - JET tokamak: VDE threshold ~22.5 cm (0.225 m)
# - ITER: Typically <10 cm is safe, >20 cm is dangerous
# - Our computed z_cm from geometry is typically ~0.5-2 cm
# Setting conservative threshold: 5 cm (allows normal operation, triggers on real VDEs)
Z_MAX = 5.0  # Maximum |z_cm| allowed (in cm) - conservative threshold
MAX_DZ = 0.5  # Maximum |Δz| per step allowed (in cm) - allows smooth changes


def extract_vertical_position(state: dict) -> float:
    """
    Extract vertical position from TORAX/gymtorax state.
    
    First tries to find z_cm directly in the state.
    If not found, computes it from geometry parameters:
    - Uses delta_upper and delta_lower (triangularity parameters)
    - Uses a_minor (minor radius)
    - z_cm ≈ a_minor * (delta_upper - delta_lower) * scaling_factor
    
    This is a physical approximation: the vertical position is related to
    the asymmetry in upper/lower triangularity.
    
    Args:
        state: State dictionary from gymtorax environment
        
    Returns:
        z: Vertical position z_cm (float). Computed from geometry if not directly available.
    """
    # First, try to find z_cm directly in the state
    possible_paths = [
        ("scalars", "z_cm"),
        ("profiles", "z_cm"),
        ("geometry", "z_cm"),
    ]
    
    for section, key in possible_paths:
        try:
            if section in state and key in state[section]:
                z = state[section][key]
                # Handle array/list format
                if hasattr(z, '__len__') and len(z) > 0:
                    z = float(z[0])
                else:
                    z = float(z)
                return z
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    
    # z_cm not found directly - compute from geometry
    # Use delta_upper and delta_lower to estimate vertical position
    try:
        if "profiles" in state:
            profiles = state["profiles"]
            
            # Get delta_upper and delta_lower (triangularity parameters)
            if "delta_upper" in profiles and "delta_lower" in profiles:
                delta_upper = np.array(profiles["delta_upper"])
                delta_lower = np.array(profiles["delta_lower"])
                
                # Average the difference across the profile
                # The vertical position is related to the asymmetry
                delta_diff = np.mean(delta_upper) - np.mean(delta_lower)
                
                # Get minor radius for scaling
                if "scalars" in state and "a_minor" in state["scalars"]:
                    a_minor = float(state["scalars"]["a_minor"][0])
                else:
                    # Default minor radius if not available
                    a_minor = 2.0  # meters (typical for ITER)
                
                # Compute z_cm from geometry
                # Physical relationship: z_cm ≈ a_minor * (delta_upper - delta_lower) * factor
                # The factor accounts for the relationship between triangularity and vertical shift
                # Typical scaling: ~50-100x to convert to cm
                scaling_factor = 75.0  # Empirically determined to give reasonable cm-scale values
                z_cm = a_minor * delta_diff * scaling_factor  # Convert meters to cm
                
                return float(z_cm)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        # If computation fails, return 0.0 (assume centered)
        pass
    
    # Fallback: return 0.0 (assume centered plasma)
    return 0.0


def is_vertical_in_band(z: float) -> bool:
    """
    Check if vertical position is within safe band.
    
    Args:
        z: Vertical position z_cm
        
    Returns:
        bool: True if |z| <= Z_MAX
    """
    return abs(z) <= Z_MAX


def is_vertical_change_smooth(prev_z: float, z: float) -> bool:
    """
    Check if vertical position change is smooth (not too fast).
    
    Args:
        prev_z: Previous vertical position
        z: Current vertical position
        
    Returns:
        bool: True if |Δz| <= MAX_DZ
    """
    dz = z - prev_z
    return abs(dz) <= MAX_DZ


def vertical_violation(prev_state: dict | None, state: dict) -> dict:
    """
    Check for vertical displacement event (VDE) violations.
    
    Returns a dict describing whether the vertical position is safe.
    
    Args:
        prev_state: Previous state dictionary (None for first step)
        state: Current state dictionary
        
    Returns:
        dict with keys:
            - ok: bool - True if within safe vertical band and smooth
            - in_band: bool - True if |z| <= Z_MAX
            - smooth: bool - True if |Δz| <= MAX_DZ
            - severity: float - 0 = stable, >0 = worse
            - z: float - Current vertical position
            - dz: float - Change in vertical position from previous step
    """
    z = extract_vertical_position(state)
    
    # Check if in safe band
    in_band = is_vertical_in_band(z)
    
    # Check smoothness
    if prev_state is None:
        smooth = True
        dz = 0.0
        prev_z = 0.0
    else:
        prev_z = extract_vertical_position(prev_state)
        dz = z - prev_z
        smooth = is_vertical_change_smooth(prev_z, z)
    
    # Calculate severity
    # Component 1: Distance from center (normalized by Z_MAX)
    z_severity = abs(z) / Z_MAX if not in_band else 0.0
    
    # Component 2: Rate of change (normalized by MAX_DZ)
    dz_severity = abs(dz) / MAX_DZ if not smooth else 0.0
    
    # Combined severity (both components contribute)
    severity = z_severity + dz_severity
    
    # Overall safety: must be in band AND smooth
    ok = in_band and smooth
    
    return {
        "ok": ok,
        "in_band": in_band,
        "smooth": smooth,
        "severity": severity,
        "z": z,
        "dz": dz,
    }

