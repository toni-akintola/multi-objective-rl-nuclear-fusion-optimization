import numpy as np
from gymtorax import rewards as torax_reward  # your rewards.py snippet

# "safe operating box" ---
BETA_N_MAX   = 3.0
BETA_N_MIN   = 0.5   # too low = useless for power

QMIN_MIN     = 1.0   # want q_min >= 1
Q95_MIN      = 3.0
Q95_MAX      = 5.0

# --- how much the "shape" is allowed to change per control step ---
MAX_DELTA_BETA_N = 0.2
MAX_DELTA_QMIN   = 0.15
MAX_DELTA_Q95    = 0.4


def extract_shape_vector(state: dict) -> np.ndarray:
    """Cheap 'shape proxy' vector from TORAX / gymtorax state."""
    beta_N = torax_reward.get_beta_N(state)
    q_min  = torax_reward.get_q_min(state)
    q95    = torax_reward.get_q95(state)
    return np.array([beta_N, q_min, q95], dtype=np.float32)


def is_shape_in_safe_box(shape_vec: np.ndarray) -> bool:
    """Check if (beta_N, q_min, q95) lie in predefined safe region."""
    beta_N, q_min, q95 = shape_vec

    if not (BETA_N_MIN <= beta_N <= BETA_N_MAX):
        return False

    if q_min < QMIN_MIN:
        return False

    if not (Q95_MIN <= q95 <= Q95_MAX):
        return False

    return True


def is_shape_change_reasonable(prev_shape: np.ndarray,
                               shape: np.ndarray) -> bool:
    """Limit how violently the 'shape' can move between time steps."""
    d_beta, d_qmin, d_q95 = shape - prev_shape

    if abs(d_beta) > MAX_DELTA_BETA_N:
        return False
    if abs(d_qmin) > MAX_DELTA_QMIN:
        return False
    if abs(d_q95) > MAX_DELTA_Q95:
        return False

    return True


def shape_violation(prev_state: dict | None,
                    state: dict) -> dict:
    """
    Returns a dict describing whether the new state is 'shape safe'.

    Keys:
      - ok: bool
      - in_box: bool  (within safe region)
      - smooth: bool  (no huge jump from prev_state)
      - severity: float  (0 = perfect, >0 worse)
    """
    shape = extract_shape_vector(state)

    in_box = is_shape_in_safe_box(shape)

    if prev_state is None:
        smooth = True
        delta_severity = 0.0
    else:
        prev_shape = extract_shape_vector(prev_state)
        smooth = is_shape_change_reasonable(prev_shape, shape)
        # normalized L2 as a severity measure
        delta = shape - prev_shape
        delta_severity = float(np.linalg.norm(delta))

    # severity based on both absolute safety and jump size
    box_severity = 0.0 if in_box else 1.0
    severity = box_severity + delta_severity

    ok = in_box and smooth

    return {
        "ok": ok,
        "in_box": in_box,
        "smooth": smooth,
        "severity": severity,
        "shape": shape,
    }
