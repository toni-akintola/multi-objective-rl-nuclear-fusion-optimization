import numpy as np
from gymtorax import rewards as torax_reward  # your rewards.py snippet
from dataclasses import dataclass

# Default "safe operating box" constraints
@dataclass
class ShapeConstraints:
    """Configurable constraints for plasma shape safety."""
    beta_n_min: float = 0.5   # too low = useless for power
    beta_n_max: float = 3.0
    
    q_min_min: float = 1.0   # want q_min >= 1
    q95_min: float = 3.0
    q95_max: float = 5.0
    
    # How much the "shape" is allowed to change per control step
    max_delta_beta_n: float = 0.2
    max_delta_qmin: float = 0.15
    max_delta_q95: float = 0.4

# Default constraints (can be overridden)
DEFAULT_CONSTRAINTS = ShapeConstraints()

# Backward compatibility - use default constraints
BETA_N_MAX = DEFAULT_CONSTRAINTS.beta_n_max
BETA_N_MIN = DEFAULT_CONSTRAINTS.beta_n_min
QMIN_MIN = DEFAULT_CONSTRAINTS.q_min_min
Q95_MIN = DEFAULT_CONSTRAINTS.q95_min
Q95_MAX = DEFAULT_CONSTRAINTS.q95_max
MAX_DELTA_BETA_N = DEFAULT_CONSTRAINTS.max_delta_beta_n
MAX_DELTA_QMIN = DEFAULT_CONSTRAINTS.max_delta_qmin
MAX_DELTA_Q95 = DEFAULT_CONSTRAINTS.max_delta_q95


def extract_shape_vector(state: dict) -> np.ndarray:
    """Cheap 'shape proxy' vector from TORAX / gymtorax state."""
    beta_N = torax_reward.get_beta_N(state)
    q_min  = torax_reward.get_q_min(state)
    q95    = torax_reward.get_q95(state)
    return np.array([beta_N, q_min, q95], dtype=np.float32)


def is_shape_in_safe_box(shape_vec: np.ndarray, constraints: ShapeConstraints = None) -> bool:
    """Check if (beta_N, q_min, q95) lie in predefined safe region."""
    if constraints is None:
        constraints = DEFAULT_CONSTRAINTS
    
    beta_N, q_min, q95 = shape_vec

    if not (constraints.beta_n_min <= beta_N <= constraints.beta_n_max):
        return False

    if q_min < constraints.q_min_min:
        return False

    if not (constraints.q95_min <= q95 <= constraints.q95_max):
        return False

    return True


def is_shape_change_reasonable(prev_shape: np.ndarray,
                               shape: np.ndarray,
                               constraints: ShapeConstraints = None) -> bool:
    """Limit how violently the 'shape' can move between time steps."""
    if constraints is None:
        constraints = DEFAULT_CONSTRAINTS
    
    d_beta, d_qmin, d_q95 = shape - prev_shape

    if abs(d_beta) > constraints.max_delta_beta_n:
        return False
    if abs(d_qmin) > constraints.max_delta_qmin:
        return False
    if abs(d_q95) > constraints.max_delta_q95:
        return False

    return True


def shape_violation(prev_state: dict | None,
                    state: dict,
                    constraints: ShapeConstraints = None) -> dict:
    """
    Returns a dict describing whether the new state is 'shape safe'.

    Keys:
      - ok: bool
      - in_box: bool  (within safe region)
      - smooth: bool  (no huge jump from prev_state)
      - severity: float  (0 = perfect, >0 worse)
      - shape: np.ndarray  [beta_N, q_min, q95]
    """
    if constraints is None:
        constraints = DEFAULT_CONSTRAINTS
    
    shape = extract_shape_vector(state)

    in_box = is_shape_in_safe_box(shape, constraints)

    if prev_state is None:
        smooth = True
        delta_severity = 0.0
    else:
        prev_shape = extract_shape_vector(prev_state)
        smooth = is_shape_change_reasonable(prev_shape, shape, constraints)
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
        "constraints": constraints,  # Include for visualization
    }
