import abc
from gymnasium import spaces  # Assuming a gym-like space object
import importlib.util
from pathlib import Path
import numpy as np

# Import shape guard module
# Try multiple possible paths
shape_guard_paths = [
    Path(__file__).parent.parent / "utils" / "shape_guard.py",
    Path(__file__).parent / "optimization-for-constraints" / "shape_guard.py",
]

shape_guard = None
shape_violation = None

for shape_guard_path in shape_guard_paths:
    if shape_guard_path.exists():
        spec = importlib.util.spec_from_file_location("shape_guard", shape_guard_path)
        shape_guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shape_guard)
        shape_violation = shape_guard.shape_violation
        break

if shape_violation is None:
    # Create a dummy shape_violation function if not found
    def shape_violation(prev_state, state):
        return {"ok": True, "shape": [0, 0, 0], "in_box": True, "smooth": True}


class Agent(abc.ABC):
    """
    Abstract base class for all agents.

    Adds optional shape-safety tracking via prev_state + last_shape_info.
    """

    def __init__(self, action_space: spaces.Space, shape_penalty: float = 0.0):
        """
        Initialize the agent with the given action space.

        Args:
            action_space: The environment's action space (e.g., from gymnasium).
            shape_penalty: coefficient for shape violation penalty (0 = off).
        """
        self.action_space = action_space
        self.shape_penalty = shape_penalty

        # For shape safety
        self.prev_state: dict | None = None
        self.last_shape_info: dict | None = None

    def reset_state(self, state: dict | None):
        """Call this at env.reset() to initialize tracking."""
        self.prev_state = state
        self.last_shape_info = None

    def apply_shape_safety(self, reward: float, state: dict | None) -> float:
        """
        Call this AFTER env.step().

        Allows corrective violations: if we're already in violation, we allow
        violations that reduce severity (corrective actions). Only penalizes
        violations that make things worse or occur when we were safe.

        Returns:
            new_reward: original reward + (possibly negative) shape penalty.
        """
        if state is None or self.shape_penalty <= 0.0:
            # nothing to do
            self.prev_state = state
            return reward

        info = shape_violation(self.prev_state, state)

        # Check if this is a corrective action
        was_in_violation = (
            self.last_shape_info is not None
            and not self.last_shape_info.get("ok", True)
        )
        is_corrective = False

        if was_in_violation and not info["ok"]:
            # We were in violation, and still are - but is severity decreasing?
            prev_severity = self.last_shape_info.get("severity", float("inf"))
            current_severity = info.get("severity", 0.0)
            is_corrective = current_severity < prev_severity

        self.last_shape_info = info
        self.prev_state = state

        # Only penalize if:
        # 1. We're violating AND it's not a corrective action (severity decreasing)
        # 2. We went from safe to violation (new violation)
        if not info["ok"] and not is_corrective:
            penalty = self.shape_penalty * (1.0 + info["severity"])
            return reward - penalty

        # If corrective, we might even give a small bonus (optional)
        # For now, just don't penalize corrective actions
        return reward

    @abc.abstractmethod
    def act(self, observation) -> dict:
        """
        Compute the next action based on the current observation.

        Subclasses MUST implement this method.

        Args:
            observation: The current observation from the environment.

        Returns:
            dict: Action dictionary for the environment.
        """
        pass  # The implementation is left to the child class


class RandomAgent(Agent):
    """Agent that produces random actions within the action space.

    This agent inherits the __init__ method from the base Agent class.
    """

    def __init__(
        self,
        action_space: spaces.Space,
        shape_penalty: float = 0.0,
        damp_on_violation: bool = False,
        damp_factor: float = 0.5,
    ):
        super().__init__(action_space, shape_penalty=shape_penalty)
        self.damp_on_violation = damp_on_violation
        self.damp_factor = damp_factor

    def act(self, observation) -> dict:
        """
        Compute a (possibly shape-aware) random action.

        If damp_on_violation=True and last_shape_info indicates
        a problem, we shrink the magnitude of the sampled action.

        Args:
            observation: The current observation (unused by this agent).

        Returns:
            dict: A random action dictionary.
        """
        _ = observation  # unused

        action = self.action_space.sample()

        if self.damp_on_violation and self.last_shape_info is not None:
            if not self.last_shape_info["ok"]:
                # Example: scale numeric entries if action is a dict
                if isinstance(action, dict):
                    scaled = {}
                    for k, v in action.items():
                        # assume v is np.ndarray or float
                        scaled[k] = self.damp_factor * v
                    action = scaled
                else:
                    # assume action is np.ndarray-like
                    action = self.damp_factor * action

        return action


_NBI_W_TO_MA = 1 / 16e6
_NBI_POWERS = np.array([0, 0, 33e6])
_R_NBI = 0.25
_W_NBI = 0.25
_ECCD_POWER = {0: 0, 99: 0, 100: 20.0e6}


class PIDAgent(Agent):
    """
    Simple PID-based controller that adjusts plasma current to follow a target
    trajectory using classic proportional-integral-derivative control with
    ramp-rate limiting and optional shape safety.
    """

    def __init__(
        self,
        action_space: spaces.Space,
        shape_penalty: float = 0.0,
        get_j_target=None,
        ramp_rate: float = 0.2e6,
        kp: float = 0.6e6,
        ki: float = 0.05e6,
        kd: float = 0.0,
    ):
        super().__init__(action_space, shape_penalty=shape_penalty)
        self.get_j_target = get_j_target or self._default_target

        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Time step in seconds (ITER hybrid sim uses 1 second fixed DT)
        self.dt = 1.0

        # Ramp rate limit in A/s (e.g., 0.2 MA/s = 0.2e6 A/s)
        self.ramp_rate = ramp_rate

        # Physical limits for Ip current (MA converted to Amps)
        self.ip_min = float(action_space.spaces["Ip"].low[0])
        self.ip_max = float(action_space.spaces["Ip"].high[0])

        # Controller state
        self._reset_controller_state()

    def _reset_controller_state(self):
        self.time = 0
        self.error_integral = 0.0
        self.previous_error = 0.0
        self.ip_controlled = 0.0

        # History buffers for analysis / plotting
        self.j_target_history: list[float] = []
        self.j_actual_history: list[float] = []
        self.time_history: list[int] = []
        self.error_history: list[float] = []
        self.action_history: list[float] = []

    def reset_state(self, state):
        super().reset_state(state)
        self._reset_controller_state()

    @staticmethod
    def _default_target(timestep: int) -> float:
        """
        Default j_target schedule (center current density) in Amps.
        Keeps a flat 3 MA target with a late ramp to 4 MA.
        """
        if timestep < 60:
            return 3.0e6
        if timestep < 120:
            return 3.5e6
        return 4.0e6

    def act(self, observation) -> dict:
        """
        Compute the next control command using PID on the center current density.
        """
        j_center = observation["profiles"]["j_total"][0]
        j_target = self.get_j_target(self.time)

        # Record for diagnostics
        self.j_target_history.append(j_target)
        self.j_actual_history.append(j_center)
        self.time_history.append(self.time)

        if self.time < 100:
            error = j_target - j_center
            self.error_history.append(error)

            if self.time > 0:
                error_derivative = (error - self.previous_error) / self.dt
            else:
                error_derivative = 0.0

            # PID components
            p_term = self.kp * error
            i_term = self.ki * self.error_integral
            d_term = self.kd * error_derivative
            pid_output = p_term + i_term + d_term

            ip_baseline = 3.0e6
            ip_desired = ip_baseline + pid_output

            max_change = self.ramp_rate * self.dt
            is_ramp_limited = False

            if self.time > 0:
                ip_change = ip_desired - self.ip_controlled
                if abs(ip_change) > max_change:
                    is_ramp_limited = True
                    ip_ramp_limited = self.ip_controlled + np.sign(ip_change) * max_change
                else:
                    ip_ramp_limited = ip_desired
            else:
                ip_ramp_limited = ip_desired

            ip_final = float(np.clip(ip_ramp_limited, self.ip_min, self.ip_max))
            is_power_limited = ip_final != ip_ramp_limited

            if not (self.anti_windup_enabled and (is_ramp_limited or is_power_limited)):
                self.error_integral += error * self.dt

            self.ip_controlled = ip_final
            self.previous_error = error
        else:
            # After 100 seconds keep the last command (flat-top)
            self.error_history.append(self.previous_error)

        action = {
            "Ip": [self.ip_controlled],
            "NBI": [float(_NBI_POWERS[0]), _R_NBI, _W_NBI],
            "ECRH": [float(_ECCD_POWER[0]), 0.35, 0.05],
        }

        if self.time == 98:
            action["ECRH"][0] = float(_ECCD_POWER[99])
            action["NBI"][0] = float(_NBI_POWERS[1])

        if self.time >= 99:
            action["ECRH"][0] = float(_ECCD_POWER[100])
            action["NBI"][0] = float(_NBI_POWERS[2])

        self.time += 1
        self.action_history.append(self.ip_controlled)

        return action

    @property
    def anti_windup_enabled(self) -> bool:
        return True