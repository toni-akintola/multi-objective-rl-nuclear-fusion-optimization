# agent.py
import abc
from gymnasium import spaces

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "shape_guard",
    Path(__file__).parent / "optimization-for-constraints" / "shape_guard.py"
)
shape_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape_guard)
shape_violation = shape_guard.shape_violation


class Agent(abc.ABC):
    """
    Abstract base class for all agents.

    Adds optional shape-safety tracking via prev_state + last_shape_info.
    """

    def __init__(self, action_space: spaces.Space, shape_penalty: float = 0.0):
        """
        Args:
            action_space: env action space.
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

        Returns:
            new_reward: original reward + (possibly negative) shape penalty.
        """
        if state is None or self.shape_penalty <= 0.0:
            # nothing to do
            self.prev_state = state
            return reward

        info = shape_violation(self.prev_state, state)
        self.last_shape_info = info
        self.prev_state = state

        if not info["ok"]:
            penalty = self.shape_penalty * (1.0 + info["severity"])
            return reward - penalty

        return reward

    @abc.abstractmethod
    def act(self, observation) -> dict:
        """Return action dict for the environment."""
        raise NotImplementedError


class RandomAgent(Agent):
    """Agent that produces random actions within the action space."""

    def __init__(self, action_space: spaces.Space,
                 shape_penalty: float = 0.0,
                 damp_on_violation: bool = False,
                 damp_factor: float = 0.5):
        super().__init__(action_space, shape_penalty=shape_penalty)
        self.damp_on_violation = damp_on_violation
        self.damp_factor = damp_factor

    def act(self, observation) -> dict:
        """
        Compute a (possibly shape-aware) random action.

        If damp_on_violation=True and last_shape_info indicates
        a problem, we shrink the magnitude of the sampled action.
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
