import abc
from gymnasium import spaces # Assuming a gym-like space object
import importlib.util
from pathlib import Path

# Import shape guard module
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
        
        Improved reward shaping:
        - Penalizes violations that worsen or occur when safe
        - Rewards safe states
        - Rewards corrective actions (self-fixing)
        - Uses normalized severity for balanced penalties
        
        Returns:
            new_reward: original reward + shape bonus/penalty.
        """
        if state is None or self.shape_penalty <= 0.0:
            # nothing to do
            self.prev_state = state
            return reward
        
        info = shape_violation(self.prev_state, state)
        
        # Get previous severity BEFORE updating last_shape_info
        prev_severity = self.last_shape_info.get("severity", float('inf')) if self.last_shape_info else float('inf')
        was_in_violation = (self.last_shape_info is not None and 
                           not self.last_shape_info.get("ok", True))
        
        # Check if this is a corrective action
        is_corrective = False
        if was_in_violation and not info["ok"]:
            # We were in violation, and still are - but is severity decreasing?
            current_severity = info.get("severity", 0.0)
            is_corrective = current_severity < prev_severity
        
        # Update state tracking
        self.last_shape_info = info
        self.prev_state = state
        
        # Normalize severity to [0, 1] range for more balanced penalties
        # Severity can be: 0 (perfect) to ~2+ (very bad)
        normalized_severity = min(info["severity"] / 2.0, 1.0)  # Cap at 1.0 for severity > 2.0
        
        # Reward safe states
        if info["ok"]:
            # Bonus for being safe: significant positive reward
            safety_bonus = self.shape_penalty * 3.0  # 3x penalty as bonus to strongly encourage safety
            return reward + safety_bonus
        
        # Handle violations
        if not info["ok"]:
            if is_corrective:
                # Reward self-fixing: give bonus proportional to improvement
                severity_reduction = prev_severity - info["severity"]
                # Bonus for reducing severity (self-fixing)
                # Normalize the reduction (severity is typically 0-2)
                normalized_reduction = min(severity_reduction / 1.0, 1.0)
                # Much larger bonus for self-fixing to strongly encourage recovery
                corrective_bonus = self.shape_penalty * 2.0 * normalized_reduction  # Up to 2x penalty as bonus
                return reward + corrective_bonus
            else:
                # Penalize worsening violations or new violations
                # Use normalized severity for more balanced penalty
                # Very small penalty: 0.1x to 0.4x of shape_penalty (much smaller)
                penalty = self.shape_penalty * (0.1 + normalized_severity * 0.3)
                return reward - penalty
        
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
        pass # The implementation is left to the child class
    
class RandomAgent(Agent):
    """Agent that produces random actions within the action space.
    
    This agent inherits the __init__ method from the base Agent class.
    """

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