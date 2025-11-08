"""
IterHybrid environment with integrated shape guard and vertical guard penalties.

This environment wraps gymtorax/IterHybrid-v0 and automatically applies
shape guard and vertical guard (VDE) penalties/bonuses to the reward based on plasma safety constraints.
"""
import gymnasium as gym
from gymnasium import Wrapper
import importlib.util
from pathlib import Path


class IterHybridShapeGuardEnv(Wrapper):
    """
    IterHybrid environment with integrated shape guard and vertical guard (VDE).
    
    The shape guard monitors β_N, q_min, and q95 to ensure plasma safety.
    The vertical guard monitors z_cm to prevent Vertical Displacement Events (VDEs).
    
    Both guards automatically modify the reward to:
    - Penalize violations that worsen
    - Reward safe states
    - Reward corrective actions (self-fixing)
    
    Args:
        env: The gymtorax IterHybrid environment to wrap (or env_id string)
        shape_penalty: Coefficient for shape violation penalty (default: 0.05)
        vertical_penalty: Coefficient for vertical guard penalty (default: 0.05)
        enable_shape_guard: Whether to enable shape guard (default: True)
        enable_vertical_guard: Whether to enable vertical guard (default: True)
        **env_kwargs: Additional arguments passed to gym.make() if env is a string
    """
    
    def __init__(self, env=None, shape_penalty=0.05, vertical_penalty=0.05, 
                 enable_shape_guard=True, enable_vertical_guard=True, **env_kwargs):
        # If env is a string, create the environment
        if isinstance(env, str):
            env = gym.make(env, **env_kwargs)
        elif env is None:
            env = gym.make("gymtorax/IterHybrid-v0", **env_kwargs)
        
        super().__init__(env)
        
        # Initialize shape guard
        self.shape_penalty = shape_penalty if enable_shape_guard else 0.0
        self.enable_shape_guard = enable_shape_guard
        
        # Initialize vertical guard
        self.vertical_penalty = vertical_penalty if enable_vertical_guard else 0.0
        self.enable_vertical_guard = enable_vertical_guard
        
        # Import shape guard module
        spec = importlib.util.spec_from_file_location(
            "shape_guard",
            Path(__file__).parent / "optimization-for-constraints" / "shape_guard.py"
        )
        shape_guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shape_guard)
        self.shape_violation = shape_guard.shape_violation
        
        # Import vertical guard module
        spec_vertical = importlib.util.spec_from_file_location(
            "vertical_guard",
            Path(__file__).parent / "optimization-for-constraints" / "vertical_guard.py"
        )
        vertical_guard = importlib.util.module_from_spec(spec_vertical)
        spec_vertical.loader.exec_module(vertical_guard)
        self.vertical_violation = vertical_guard.vertical_violation
        
        # Track previous state for both guards
        self.prev_state = None
        self.last_shape_info = None
        self.last_shape_severity = float('inf')
        self.last_vertical_info = None
        self.last_vertical_severity = float('inf')
        
    def reset(self, **kwargs):
        """Reset environment and initialize shape and vertical tracking."""
        observation, info = self.env.reset(**kwargs)
        self.prev_state = observation
        self.last_shape_info = None
        self.last_shape_severity = float('inf')
        self.last_vertical_info = None
        self.last_vertical_severity = float('inf')
        return observation, info
    
    def step(self, action):
        """Step environment and apply shape guard and vertical guard to reward."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply both guards to reward (combined in single method)
        reward = self._apply_guards(reward, observation)
        
        # Update previous state
        self.prev_state = observation
        
        return observation, reward, terminated, truncated, info
    
    def _apply_guards(self, reward: float, state: dict) -> float:
        """
        Apply both shape guard and vertical guard to reward computation.
        
        Combined reward shaping:
        - Penalizes violations that worsen or occur when safe
        - Rewards safe states
        - Rewards corrective actions (self-fixing)
        - Uses normalized severity for balanced penalties
        
        Returns:
            new_reward: original reward + combined shape and vertical bonuses/penalties.
        """
        if state is None:
            return reward
        
        # Apply shape guard if enabled
        if self.enable_shape_guard and self.shape_penalty > 0.0:
            reward = self._apply_shape_guard(reward, state)
        
        # Apply vertical guard if enabled
        if self.enable_vertical_guard and self.vertical_penalty > 0.0:
            reward = self._apply_vertical_guard(reward, state)
        
        return reward
    
    def _apply_shape_guard(self, reward: float, state: dict) -> float:
        """
        Apply shape guard to reward computation.
        
        Returns:
            new_reward: original reward + shape bonus/penalty.
        """
        # Get shape violation info
        info = self.shape_violation(self.prev_state, state)
        
        # Get previous severity BEFORE updating last_shape_info
        prev_severity = self.last_shape_severity
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
        self.last_shape_severity = info.get("severity", 0.0)
        
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
    
    def _apply_vertical_guard(self, reward: float, state: dict) -> float:
        """
        Apply vertical guard (VDE) to reward computation.
        
        Returns:
            new_reward: original reward + vertical bonus/penalty.
        """
        # Get vertical violation info
        info = self.vertical_violation(self.prev_state, state)
        
        # Get previous severity BEFORE updating last_vertical_info
        prev_severity = self.last_vertical_severity
        was_in_violation = (self.last_vertical_info is not None and 
                           not self.last_vertical_info.get("ok", True))
        
        # Check if this is a corrective action
        is_corrective = False
        if was_in_violation and not info["ok"]:
            # We were in violation, and still are - but is severity decreasing?
            current_severity = info.get("severity", 0.0)
            is_corrective = current_severity < prev_severity
        
        # Update state tracking
        self.last_vertical_info = info
        self.last_vertical_severity = info.get("severity", 0.0)
        
        # Normalize severity to [0, 1] range for more balanced penalties
        # Vertical severity can be higher (up to 10+), so normalize more aggressively
        normalized_severity = min(info["severity"] / 10.0, 1.0)  # Cap at 1.0 for severity > 10.0
        
        # Reward safe states
        if info["ok"]:
            # Bonus for being safe: significant positive reward
            safety_bonus = self.vertical_penalty * 3.0  # 3x penalty as bonus to strongly encourage safety
            return reward + safety_bonus
        
        # Handle violations
        if not info["ok"]:
            if is_corrective:
                # Reward self-fixing: give bonus proportional to improvement
                severity_reduction = prev_severity - info["severity"]
                # Bonus for reducing severity (self-fixing)
                # Normalize the reduction (severity can be 0-10+)
                normalized_reduction = min(severity_reduction / 5.0, 1.0)
                # Much larger bonus for self-fixing to strongly encourage recovery
                corrective_bonus = self.vertical_penalty * 2.0 * normalized_reduction  # Up to 2x penalty as bonus
                return reward + corrective_bonus
            else:
                # Penalize worsening violations or new violations
                # Use normalized severity for more balanced penalty
                # Very small penalty: 0.1x to 0.4x of vertical_penalty (much smaller)
                penalty = self.vertical_penalty * (0.1 + normalized_severity * 0.3)
                return reward - penalty
        
        return reward
    
    @property
    def shape_info(self):
        """Get the last shape guard information."""
        return self.last_shape_info
    
    @property
    def vertical_info(self):
        """Get the last vertical guard information."""
        return self.last_vertical_info


def make_iter_hybrid_shape_guard_env(shape_penalty=0.05, vertical_penalty=0.05,
                                     enable_shape_guard=True, enable_vertical_guard=True, **env_kwargs):
    """
    Factory function to create IterHybrid environment with shape guard and vertical guard.
    
    Args:
        shape_penalty: Coefficient for shape violation penalty (default: 0.05)
        vertical_penalty: Coefficient for vertical guard penalty (default: 0.05)
        enable_shape_guard: Whether to enable shape guard (default: True)
        enable_vertical_guard: Whether to enable vertical guard (default: True)
        **env_kwargs: Additional arguments passed to gym.make("gymtorax/IterHybrid-v0")
    
    Returns:
        IterHybridShapeGuardEnv: Wrapped environment with shape and vertical guards
    
    Example:
        >>> env = make_iter_hybrid_shape_guard_env(shape_penalty=0.05, vertical_penalty=0.05)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> if env.shape_info:
        ...     print(f"Shape status: {'SAFE' if env.shape_info['ok'] else 'VIOLATION'}")
        >>> if env.vertical_info:
        ...     print(f"Vertical status: {'SAFE' if env.vertical_info['ok'] else 'VIOLATION'}")
    """
    return IterHybridShapeGuardEnv(
        env="gymtorax/IterHybrid-v0",
        shape_penalty=shape_penalty,
        vertical_penalty=vertical_penalty,
        enable_shape_guard=enable_shape_guard,
        enable_vertical_guard=enable_vertical_guard,
        **env_kwargs
    )


if __name__ == "__main__":
    # Example usage
    print("Creating IterHybrid environment with shape guard and vertical guard...")
    env = make_iter_hybrid_shape_guard_env(
        shape_penalty=0.05, 
        vertical_penalty=0.05,
        enable_shape_guard=True,
        enable_vertical_guard=True
    )
    
    print("Resetting environment...")
    observation, info = env.reset()
    
    print("Running a few steps...")
    for step in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        status_parts = []
        if env.shape_info:
            shape_info = env.shape_info
            shape_status = "🟢 Shape SAFE" if shape_info["ok"] else "🔴 Shape VIOLATION"
            status_parts.append(f"{shape_status} (β_N={shape_info['shape'][0]:.3f}, severity={shape_info['severity']:.3f})")
        
        if env.vertical_info:
            vertical_info = env.vertical_info
            vert_status = "🟢 Vertical SAFE" if vertical_info["ok"] else "🔴 Vertical VIOLATION"
            status_parts.append(f"{vert_status} (z={vertical_info['z']:.3f} cm, severity={vertical_info['severity']:.3f})")
        
        print(f"Step {step+1}: Reward={reward:.3f}")
        for status in status_parts:
            print(f"  {status}")
        
        if terminated or truncated:
            break
    
    print("Done!")
    env.close()

