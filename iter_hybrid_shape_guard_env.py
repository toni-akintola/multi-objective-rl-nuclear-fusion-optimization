"""
IterHybrid environment with integrated shape guard penalty.

This environment wraps gymtorax/IterHybrid-v0 and automatically applies
shape guard penalties/bonuses to the reward based on plasma safety constraints.
"""
import gymnasium as gym
from gymnasium import Wrapper
import importlib.util
from pathlib import Path


class IterHybridShapeGuardEnv(Wrapper):
    """
    IterHybrid environment with integrated shape guard.
    
    The shape guard monitors β_N, q_min, and q95 to ensure plasma safety.
    It automatically modifies the reward to:
    - Penalize violations that worsen
    - Reward safe states
    - Reward corrective actions (self-fixing)
    
    Args:
        env: The gymtorax IterHybrid environment to wrap (or env_id string)
        shape_penalty: Coefficient for shape violation penalty (default: 0.05)
        enable_shape_guard: Whether to enable shape guard (default: True)
        **env_kwargs: Additional arguments passed to gym.make() if env is a string
    """
    
    def __init__(self, env=None, shape_penalty=0.05, enable_shape_guard=True, **env_kwargs):
        # If env is a string, create the environment
        if isinstance(env, str):
            env = gym.make(env, **env_kwargs)
        elif env is None:
            env = gym.make("gymtorax/IterHybrid-v0", **env_kwargs)
        
        super().__init__(env)
        
        # Initialize shape guard
        self.shape_penalty = shape_penalty if enable_shape_guard else 0.0
        self.enable_shape_guard = enable_shape_guard
        
        # Import shape guard module
        spec = importlib.util.spec_from_file_location(
            "shape_guard",
            Path(__file__).parent / "optimization-for-constraints" / "shape_guard.py"
        )
        shape_guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shape_guard)
        self.shape_violation = shape_guard.shape_violation
        
        # Track previous state for shape guard
        self.prev_state = None
        self.last_shape_info = None
        self.last_shape_severity = float('inf')
        
    def reset(self, **kwargs):
        """Reset environment and initialize shape tracking."""
        observation, info = self.env.reset(**kwargs)
        self.prev_state = observation
        self.last_shape_info = None
        self.last_shape_severity = float('inf')
        return observation, info
    
    def step(self, action):
        """Step environment and apply shape guard to reward."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply shape guard to reward if enabled
        if self.enable_shape_guard and self.shape_penalty > 0.0:
            reward = self._apply_shape_guard(reward, observation)
        
        # Update previous state
        self.prev_state = observation
        
        return observation, reward, terminated, truncated, info
    
    def _apply_shape_guard(self, reward: float, state: dict) -> float:
        """
        Apply shape guard to reward computation.
        
        Improved reward shaping:
        - Penalizes violations that worsen or occur when safe
        - Rewards safe states
        - Rewards corrective actions (self-fixing)
        - Uses normalized severity for balanced penalties
        
        Returns:
            new_reward: original reward + shape bonus/penalty.
        """
        if state is None:
            return reward
        
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
    
    @property
    def shape_info(self):
        """Get the last shape guard information."""
        return self.last_shape_info


def make_iter_hybrid_shape_guard_env(shape_penalty=0.05, enable_shape_guard=True, **env_kwargs):
    """
    Factory function to create IterHybrid environment with shape guard.
    
    Args:
        shape_penalty: Coefficient for shape violation penalty (default: 0.05)
        enable_shape_guard: Whether to enable shape guard (default: True)
        **env_kwargs: Additional arguments passed to gym.make("gymtorax/IterHybrid-v0")
    
    Returns:
        IterHybridShapeGuardEnv: Wrapped environment with shape guard
    
    Example:
        >>> env = make_iter_hybrid_shape_guard_env(shape_penalty=0.05)
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        >>> if env.shape_info:
        ...     print(f"Shape status: {'SAFE' if env.shape_info['ok'] else 'VIOLATION'}")
    """
    return IterHybridShapeGuardEnv(
        env="gymtorax/IterHybrid-v0",
        shape_penalty=shape_penalty,
        enable_shape_guard=enable_shape_guard,
        **env_kwargs
    )


if __name__ == "__main__":
    # Example usage
    print("Creating IterHybrid environment with shape guard...")
    env = make_iter_hybrid_shape_guard_env(shape_penalty=0.05, enable_shape_guard=True)
    
    print("Resetting environment...")
    observation, info = env.reset()
    
    print("Running a few steps...")
    for step in range(5):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if env.shape_info:
            shape_info = env.shape_info
            status = "🟢 SAFE" if shape_info["ok"] else "🔴 VIOLATION"
            print(f"Step {step+1}: Reward={reward:.3f}, Status={status}, "
                  f"β_N={shape_info['shape'][0]:.3f}, q_min={shape_info['shape'][1]:.3f}, "
                  f"q95={shape_info['shape'][2]:.3f}, Severity={shape_info['severity']:.3f}")
        
        if terminated or truncated:
            break
    
    print("Done!")
    env.close()

