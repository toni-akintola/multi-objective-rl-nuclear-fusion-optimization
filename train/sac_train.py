# sac_torax_flat.py
import gymnasium as gym
import numpy as np
import gymtorax  # IMPORTANT: registers gymtorax envs

from gymnasium.spaces import Dict as SpaceDict, Tuple as SpaceTuple, Box
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn


# --------- Wrappers ---------
class FlattenObsWrapper(gym.ObservationWrapper):
    """
    Flattens ANY observation space (nested Dict/Tuple/etc.) into a single Box,
    so Stable-Baselines3 won't complain about nested spaces.
    """
    def __init__(self, env):
        super().__init__(env)
        self._orig_obs_space = env.observation_space
        self.observation_space = flatten_space(self._orig_obs_space)

    def observation(self, obs):
        flattened = flatten(self._orig_obs_space, obs)
        # Check for NaN/Inf in observations
        if np.any(~np.isfinite(flattened)):
            flattened = np.nan_to_num(flattened, nan=0.0, posinf=1e6, neginf=-1e6)
        return flattened


class FlattenActionWrapper(gym.ActionWrapper):
    """
    If the env has a Dict (or Tuple) action space, expose it as a flat Box for SB3.
    We unflatten before sending to the underlying env.
    """
    def __init__(self, env):
        super().__init__(env)
        self._orig_action_space = env.action_space
        if isinstance(self._orig_action_space, (SpaceDict, SpaceTuple)):
            self.action_space = flatten_space(self._orig_action_space)
            self._needs_unflatten = True
        elif isinstance(self._orig_action_space, Box):
            self.action_space = self._orig_action_space
            self._needs_unflatten = False
        else:
            raise TypeError(f"Unsupported action space type: {type(self._orig_action_space)}")

    def action(self, action: np.ndarray):
        if self._needs_unflatten:
            return unflatten(self._orig_action_space, action)
        return action


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalize observations using running statistics.
    This helps prevent extreme values that can cause NaN gradients.
    """
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.obs_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.obs_count = epsilon

    def observation(self, obs):
        # Update running statistics
        batch_mean = obs
        batch_var = np.zeros_like(obs)
        batch_count = 1

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count

        self.obs_mean += delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.obs_count * batch_count / total_count
        self.obs_var = M2 / total_count
        self.obs_count = total_count

        # Normalize
        normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(normalized, -10, 10)  # Clip to prevent extreme values


def make_env(normalize=True):
    # Use the namespaced ID; omit render_mode unless the env docs say it exists
    env = gym.make("gymtorax/IterHybrid-v0")

    # Flatten observations first (SB3 barfs on nested Dict/Tuple)
    if isinstance(env.observation_space, (SpaceDict, SpaceTuple)) or not isinstance(env.observation_space, Box):
        env = FlattenObsWrapper(env)

    # Add normalization to help with training stability
    if normalize:
        env = NormalizeObservation(env)

    # Flatten actions if needed (SB3 SAC needs a single Box)
    if isinstance(env.action_space, (SpaceDict, SpaceTuple)) or not isinstance(env.action_space, Box):
        env = FlattenActionWrapper(env)

    # After wrapping, SB3 will see Box obs & Box action
    assert isinstance(env.observation_space, Box), "Obs must be Box after wrapping"
    assert isinstance(env.action_space, Box), "Action must be Box after wrapping"
    return env


def main():
    env = make_env()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test the environment first
    print("\nTesting environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Observation contains NaN: {np.any(np.isnan(obs))}")
    print(f"Observation contains Inf: {np.any(np.isinf(obs))}")
    
    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"After step - Observation contains NaN: {np.any(np.isnan(obs))}")
    print(f"After step - Reward: {reward}")

    # Configure SAC with more conservative hyperparameters
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-3,  # Lower learning rate
        buffer_size=30000,  # Reduced buffer size to fit in memory (~1.4 GB)
        learning_starts=1000,  # Start learning after some random exploration
        batch_size=256,
        tau=0.005,  # Soft update coefficient
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',  # Automatic entropy tuning
        target_update_interval=1,
        policy_kwargs=dict(
            net_arch=[512, 512],  # Smaller network
            activation_fn=nn.Tanh
        )
    )
    
    # Add callbacks for monitoring
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path='./logs/',
        name_prefix='sac_torax'
    )
    
    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=50000, 
            log_interval=4,
            callback=checkpoint_callback
        )
        model.save("sac_torax_flat")
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()