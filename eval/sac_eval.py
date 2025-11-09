# eval_sac_torax.py
import argparse
import numpy as np
import gymnasium as gym
import gymtorax  # registers envs
from gymnasium.spaces import Dict as SpaceDict, Tuple as SpaceTuple, Box
from gymnasium.spaces.utils import flatten_space, flatten, unflatten

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn


# ---------- Wrappers (same as training) ----------
class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._orig_obs_space = env.observation_space
        self.observation_space = flatten_space(self._orig_obs_space)

    def observation(self, obs):
        flat = flatten(self._orig_obs_space, obs)
        if np.any(~np.isfinite(flat)):
            flat = np.nan_to_num(flat, nan=0.0, posinf=1e6, neginf=-1e6)
        return flat


class FlattenActionWrapper(gym.ActionWrapper):
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
            raise TypeError(
                f"Unsupported action space: {type(self._orig_action_space)}"
            )

    def action(self, action):
        return (
            unflatten(self._orig_action_space, action)
            if self._needs_unflatten
            else action
        )


class NormalizeObservation(gym.ObservationWrapper):
    """Running-stats normalization; can be frozen for eval."""

    def __init__(self, env, epsilon=1e-8, update_stats=True):
        super().__init__(env)
        self.epsilon = epsilon
        self.update_stats = update_stats
        self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.obs_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.obs_count = float(epsilon)

    def freeze(self):
        self.update_stats = False

    def unfreeze(self):
        self.update_stats = True

    def get_state(self):
        return dict(
            mean=self.obs_mean.copy(),
            var=self.obs_var.copy(),
            count=float(self.obs_count),
        )

    def set_state(self, state):
        self.obs_mean = state["mean"].copy()
        self.obs_var = state["var"].copy()
        self.obs_count = float(state["count"])

    def observation(self, obs):
        if self.update_stats:
            batch_mean, batch_var, batch_count = obs, np.zeros_like(obs), 1
            delta = batch_mean - self.obs_mean
            total = self.obs_count + batch_count
            self.obs_mean += delta * batch_count / total
            m_a = self.obs_var * self.obs_count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + np.square(delta) * self.obs_count * batch_count / total
            self.obs_var = m2 / total
            self.obs_count = total
        norm = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(norm, -10, 10)


def make_env(normalize=True):
    env = gym.make("gymtorax/IterHybrid-v0")
    if isinstance(env.observation_space, (SpaceDict, SpaceTuple)) or not isinstance(
        env.observation_space, Box
    ):
        env = FlattenObsWrapper(env)
    if normalize:
        env = NormalizeObservation(env)
    if isinstance(env.action_space, (SpaceDict, SpaceTuple)) or not isinstance(
        env.action_space, Box
    ):
        env = FlattenActionWrapper(env)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Box)
    return env


def find_wrapper(env, wrapper_type):
    cur = env
    while isinstance(cur, gym.Wrapper):
        if isinstance(cur, wrapper_type):
            return cur
        cur = cur.env
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="sac_torax_flat.zip")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--det", action="store_true", help="deterministic policy")
    args = ap.parse_args()

    # Build eval env and load model
    eval_env = make_env(normalize=True)
    model = SAC.load(args.model, env=eval_env, device="auto")

    # Freeze normalization (use the stats as-is for fair eval)
    norm = find_wrapper(eval_env, NormalizeObservation)
    if norm is not None:
        norm.freeze()

    # Evaluate
    print(
        f"\nEvaluating {args.model} for {args.episodes} episodes (deterministic={args.det})..."
    )
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=args.episodes, deterministic=args.det
    )
    print(f"Mean episodic return: {mean_r:.4f} ± {std_r:.4f}")

    ep_rewards, ep_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.episodes,
        deterministic=args.det,
        return_episode_rewards=True,
    )
    print("Per-episode returns:", [f"{r:.4f}" for r in ep_rewards])
    print("Per-episode lengths:", ep_lengths)
    print(f"Avg length: {np.mean(ep_lengths):.1f} ± {np.std(ep_lengths):.1f}")


if __name__ == "__main__":
    main()
