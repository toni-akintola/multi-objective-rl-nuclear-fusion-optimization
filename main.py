# eval_random_torax.py
import argparse
import numpy as np
import gymnasium as gym
import gymtorax  # registers envs
from gymnasium.spaces import Dict as SpaceDict, Tuple as SpaceTuple, Box
from gymnasium.spaces.utils import flatten_space, flatten, unflatten

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


# ---------- Wrappers (identical to your train/eval) ----------
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
        env.freeze()  # freeze stats for fair eval
    if isinstance(env.action_space, (SpaceDict, SpaceTuple)) or not isinstance(
        env.action_space, Box
    ):
        env = FlattenActionWrapper(env)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Box)
    return env


# ---------- Random policy that handles vec obs ----------
class RandomPolicy:
    """
    SB3-compatible .predict(). If obs is batched (n_envs, obs_dim),
    return a batch of actions (n_envs, act_dim).
    """

    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        if seed is not None:
            # Gymnasium spaces have their own RNG; this seeds numpy for any local use
            np.random.seed(seed)

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        # Vectorized obs: (n_envs, ...) -> return (n_envs, act_dim)
        if isinstance(obs, np.ndarray) and obs.ndim >= 2:
            n_envs = obs.shape[0]
            acts = [self.action_space.sample() for _ in range(n_envs)]
            return np.stack(acts), None
        # Single obs
        return self.action_space.sample(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # Build a *vectorized* env with monitor to satisfy evaluate_policy expectations
    vec_env = DummyVecEnv([lambda: make_env(normalize=True)])
    vec_env = VecMonitor(vec_env)

    if args.seed is not None:
        try:
            vec_env.seed(args.seed)
        except Exception:
            pass

    model = RandomPolicy(vec_env.action_space, seed=args.seed)

    print(f"\nEvaluating RANDOM policy for {args.episodes} episodes...")
    mean_r, std_r = evaluate_policy(
        model, vec_env, n_eval_episodes=args.episodes, deterministic=False
    )
    print(f"Mean episodic return: {mean_r:.4f} ± {std_r:.4f}")

    ep_rewards, ep_lengths = evaluate_policy(
        model,
        vec_env,
        n_eval_episodes=args.episodes,
        deterministic=False,
        return_episode_rewards=True,
    )
    print("Per-episode returns:", [f"{r:.4f}" for r in ep_rewards])
    print("Per-episode lengths:", ep_lengths)
    print(f"Avg length: {np.mean(ep_lengths):.1f} ± {np.std(ep_lengths):.1f}")


if __name__ == "__main__":
    main()
