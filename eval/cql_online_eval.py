"""
Evaluate CQL agent trained offline in the online IterHybrid environment.
"""

import argparse
import numpy as np
import gymnasium as gym
import gymtorax
import d3rlpy
from pathlib import Path


class CQLPolicyWrapper:
    """
    Wrapper to use d3rlpy CQL agent in gymtorax environment.

    Handles:
    - Observation flattening/normalization
    - Action prediction
    - Action space compatibility
    """

    def __init__(self, cql_agent, obs_mean=None, obs_std=None):
        """
        Args:
            cql_agent: Trained d3rlpy CQL agent
            obs_mean: Mean for observation normalization (optional)
            obs_std: Std for observation normalization (optional)
        """
        self.cql_agent = cql_agent
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def predict(self, observation, deterministic=True):
        """
        Predict action from observation.

        Args:
            observation: Raw observation from environment (dict or array)
            deterministic: Whether to use deterministic policy

        Returns:
            action: Action compatible with environment
        """
        # Flatten observation if it's a dict
        if isinstance(observation, dict):
            obs_flat = self._flatten_observation(observation)
        else:
            obs_flat = observation

        # Normalize if stats provided
        if self.obs_mean is not None and self.obs_std is not None:
            obs_flat = (obs_flat - self.obs_mean) / (self.obs_std + 1e-8)

        # Get action from CQL agent
        action = self.cql_agent.predict(obs_flat)

        return action

    def _flatten_observation(self, obs_dict):
        """Flatten observation dictionary to array."""
        # This should match how you built the dataset
        # For AllObservation, it's already flattened in the environment
        if isinstance(obs_dict, np.ndarray):
            return obs_dict

        # If it's a dict, concatenate all values
        obs_list = []
        for key in sorted(obs_dict.keys()):
            val = obs_dict[key]
            if isinstance(val, (int, float)):
                obs_list.append(val)
            elif isinstance(val, np.ndarray):
                obs_list.extend(val.flatten())

        return np.array(obs_list, dtype=np.float32)


def evaluate_cql_online(
    model_path: str,
    n_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True,
):
    """
    Evaluate trained CQL agent in the online environment.

    Args:
        model_path: Path to saved CQL model (.d3 file)
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        deterministic: Whether to use deterministic policy
    """
    print("=" * 60)
    print("CQL Online Evaluation in IterHybrid Environment")
    print("=" * 60)

    # Load trained CQL agent
    print(f"\nLoading CQL agent from {model_path}...")
    cql_agent = d3rlpy.load_learnable(model_path)
    print("Agent loaded successfully!")

    # Create environment
    print("\nCreating IterHybrid environment...")
    env = gym.make("gymtorax/IterHybrid-v0")
    print(f"Environment created!")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Wrap the CQL agent
    policy = CQLPolicyWrapper(cql_agent)

    # Evaluate
    print(f"\nEvaluating for {n_episodes} episodes...")
    print("=" * 60)

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not (done or truncated):
            # Get action from CQL policy
            action = policy.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(
            f"Episode {episode + 1}/{n_episodes}: "
            f"Reward = {episode_reward:.4f}, Length = {episode_length}"
        )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(
        f"Mean reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}"
    )
    print(f"Min reward:  {np.min(episode_rewards):.4f}")
    print(f"Max reward:  {np.max(episode_rewards):.4f}")
    print(
        f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    print("=" * 60)

    env.close()

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CQL agent in online IterHybrid environment"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved CQL model (.d3 file)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        print("\nTo download from Modal:")
        print("  modal volume get cql-models models/cql_torax/cql_final.d3 ./logs/")
        return

    # Run evaluation
    results = evaluate_cql_online(
        model_path=args.model_path,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
