import gymnasium as gym
import gymtorax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import Agent, RandomAgent


def run(agent: Agent, num_episodes=10):
    """Run random agent for multiple episodes and track rewards."""
    env = gym.make("gymtorax/IterHybrid-v0")

    episode_rewards = []
    episode_lengths = []

    for episode in tqdm(range(num_episodes)):
        observation, info = env.reset()
        episode_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.act()  # action
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            # Safety check for infinite loops
            if steps > 1000:
                print(f"Episode {episode + 1}: Hit max steps (1000)")
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(
            f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {steps}"
        )

    env.close()

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot rewards
    ax1.plot(
        range(1, num_episodes + 1),
        episode_rewards,
        marker="o",
        linestyle="-",
        linewidth=2,
    )
    ax1.axhline(
        y=np.mean(episode_rewards),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(episode_rewards):.2f}",
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Random Agent Performance")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot episode lengths
    ax2.plot(
        range(1, num_episodes + 1),
        episode_lengths,
        marker="s",
        linestyle="-",
        linewidth=2,
        color="green",
    )
    ax2.axhline(
        y=np.mean(episode_lengths),
        color="r",
        linestyle="--",
        label=f"Mean: {np.mean(episode_lengths):.1f}",
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length (Steps)")
    ax2.set_title("Episode Duration")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("random_baseline.png", dpi=150)
    print(f"\nPlot saved to: random_baseline.png")
    plt.show()

    # Print statistics
    print(f"\n{'='*50}")
    print(f"Random Agent Baseline Statistics ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(
        f"Reward  - Mean: {np.mean(episode_rewards):>10.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(f"        - Min:  {np.min(episode_rewards):>10.2f}")
    print(f"        - Max:  {np.max(episode_rewards):>10.2f}")
    print(
        f"Steps   - Mean: {np.mean(episode_lengths):>10.1f} ± {np.std(episode_lengths):.1f}"
    )
    print(f"        - Min:  {np.min(episode_lengths):>10}")
    print(f"        - Max:  {np.max(episode_lengths):>10}")
    print(f"{'='*50}")

    return episode_rewards, episode_lengths


if __name__ == "__main__":
    env = gym.make("gymtorax/IterHybrid-v0")
    # Run baseline with random actions
    rewards, lengths = run(agent=RandomAgent(action_space=env.action_space), num_episodes=10)