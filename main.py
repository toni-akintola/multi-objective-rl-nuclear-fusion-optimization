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
        agent.reset_state(observation)  # Initialize shape tracking
        episode_reward = 0
        steps = 0
        terminated = False
        truncated = False

        shape_violations_count = 0
        while not (terminated or truncated):
            action = agent.act(observation)  # action
            observation, reward, terminated, truncated, info = env.step(action)
            original_reward = reward
            reward = agent.apply_shape_safety(reward, observation)  # Apply shape penalty
            
            # Track violations for logging
            if agent.last_shape_info and not agent.last_shape_info["ok"]:
                shape_violations_count += 1
            
            episode_reward += reward
            steps += 1

            # Safety check for infinite loops
            if steps > 1000:
                print(f"Episode {episode + 1}: Hit max steps (1000)")
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        violation_info = f", Shape Violations = {shape_violations_count}" if shape_violations_count > 0 else ""
        print(
            f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {steps}{violation_info}"
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
    plt.close()  # Close to free memory, will show all plots at the end

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
    # Create environment to get action space
    env = gym.make("gymtorax/IterHybrid-v0")
    
    # Create agent WITHOUT shape guard (baseline)
    agent_no_guard = RandomAgent(
        action_space=env.action_space,
        shape_penalty=0.0,  # Shape guard OFF
    )
    
    # Create agent WITH shape guard (safety enabled)
    # Using lower penalty to avoid overwhelming the reward signal
    agent_with_guard = RandomAgent(
        action_space=env.action_space,
        shape_penalty=0.1,  # Reduced penalty coefficient (was 1.0)
        damp_on_violation=True,  # Reduce action magnitude on violations
        damp_factor=0.5,
    )
    env.close()
    
    print("=" * 60)
    print("Running WITHOUT shape guard (baseline)")
    print("=" * 60)
    rewards_no_guard, lengths_no_guard = run(agent_no_guard, num_episodes=10)
    
    print("\n" + "=" * 60)
    print("Running WITH shape guard (safety enabled)")
    print("=" * 60)
    rewards_with_guard, lengths_with_guard = run(agent_with_guard, num_episodes=10)
    
    # Comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Without Shape Guard:")
    print(f"  Mean Reward: {np.mean(rewards_no_guard):.2f} ± {np.std(rewards_no_guard):.2f}")
    print(f"  Mean Steps:  {np.mean(lengths_no_guard):.1f} ± {np.std(lengths_no_guard):.1f}")
    print(f"\nWith Shape Guard:")
    print(f"  Mean Reward: {np.mean(rewards_with_guard):.2f} ± {np.std(rewards_with_guard):.2f}")
    print(f"  Mean Steps:  {np.mean(lengths_with_guard):.1f} ± {np.std(lengths_with_guard):.1f}")
    print("=" * 60)