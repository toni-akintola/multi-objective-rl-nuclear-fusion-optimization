import gymnasium as gym
import gymtorax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import Agent, RandomAgent


def run(agent: Agent, num_episodes=10, track_shape=True):
    """Run random agent for multiple episodes and track rewards."""
    env = gym.make("gymtorax/IterHybrid-v0")

    episode_rewards = []
    episode_lengths = []
    
    # Track shape parameters for visualization
    shape_history = [] if track_shape else None  # List of (episode, step, shape_info, penalty)
    all_shape_data = [] if track_shape else None  # All shape vectors for 3D plot

    for episode in tqdm(range(num_episodes)):
        observation, info = env.reset()
        agent.reset_state(observation)  # Initialize shape tracking
        episode_reward = 0
        steps = 0
        terminated = False
        truncated = False

        shape_violations_count = 0
        episode_shape_data = []
        
        while not (terminated or truncated):
            action = agent.act(observation)  # action
            observation, reward, terminated, truncated, info = env.step(action)
            original_reward = reward
            reward = agent.apply_shape_safety(reward, observation)  # Apply shape penalty
            
            # Track shape data for visualization
            if track_shape and agent.last_shape_info:
                shape_info = agent.last_shape_info
                penalty = original_reward - reward if not shape_info["ok"] else 0.0
                shape_history.append({
                    "episode": episode,
                    "step": steps,
                    "beta_N": shape_info["shape"][0],
                    "q_min": shape_info["shape"][1],
                    "q95": shape_info["shape"][2],
                    "in_box": shape_info["in_box"],
                    "smooth": shape_info["smooth"],
                    "ok": shape_info["ok"],
                    "severity": shape_info["severity"],
                    "penalty": penalty,
                })
                episode_shape_data.append(shape_info["shape"])
                
                # Track violations for logging
                if not shape_info["ok"]:
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
        
        if track_shape:
            all_shape_data.extend(episode_shape_data)

    env.close()
    
    # Plot basic results
    if not track_shape:
        return episode_rewards, episode_lengths, None, None
    
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

    return episode_rewards, episode_lengths, shape_history, all_shape_data


def visualize_shape_guard(shape_history, all_shape_data, agent_name="Agent"):
    """Create comprehensive visualizations of shape guard behavior."""
    if not shape_history or len(shape_history) == 0:
        print("No shape data to visualize")
        return
    
    # Import shape guard constraints for visualization
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "shape_guard",
        Path(__file__).parent / "optimization-for-constraints" / "shape_guard.py"
    )
    shape_guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shape_guard)
    constraints = shape_guard.DEFAULT_CONSTRAINTS
    
    # Convert to arrays for easier plotting
    beta_N = np.array([s["beta_N"] for s in shape_history])
    q_min = np.array([s["q_min"] for s in shape_history])
    q95 = np.array([s["q95"] for s in shape_history])
    violations = np.array([not s["ok"] for s in shape_history])
    severity = np.array([s["severity"] for s in shape_history])
    penalties = np.array([s["penalty"] for s in shape_history])
    steps = np.array([s["step"] for s in shape_history])
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Shape parameters over time with safe box boundaries
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(steps, beta_N, 'b-', alpha=0.6, label='β_N', linewidth=1)
    ax1.axhspan(constraints.beta_n_min, constraints.beta_n_max, alpha=0.2, color='green', label='Safe Zone')
    ax1.axhline(constraints.beta_n_min, 'g--', linewidth=2)
    ax1.axhline(constraints.beta_n_max, 'g--', linewidth=2)
    ax1.scatter(steps[violations], beta_N[violations], c='red', s=20, alpha=0.7, label='Violations', zorder=5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('β_N (Normalized Beta)')
    ax1.set_title('β_N Over Time')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(steps, q_min, 'r-', alpha=0.6, label='q_min', linewidth=1)
    ax2.axhspan(constraints.q_min_min, max(q_min)*1.1, alpha=0.2, color='green', label='Safe Zone')
    ax2.axhline(constraints.q_min_min, 'g--', linewidth=2)
    ax2.scatter(steps[violations], q_min[violations], c='red', s=20, alpha=0.7, label='Violations', zorder=5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('q_min (Minimum Safety Factor)')
    ax2.set_title('q_min Over Time')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(steps, q95, 'g-', alpha=0.6, label='q95', linewidth=1)
    ax3.axhspan(constraints.q95_min, constraints.q95_max, alpha=0.2, color='green', label='Safe Zone')
    ax3.axhline(constraints.q95_min, 'g--', linewidth=2)
    ax3.axhline(constraints.q95_max, 'g--', linewidth=2)
    ax3.scatter(steps[violations], q95[violations], c='red', s=20, alpha=0.7, label='Violations', zorder=5)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('q95 (Edge Safety Factor)')
    ax3.set_title('q95 Over Time')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 2. Violation severity over time
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(steps, severity, 'orange', alpha=0.7, linewidth=1)
    ax4.fill_between(steps, 0, severity, where=(severity > 0), alpha=0.3, color='red')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Severity')
    ax4.set_title('Shape Violation Severity')
    ax4.grid(True, alpha=0.3)
    
    # 3. Penalties applied
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(steps, penalties, 'purple', alpha=0.7, linewidth=1)
    ax5.fill_between(steps, 0, penalties, where=(penalties > 0), alpha=0.3, color='purple')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Penalty Applied')
    ax5.set_title('Shape Penalties Over Time')
    ax5.grid(True, alpha=0.3)
    
    # 4. 2D Projection: beta_N vs q95
    ax6 = plt.subplot(3, 3, 6)
    safe_mask = ~violations
    ax6.scatter(beta_N[safe_mask], q95[safe_mask], c='green', s=10, alpha=0.5, label='Safe')
    ax6.scatter(beta_N[violations], q95[violations], c='red', s=20, alpha=0.7, label='Violations', zorder=5)
    # Draw safe box
    rect = plt.Rectangle((constraints.beta_n_min, constraints.q95_min), 
                        constraints.beta_n_max - constraints.beta_n_min,
                        constraints.q95_max - constraints.q95_min,
                        fill=False, edgecolor='green', linewidth=2, linestyle='--', label='Safe Box')
    ax6.add_patch(rect)
    ax6.set_xlabel('β_N')
    ax6.set_ylabel('q95')
    ax6.set_title('β_N vs q95 (2D Projection)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 5. 2D Projection: q_min vs q95
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(q_min[safe_mask], q95[safe_mask], c='green', s=10, alpha=0.5, label='Safe')
    ax7.scatter(q_min[violations], q95[violations], c='red', s=20, alpha=0.7, label='Violations', zorder=5)
    # Draw safe box
    ax7.axvline(constraints.q_min_min, color='green', linestyle='--', linewidth=2)
    ax7.axhline(constraints.q95_min, color='green', linestyle='--', linewidth=2)
    ax7.axhline(constraints.q95_max, color='green', linestyle='--', linewidth=2)
    ax7.set_xlabel('q_min')
    ax7.set_ylabel('q95')
    ax7.set_title('q_min vs q95 (2D Projection)')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 6. Violation statistics
    ax8 = plt.subplot(3, 3, 8)
    violation_types = {
        'In Box': np.sum([s["in_box"] and not s["smooth"] for s in shape_history]),
        'Smooth': np.sum([s["smooth"] and not s["in_box"] for s in shape_history]),
        'Both': np.sum([not s["in_box"] and not s["smooth"] for s in shape_history]),
        'OK': np.sum([s["ok"] for s in shape_history])
    }
    colors = ['red', 'orange', 'darkred', 'green']
    ax8.bar(violation_types.keys(), violation_types.values(), color=colors, alpha=0.7)
    ax8.set_ylabel('Count')
    ax8.set_title('Violation Type Distribution')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 7. Cumulative violations
    ax9 = plt.subplot(3, 3, 9)
    cumulative_violations = np.cumsum(violations)
    ax9.plot(steps, cumulative_violations, 'r-', linewidth=2)
    ax9.set_xlabel('Step')
    ax9.set_ylabel('Cumulative Violations')
    ax9.set_title(f'Total Violations: {int(cumulative_violations[-1])}')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'Shape Guard Visualization - {agent_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"shape_guard_visualization_{agent_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Shape guard visualization saved to: {filename}")
    plt.close()


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
    rewards_no_guard, lengths_no_guard, _, _ = run(agent_no_guard, num_episodes=10, track_shape=False)
    
    print("\n" + "=" * 60)
    print("Running WITH shape guard (safety enabled)")
    print("=" * 60)
    rewards_with_guard, lengths_with_guard, shape_history, all_shape_data = run(agent_with_guard, num_episodes=10, track_shape=True)
    
    # Visualize shape guard behavior
    if shape_history:
        print("\n" + "=" * 60)
        print("Generating shape guard visualizations...")
        print("=" * 60)
        visualize_shape_guard(shape_history, all_shape_data, agent_name="With Shape Guard")
    
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
    if shape_history:
        total_violations = sum(not s["ok"] for s in shape_history)
        print(f"  Total Violations: {total_violations}")
        print(f"  Violation Rate: {total_violations/len(shape_history)*100:.1f}%")
    print("=" * 60)