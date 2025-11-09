# eval_random_torax.py
import argparse
import numpy as np
import gymnasium as gym
import gymtorax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from agent import Agent, RandomAgent
import importlib.util
from pathlib import Path

# Import shape_violation for diagnostics
spec = importlib.util.spec_from_file_location(
    "shape_guard",
    Path(__file__).parent / "optimization-for-constraints" / "shape_guard.py"
)
shape_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape_guard)
shape_violation = shape_guard.shape_violation


def run(agent: Agent, num_episodes=10, track_shape=False, interactive=False):
    """Run random agent for multiple episodes and track rewards."""
    env = gym.make("gymtorax/IterHybrid-v0")

    episode_rewards = []
    episode_lengths = []
    
    # Track shape data for visualization
    shape_history = [] if track_shape else None

    for episode in tqdm(range(num_episodes)):
        observation, info = env.reset()
        agent.reset_state(observation)  # Initialize shape tracking
        episode_reward = 0
        steps = 0
        terminated = False
        truncated = False
        shape_violations_count = 0
        corrective_actions_count = 0
        total_penalty = 0.0
        
        # Diagnostic: Check initial state
        if episode == 0 and agent.shape_penalty > 0:
            initial_info = shape_violation(None, observation)
            print(f"\n[Diagnostic] Initial state: β_N={initial_info['shape'][0]:.3f}, "
                  f"q_min={initial_info['shape'][1]:.3f}, q95={initial_info['shape'][2]:.3f}, "
                  f"OK={initial_info['ok']}, In Box={initial_info['in_box']}, Smooth={initial_info['smooth']}")

        while not (terminated or truncated):
            action = agent.act(observation)  # action
            observation, reward, terminated, truncated, info = env.step(action)
            original_reward = reward
            reward = agent.apply_shape_safety(reward, observation)  # Apply shape penalty
            
            # Interactive step-by-step display
            if interactive and agent.last_shape_info:
                shape_info = agent.last_shape_info
                # Get previous severity from history if available
                prev_severity = shape_history[-1]["severity"] if shape_history else float('inf')
                
                is_corrective = (not shape_info["ok"] and 
                               prev_severity != float('inf') and 
                               shape_info["severity"] < prev_severity)
                
                status = "🟢 SAFE" if shape_info["ok"] else ("🟠 SELF-FIXING!" if is_corrective else "🔴 VIOLATION")
                severity_change = ""
                if prev_severity != float('inf'):
                    if is_corrective:
                        severity_change = f"↓ (was {prev_severity:.3f})"
                    elif shape_info["severity"] > prev_severity:
                        severity_change = f"↑ (was {prev_severity:.3f})"
                
                print(f"\n  Step {steps+1}:")
                print(f"    Shape: β_N={shape_info['shape'][0]:.3f}, q_min={shape_info['shape'][1]:.3f}, q95={shape_info['shape'][2]:.3f}")
                print(f"    Status: {status}")
                print(f"    In Safe Box: {shape_info['in_box']} | Smooth: {shape_info['smooth']}")
                print(f"    Severity: {shape_info['severity']:.3f} {severity_change}")
                penalty_applied = original_reward - reward
                if penalty_applied > 0:
                    print(f"    Reward: {original_reward:.3f} → {reward:.3f} (penalty: -{penalty_applied:.3f})")
                else:
                    print(f"    Reward: {original_reward:.3f} → {reward:.3f} (no penalty)")
                if is_corrective:
                    print(f"    ⭐ Corrective action! Severity reduced from {prev_severity:.3f} to {shape_info['severity']:.3f}")
            
            # Track violations and penalties
            if agent.last_shape_info:
                if not agent.last_shape_info["ok"]:
                    shape_violations_count += 1
                    penalty = original_reward - reward
                    total_penalty += penalty
                    # Check if it was corrective
                    if hasattr(agent, 'last_shape_info') and agent.last_shape_info.get('corrective', False):
                        corrective_actions_count += 1
                
                # Track shape data for visualization
                if track_shape and agent.shape_penalty > 0:
                    shape_info = agent.last_shape_info
                    prev_severity = shape_history[-1]["severity"] if shape_history else float('inf')
                    is_corrective = (not shape_info["ok"] and 
                                   prev_severity != float('inf') and 
                                   shape_info["severity"] < prev_severity)
                    
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
                        "penalty": original_reward - reward if not shape_info["ok"] and not is_corrective else 0.0,
                        "corrective": is_corrective,
                    })
            
            episode_reward += reward
            steps += 1

            # Safety check for infinite loops
            if steps > 1000:
                print(f"Episode {episode + 1}: Hit max steps (1000)")
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        if agent.shape_penalty > 0 and shape_violations_count > 0:
            violation_info = f", Violations = {shape_violations_count}, Corrective = {corrective_actions_count}, Total Penalty = {total_penalty:.2f}"
        else:
            violation_info = ""
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
    output_path = Path(__file__).parent / "results" / "random_baseline.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
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

    return episode_rewards, episode_lengths, shape_history


def visualize_shape_self_fixing(shape_history, agent_name="Agent"):
    """Create a cool visualization showing how the shape guard self-fixes the plasma shape."""
    if not shape_history or len(shape_history) == 0:
        print("No shape data to visualize")
        return
    
    # Get constraints from shape_guard module
    constraints = type('Constraints', (), {
        'beta_n_min': shape_guard.BETA_N_MIN,
        'beta_n_max': shape_guard.BETA_N_MAX,
        'q_min_min': shape_guard.QMIN_MIN,
        'q95_min': shape_guard.Q95_MIN,
        'q95_max': shape_guard.Q95_MAX,
    })()
    
    # Convert to arrays
    beta_N = np.array([s["beta_N"] for s in shape_history])
    q_min = np.array([s["q_min"] for s in shape_history])
    q95 = np.array([s["q95"] for s in shape_history])
    violations = np.array([not s["ok"] for s in shape_history])
    corrective = np.array([s.get("corrective", False) for s in shape_history])
    severity = np.array([s["severity"] for s in shape_history])
    steps = np.array([s["step"] for s in shape_history])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Shape parameters over time with safe zones (top row)
    ax1 = plt.subplot(3, 3, 1)
    safe_mask = ~violations
    ax1.plot(steps, beta_N, 'b-', alpha=0.3, linewidth=1, label='β_N')
    ax1.scatter(steps[safe_mask], beta_N[safe_mask], c='green', s=15, alpha=0.6, label='Safe', zorder=3)
    ax1.scatter(steps[violations & ~corrective], beta_N[violations & ~corrective], 
                c='red', s=30, alpha=0.8, marker='x', label='Violation', zorder=4)
    ax1.scatter(steps[corrective], beta_N[corrective], 
                c='orange', s=50, alpha=0.9, marker='*', label='Self-Fixing!', zorder=5)
    ax1.axhspan(constraints.beta_n_min, constraints.beta_n_max, alpha=0.15, color='green', label='Safe Zone')
    ax1.axhline(constraints.beta_n_min, color='g', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(constraints.beta_n_max, color='g', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('β_N (Normalized Beta)')
    ax1.set_title('β_N: Self-Fixing Trajectory')
    ax1.legend(fontsize=7, loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(steps, q_min, 'r-', alpha=0.3, linewidth=1, label='q_min')
    ax2.scatter(steps[safe_mask], q_min[safe_mask], c='green', s=15, alpha=0.6, label='Safe', zorder=3)
    ax2.scatter(steps[violations & ~corrective], q_min[violations & ~corrective], 
                c='red', s=30, alpha=0.8, marker='x', label='Violation', zorder=4)
    ax2.scatter(steps[corrective], q_min[corrective], 
                c='orange', s=50, alpha=0.9, marker='*', label='Self-Fixing!', zorder=5)
    ax2.axhspan(constraints.q_min_min, max(q_min)*1.1, alpha=0.15, color='green', label='Safe Zone')
    ax2.axhline(constraints.q_min_min, color='g', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('q_min (Minimum Safety Factor)')
    ax2.set_title('q_min: Self-Fixing Trajectory')
    ax2.legend(fontsize=7, loc='best')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(steps, q95, 'g-', alpha=0.3, linewidth=1, label='q95')
    ax3.scatter(steps[safe_mask], q95[safe_mask], c='green', s=15, alpha=0.6, label='Safe', zorder=3)
    ax3.scatter(steps[violations & ~corrective], q95[violations & ~corrective], 
                c='red', s=30, alpha=0.8, marker='x', label='Violation', zorder=4)
    ax3.scatter(steps[corrective], q95[corrective], 
                c='orange', s=50, alpha=0.9, marker='*', label='Self-Fixing!', zorder=5)
    ax3.axhspan(constraints.q95_min, constraints.q95_max, alpha=0.15, color='green', label='Safe Zone')
    ax3.axhline(constraints.q95_min, color='g', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axhline(constraints.q95_max, color='g', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('q95 (Edge Safety Factor)')
    ax3.set_title('q95: Self-Fixing Trajectory')
    ax3.legend(fontsize=7, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 2. Severity over time showing self-fixing (middle left)
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(steps, severity, 'purple', alpha=0.5, linewidth=1.5, label='Severity')
    ax4.fill_between(steps, 0, severity, where=(severity > 0), alpha=0.3, color='red', label='Violation')
    ax4.scatter(steps[corrective], severity[corrective], 
                c='orange', s=60, alpha=1.0, marker='*', label='Self-Fixing!', zorder=5, edgecolors='darkorange', linewidths=1)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Severity')
    ax4.set_title('Severity Reduction (Self-Fixing)')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    
    # 3. 2D Trajectory: β_N vs q95 showing path (middle center)
    ax5 = plt.subplot(3, 3, 5)
    # Draw trajectory with arrows showing direction
    for i in range(len(beta_N)-1):
        if violations[i] or violations[i+1]:
            color = 'orange' if corrective[i+1] else 'red'
            alpha = 0.8 if corrective[i+1] else 0.4
            ax5.plot([beta_N[i], beta_N[i+1]], [q95[i], q95[i+1]], 
                    color=color, alpha=alpha, linewidth=1.5, zorder=2)
        else:
            ax5.plot([beta_N[i], beta_N[i+1]], [q95[i], q95[i+1]], 
                    color='green', alpha=0.3, linewidth=1, zorder=1)
    
    # Mark points
    ax5.scatter(beta_N[safe_mask], q95[safe_mask], c='green', s=20, alpha=0.7, label='Safe', zorder=3)
    ax5.scatter(beta_N[violations & ~corrective], q95[violations & ~corrective], 
                c='red', s=40, alpha=0.8, marker='x', label='Violation', zorder=4)
    ax5.scatter(beta_N[corrective], q95[corrective], 
                c='orange', s=80, alpha=1.0, marker='*', label='Self-Fixing!', zorder=5, 
                edgecolors='darkorange', linewidths=1.5)
    
    # Draw safe box
    rect = plt.Rectangle((constraints.beta_n_min, constraints.q95_min), 
                        constraints.beta_n_max - constraints.beta_n_min,
                        constraints.q95_max - constraints.q95_min,
                        fill=False, edgecolor='green', linewidth=2.5, linestyle='--', 
                        label='Safe Operating Box', zorder=6)
    ax5.add_patch(rect)
    
    # Mark start and end
    ax5.scatter(beta_N[0], q95[0], c='blue', s=150, marker='o', 
                label='Start', zorder=7, edgecolors='darkblue', linewidths=2)
    ax5.scatter(beta_N[-1], q95[-1], c='purple', s=150, marker='s', 
                label='End', zorder=7, edgecolors='darkviolet', linewidths=2)
    
    ax5.set_xlabel('β_N')
    ax5.set_ylabel('q95')
    ax5.set_title('2D Trajectory: Moving Toward Safety')
    ax5.legend(fontsize=6, loc='best')
    ax5.grid(True, alpha=0.3)
    
    # 4. Recovery timeline (middle right)
    ax6 = plt.subplot(3, 3, 6)
    cumulative_recovery = np.cumsum(corrective)
    cumulative_violations = np.cumsum(violations & ~corrective)
    ax6.plot(steps, cumulative_recovery, 'orange', linewidth=2.5, label='Self-Fixing Actions', marker='*', markersize=4)
    ax6.plot(steps, cumulative_violations, 'red', linewidth=2, label='New Violations', alpha=0.7)
    ax6.fill_between(steps, 0, cumulative_recovery, alpha=0.2, color='orange')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Cumulative Count')
    ax6.set_title(f'Self-Fixing Progress\n({int(cumulative_recovery[-1])} corrective actions)')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)
    
    # 5. Severity reduction rate (bottom left)
    ax7 = plt.subplot(3, 3, 7)
    severity_change = np.diff(severity)
    ax7.bar(steps[1:][severity_change < 0], severity_change[severity_change < 0], 
            color='orange', alpha=0.7, label='Improving (Self-Fixing)')
    ax7.bar(steps[1:][severity_change >= 0], severity_change[severity_change >= 0], 
            color='red', alpha=0.5, label='Worsening')
    ax7.axhline(0, color='black', linestyle='-', linewidth=1, zorder=0)
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Severity Change')
    ax7.set_title('Severity Change Per Step')
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 6. State distribution (bottom center)
    ax8 = plt.subplot(3, 3, 8)
    state_counts = {
        'Safe': np.sum(safe_mask),
        'Violating': np.sum(violations & ~corrective),
        'Self-Fixing': np.sum(corrective)
    }
    colors = ['green', 'red', 'orange']
    bars = ax8.bar(state_counts.keys(), state_counts.values(), color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax8.set_ylabel('Count')
    ax8.set_title('State Distribution')
    ax8.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # 7. Distance to safe zone (bottom right)
    ax9 = plt.subplot(3, 3, 9)
    # Calculate distance to safe zone center
    safe_center_beta = (constraints.beta_n_min + constraints.beta_n_max) / 2
    safe_center_q95 = (constraints.q95_min + constraints.q95_max) / 2
    distance_to_safe = np.sqrt((beta_N - safe_center_beta)**2 + (q95 - safe_center_q95)**2)
    
    ax9.plot(steps, distance_to_safe, 'purple', linewidth=2, alpha=0.7, label='Distance to Safe Zone')
    ax9.scatter(steps[corrective], distance_to_safe[corrective], 
                c='orange', s=60, alpha=1.0, marker='*', label='Self-Fixing!', zorder=5)
    ax9.fill_between(steps, 0, distance_to_safe, alpha=0.2, color='purple')
    ax9.set_xlabel('Step')
    ax9.set_ylabel('Distance to Safe Zone')
    ax9.set_title('Convergence to Safety')
    ax9.legend(fontsize=7)
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'🔥 Plasma Shape Self-Fixing Visualization - {agent_name} 🔥', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    filename = f"shape_self_fixing_{agent_name.lower().replace(' ', '_')}.png"
    output_path = Path(__file__).parent / "results" / filename
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✨ Shape self-fixing visualization saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Create environment to get action space
    env = gym.make("gymtorax/IterHybrid-v0")
    
    # Create random agent WITHOUT shape guard (baseline)
    agent_no_guard = RandomAgent(
        action_space=env.action_space,
        shape_penalty=0.0,  # Shape guard OFF
    )
    
    # Create random agent WITH shape guard (safety enabled)
    agent_with_guard = RandomAgent(
        action_space=env.action_space,
        shape_penalty=0.1,  # Penalty coefficient (reduced to avoid overwhelming reward)
        damp_on_violation=True,  # Reduce action magnitude on violations
        damp_factor=0.5,
    )
    
    print("=" * 60)
    print("Running WITHOUT shape guard (baseline)")
    print("=" * 60)
    rewards_no_guard, lengths_no_guard, _ = run(agent_no_guard, num_episodes=10, track_shape=False)
    
    print("\n" + "=" * 60)
    print("Running WITH shape guard (safety enabled)")
    print("=" * 60)
    print("💡 Interactive mode: Showing step-by-step shape guard behavior")
    print("=" * 60)
    rewards_with_guard, lengths_with_guard, shape_history = run(
        agent_with_guard, 
        num_episodes=3,  # Reduced for interactive viewing
        track_shape=True, 
        interactive=True
    )
    
    # Visualize shape self-fixing
    if shape_history:
        print("\n" + "=" * 60)
        print("Generating shape self-fixing visualization...")
        print("=" * 60)
        visualize_shape_self_fixing(shape_history, agent_name="With Shape Guard")
    
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
    
    env.close()