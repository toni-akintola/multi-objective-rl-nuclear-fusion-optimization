"""
Interactive Streamlit app for visualizing plasma shape self-fixing.
No TypeScript needed - pure Python!
"""
import streamlit as st
import gymnasium as gym
import gymtorax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import pandas as pd
import time
from agent import RandomAgent
import importlib.util
from pathlib import Path

# Import shape guard
spec = importlib.util.spec_from_file_location(
    "shape_guard",
    Path(__file__).parent / "optimization-for-constraints" / "shape_guard.py"
)
shape_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape_guard)
shape_violation = shape_guard.shape_violation

st.set_page_config(page_title="Plasma Shape Self-Fixing", layout="wide")
st.title("🔥 Plasma Shape Self-Fixing Visualization")
st.markdown("Watch the shape guard self-fix the fusion plasma in real-time!")

# Sidebar controls
st.sidebar.header("Configuration")
shape_penalty = st.sidebar.slider("Shape Penalty", 0.0, 1.0, 0.1, 0.01)
damp_on_violation = st.sidebar.checkbox("Damp Actions on Violation", True)
damp_factor = st.sidebar.slider("Damp Factor", 0.1, 1.0, 0.5, 0.1)
num_episodes = st.sidebar.slider("Number of Episodes", 1, 10, 3)
num_steps_per_episode = st.sidebar.slider("Steps per Episode", 10, 151, 50)

if st.sidebar.button("🚀 Run Simulation", type="primary"):
    # Create environment
    env = gym.make("gymtorax/IterHybrid-v0")
    agent = RandomAgent(
        action_space=env.action_space,
        shape_penalty=shape_penalty,
        damp_on_violation=damp_on_violation,
        damp_factor=damp_factor,
    )
    
    # Track data
    all_shape_data = []
    all_steps_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        agent.reset_state(observation)
        
        episode_data = []
        
        for step in range(num_steps_per_episode):
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            original_reward = reward
            reward = agent.apply_shape_safety(reward, observation)
            
            if agent.last_shape_info:
                shape_info = agent.last_shape_info
                prev_severity = episode_data[-1]["severity"] if episode_data else float('inf')
                is_corrective = (not shape_info["ok"] and 
                               prev_severity != float('inf') and 
                               shape_info["severity"] < prev_severity)
                
                step_data = {
                    "episode": episode,
                    "step": step,
                    "beta_N": shape_info["shape"][0],
                    "q_min": shape_info["shape"][1],
                    "q95": shape_info["shape"][2],
                    "in_box": shape_info["in_box"],
                    "smooth": shape_info["smooth"],
                    "ok": shape_info["ok"],
                    "severity": shape_info["severity"],
                    "corrective": is_corrective,
                    "reward": reward,
                    "original_reward": original_reward,
                    "penalty": original_reward - reward,
                }
                episode_data.append(step_data)
                all_shape_data.append(step_data)
            
            if terminated or truncated:
                break
        
        all_steps_data.append(episode_data)
        progress_bar.progress((episode + 1) / num_episodes)
        status_text.text(f"Episode {episode + 1}/{num_episodes} completed")
    
    env.close()
    
    # Store in session state
    st.session_state.shape_data = all_shape_data
    st.session_state.ready = True
    
    st.success(f"✅ Simulation complete! Collected {len(all_shape_data)} data points.")

if "ready" in st.session_state and st.session_state.ready:
    shape_data = st.session_state.shape_data
    df = pd.DataFrame(shape_data)
    
    # ========== TOKAMAK VISUALIZATION (TOP) ==========
    st.subheader("🔥 Tokamak Plasma Shape Visualization")
    st.markdown("Watch the plasma shape change in real-time inside the tokamak vessel")
    
    # Show diagnostics
    if len(df) > 0:
        in_box_count = df["in_box"].sum()
        smooth_count = df["smooth"].sum()
        ok_count = df["ok"].sum()
        total = len(df)
        
        col_diag1, col_diag2, col_diag3, col_diag4 = st.columns(4)
        with col_diag1:
            st.metric("In Safe Box", f"{in_box_count}/{total}", f"{in_box_count/total*100:.1f}%")
        with col_diag2:
            st.metric("Smooth Changes", f"{smooth_count}/{total}", f"{smooth_count/total*100:.1f}%")
        with col_diag3:
            st.metric("Fully Safe (OK)", f"{ok_count}/{total}", f"{ok_count/total*100:.1f}%")
        with col_diag4:
            st.metric("Self-Fixing", f"{df['corrective'].sum()}/{total}", f"{df['corrective'].sum()/total*100:.1f}%")
    
    def draw_tokamak_plasma(beta_N, q_min, q95, ok, in_box, smooth, corrective, step_num, total_steps):
        """Draw elegant tokamak with smooth swirling particle trails."""
        fig = plt.figure(figsize=(14, 14), facecolor='#000011')
        ax = fig.add_subplot(111, facecolor='#000011')
        
        R0, a0 = 0.0, 1.0
        size_factor = 0.7 + (beta_N - 0.5) / (3.0 - 0.5) * 0.2
        a = a0 * size_factor
        elongation = 1.6 + (q95 - 3.0) / (5.0 - 3.0) * 0.2
        triangularity = 0.45 + (1.0 - q_min) / (1.0 - 0.5) * 0.1
        
        theta = np.linspace(0, 2*np.pi, 600)
        R_plasma = R0 + a * (np.cos(theta) + triangularity * np.cos(2*theta))
        Z_plasma = a * elongation * np.sin(theta)
        
        # Elegant color schemes
        if ok:
            core_colors = ['#00ffff', '#00ffcc', '#66ffff']
            trail_colors = ['#00aaff', '#00ffaa', '#88ffff']
            status_text = "SAFE"
        elif in_box and not smooth:
            core_colors = ['#ffaa00', '#ffcc33', '#ffdd66']
            trail_colors = ['#ff8800', '#ffaa33', '#ffcc66']
            status_text = "IN BOX (ROUGH)"
        elif corrective:
            core_colors = ['#ff7700', '#ff9900', '#ffbb44']
            trail_colors = ['#ff5500', '#ff8800', '#ffaa44']
            status_text = "SELF-FIXING"
        else:
            core_colors = ['#ff0055', '#ff3377', '#ff6699']
            trail_colors = ['#cc0033', '#ff3366', '#ff6688']
            status_text = "VIOLATION"
        
        # Smooth particle trails - toroidal pattern
        rotation = step_num * 0.12
        num_trails = 100
        
        for i in range(num_trails):
            phi = (i / num_trails) * 2 * np.pi + rotation
            r_base = a * (0.9 + 0.2 * np.sin(i * 0.3))
            
            # Create smooth trail
            n_points = 30
            trail_x, trail_y = [], []
            for j in range(n_points):
                t = j / n_points
                angle = phi + t * np.pi * 0.6
                r = r_base * (1 - t * 0.15)
                x = R0 + r * np.cos(angle)
                y = r * elongation * np.sin(angle)
                trail_x.append(x)
                trail_y.append(y)
            
            # Draw smooth trail
            color_idx = i % len(trail_colors)
            for k in range(len(trail_x) - 1):
                alpha = (1 - k / len(trail_x)) * 0.5
                ax.plot([trail_x[k], trail_x[k+1]], [trail_y[k], trail_y[k+1]], 
                       color=trail_colors[color_idx], linewidth=1.2, alpha=alpha, zorder=2)
        
        # Plasma core with smooth gradient
        for scale, color, alpha_val in [(1.1, core_colors[0], 0.2), (1.05, core_colors[0], 0.3)]:
            R_g = R0 + a * scale * (np.cos(theta) + triangularity * np.cos(2*theta))
            Z_g = a * scale * elongation * np.sin(theta)
            ax.fill(R_g, Z_g, color=color, alpha=alpha_val, zorder=3)
        
        ax.fill(R_plasma, Z_plasma, color=core_colors[0], alpha=0.65, zorder=4)
        ax.fill(R0 + a * 0.75 * (np.cos(theta) + triangularity * np.cos(2*theta)),
                a * 0.75 * elongation * np.sin(theta),
                color=core_colors[1], alpha=0.8, zorder=5)
        ax.fill(R0 + a * 0.5 * (np.cos(theta) + triangularity * np.cos(2*theta)),
                a * 0.5 * elongation * np.sin(theta),
                color=core_colors[2], alpha=0.95, zorder=6)
        
        ax.plot(R_plasma, Z_plasma, color=core_colors[0], linewidth=2.5, alpha=0.9, zorder=7)
        
        # Status
        ax.text(0.5, 0.02, f"{status_text}  |  Step {step_num+1}/{total_steps}",
               transform=ax.transAxes, fontsize=14, ha='center', fontweight='500',
               color='white', zorder=10)
        
        ax.set_aspect('equal')
        margin = 0.4
        ax.set_xlim(R0 - a0 - margin, R0 + a0 + margin)
        ax.set_ylim(-a0 * elongation - margin, a0 * elongation + margin)
        ax.axis('off')
        plt.tight_layout(pad=0)
        return fig
    
    # Show current tokamak state
    if len(df) > 0:
        current_idx = len(df) - 1
        current_row = df.iloc[current_idx]
        tokamak_fig = draw_tokamak_plasma(
            current_row["beta_N"],
            current_row["q_min"],
            current_row["q95"],
            current_row["ok"],
            current_row["in_box"],
            current_row["smooth"],
            current_row["corrective"],
            current_idx,
            len(df)
        )
        st.pyplot(tokamak_fig)
        plt.close(tokamak_fig)
    
    st.markdown("---")
    
    # ========== EXISTING GRAPHS (BELOW) ==========
    # Main visualization area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Shape Parameters Over Time")
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        constraints = shape_guard
        steps = df["step"].values
        beta_N = df["beta_N"].values
        q_min = df["q_min"].values
        q95 = df["q95"].values
        violations = ~df["ok"].values
        corrective = df["corrective"].values
        safe = df["ok"].values
        
        # β_N plot
        axes[0].plot(steps, beta_N, 'b-', alpha=0.3, linewidth=1)
        axes[0].scatter(steps[safe], beta_N[safe], c='green', s=20, alpha=0.7, label='Safe')
        axes[0].scatter(steps[violations & ~corrective], beta_N[violations & ~corrective], 
                       c='red', s=40, marker='x', label='Violation', alpha=0.8)
        axes[0].scatter(steps[corrective], beta_N[corrective], 
                       c='orange', s=80, marker='*', label='Self-Fixing!', alpha=1.0)
        axes[0].axhspan(constraints.BETA_N_MIN, constraints.BETA_N_MAX, alpha=0.15, color='green')
        axes[0].axhline(constraints.BETA_N_MIN, color='g', linestyle='--', linewidth=2)
        axes[0].axhline(constraints.BETA_N_MAX, color='g', linestyle='--', linewidth=2)
        axes[0].set_ylabel('β_N')
        axes[0].set_title('β_N Over Time')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # q_min plot
        axes[1].plot(steps, q_min, 'r-', alpha=0.3, linewidth=1)
        axes[1].scatter(steps[safe], q_min[safe], c='green', s=20, alpha=0.7, label='Safe')
        axes[1].scatter(steps[violations & ~corrective], q_min[violations & ~corrective], 
                       c='red', s=40, marker='x', label='Violation', alpha=0.8)
        axes[1].scatter(steps[corrective], q_min[corrective], 
                       c='orange', s=80, marker='*', label='Self-Fixing!', alpha=1.0)
        axes[1].axhspan(constraints.QMIN_MIN, max(q_min)*1.1, alpha=0.15, color='green')
        axes[1].axhline(constraints.QMIN_MIN, color='g', linestyle='--', linewidth=2)
        axes[1].set_ylabel('q_min')
        axes[1].set_title('q_min Over Time')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        # q95 plot
        axes[2].plot(steps, q95, 'g-', alpha=0.3, linewidth=1)
        axes[2].scatter(steps[safe], q95[safe], c='green', s=20, alpha=0.7, label='Safe')
        axes[2].scatter(steps[violations & ~corrective], q95[violations & ~corrective], 
                       c='red', s=40, marker='x', label='Violation', alpha=0.8)
        axes[2].scatter(steps[corrective], q95[corrective], 
                       c='orange', s=80, marker='*', label='Self-Fixing!', alpha=1.0)
        axes[2].axhspan(constraints.Q95_MIN, constraints.Q95_MAX, alpha=0.15, color='green')
        axes[2].axhline(constraints.Q95_MIN, color='g', linestyle='--', linewidth=2)
        axes[2].axhline(constraints.Q95_MAX, color='g', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('q95')
        axes[2].set_title('q95 Over Time')
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("📈 Statistics")
        
        # Key metrics
        total_steps = len(df)
        safe_count = df["ok"].sum()
        violation_count = (~df["ok"]).sum()
        corrective_count = df["corrective"].sum()
        
        st.metric("Total Steps", total_steps)
        st.metric("🟢 Safe Steps", safe_count, f"{safe_count/total_steps*100:.1f}%")
        st.metric("🔴 Violations", violation_count, f"{violation_count/total_steps*100:.1f}%")
        st.metric("🟠 Self-Fixing Actions", corrective_count, f"{corrective_count/total_steps*100:.1f}%")
        
        # Severity over time
        fig2, ax = plt.subplots(figsize=(8, 4))
        severity = df["severity"].values
        ax.plot(steps, severity, 'purple', linewidth=2, alpha=0.7, label='Severity')
        ax.scatter(steps[corrective], severity[corrective], 
                  c='orange', s=100, marker='*', label='Self-Fixing!', zorder=5, alpha=1.0)
        ax.fill_between(steps, 0, severity, where=(severity > 0), alpha=0.3, color='red')
        ax.set_xlabel('Step')
        ax.set_ylabel('Severity')
        ax.set_title('Severity Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    
    # 2D Trajectory
    st.subheader("🗺️ 2D Trajectory: β_N vs q95")
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    # Draw trajectory
    for i in range(len(beta_N)-1):
        if violations[i] or violations[i+1]:
            color = 'orange' if corrective[i+1] else 'red'
            alpha = 0.8 if corrective[i+1] else 0.4
            ax.plot([beta_N[i], beta_N[i+1]], [q95[i], q95[i+1]], 
                   color=color, alpha=alpha, linewidth=1.5, zorder=2)
        else:
            ax.plot([beta_N[i], beta_N[i+1]], [q95[i], q95[i+1]], 
                   color='green', alpha=0.3, linewidth=1, zorder=1)
    
    # Mark points
    ax.scatter(beta_N[safe], q95[safe], c='green', s=30, alpha=0.7, label='Safe', zorder=3)
    ax.scatter(beta_N[violations & ~corrective], q95[violations & ~corrective], 
              c='red', s=60, marker='x', label='Violation', zorder=4, alpha=0.8)
    ax.scatter(beta_N[corrective], q95[corrective], 
              c='orange', s=120, marker='*', label='Self-Fixing!', zorder=5, 
              edgecolors='darkorange', linewidths=1.5, alpha=1.0)
    
    # Safe box
    rect = Rectangle((constraints.BETA_N_MIN, constraints.Q95_MIN), 
                    constraints.BETA_N_MAX - constraints.BETA_N_MIN,
                    constraints.Q95_MAX - constraints.Q95_MIN,
                    fill=False, edgecolor='green', linewidth=2.5, linestyle='--', 
                    label='Safe Operating Box', zorder=6)
    ax.add_patch(rect)
    
    # Start/End markers
    ax.scatter(beta_N[0], q95[0], c='blue', s=200, marker='o', 
              label='Start', zorder=7, edgecolors='darkblue', linewidths=2)
    ax.scatter(beta_N[-1], q95[-1], c='purple', s=200, marker='s', 
              label='End', zorder=7, edgecolors='darkviolet', linewidths=2)
    
    ax.set_xlabel('β_N')
    ax.set_ylabel('q95')
    ax.set_title('Plasma Shape Trajectory: Moving Toward Safety')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    
    # Animated trajectory visualization
    st.subheader("🎬 Live Trajectory Animation")
    
    # Animation controls
    col_anim1, col_anim2 = st.columns(2)
    with col_anim1:
        animation_speed = st.slider("Animation Speed", 0.1, 5.0, 1.0, 0.1, key="anim_speed")
    with col_anim2:
        start_animation = st.button("▶️ Play Animation", type="primary")
    
    if start_animation and len(df) > 0:
        # Create animated plot
        placeholder = st.empty()
        tokamak_placeholder = st.empty()
        
        # Prepare data
        beta_N = df["beta_N"].values
        q95 = df["q95"].values
        q_min = df["q_min"].values
        violations = ~df["ok"].values
        corrective = df["corrective"].values
        safe = df["ok"].values
        
        # Get constraints for animation
        constraints = shape_guard
        
        # Animate step by step
        for frame in range(len(df)):
            # Update tokamak visualization
            current_row = df.iloc[frame]
            tokamak_fig = draw_tokamak_plasma(
                current_row["beta_N"],
                current_row["q_min"],
                current_row["q95"],
                current_row["ok"],
                current_row["in_box"],
                current_row["smooth"],
                current_row["corrective"],
                frame,
                len(df)
            )
            tokamak_placeholder.pyplot(tokamak_fig)
            plt.close(tokamak_fig)
            
            # Update trajectory plots
            fig_anim, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Left: 2D trajectory
            ax1.set_xlim(constraints.BETA_N_MIN - 0.5, constraints.BETA_N_MAX + 0.5)
            ax1.set_ylim(constraints.Q95_MIN - 0.5, constraints.Q95_MAX + 0.5)
            
            # Draw safe box
            rect = Rectangle((constraints.BETA_N_MIN, constraints.Q95_MIN), 
                           constraints.BETA_N_MAX - constraints.BETA_N_MIN,
                           constraints.Q95_MAX - constraints.Q95_MIN,
                           fill=True, facecolor='lightgreen', edgecolor='green', 
                           linewidth=2, linestyle='--', alpha=0.2)
            ax1.add_patch(rect)
            
            # Draw path up to current frame with color coding
            if frame > 0:
                # Draw path segments with different colors based on state
                for i in range(frame):
                    if i == 0:
                        continue
                    if corrective[i]:
                        # Self-fixing segment - orange
                        ax1.plot([beta_N[i-1], beta_N[i]], [q95[i-1], q95[i]], 
                               'orange', alpha=0.6, linewidth=2.5, zorder=2)
                    elif violations[i]:
                        # Violation segment - red
                        ax1.plot([beta_N[i-1], beta_N[i]], [q95[i-1], q95[i]], 
                               'red', alpha=0.4, linewidth=1.5, zorder=1)
                    else:
                        # Safe segment - green
                        ax1.plot([beta_N[i-1], beta_N[i]], [q95[i-1], q95[i]], 
                               'green', alpha=0.4, linewidth=1.5, zorder=1)
            
            # Mark all points up to current frame
            if frame > 0:
                frame_safe = safe[:frame]
                frame_violations = violations[:frame]
                frame_corrective = corrective[:frame]
                frame_beta = beta_N[:frame]
                frame_q95 = q95[:frame]
                
                if np.any(frame_safe):
                    ax1.scatter(frame_beta[frame_safe], frame_q95[frame_safe], 
                               c='green', s=40, alpha=0.7, label='Safe', zorder=3, edgecolors='darkgreen', linewidths=1)
                if np.any(frame_violations & ~frame_corrective):
                    ax1.scatter(frame_beta[frame_violations & ~frame_corrective], 
                               frame_q95[frame_violations & ~frame_corrective], 
                               c='red', s=60, marker='x', label='Violation', zorder=3, alpha=0.8, linewidths=2)
                if np.any(frame_corrective):
                    ax1.scatter(frame_beta[frame_corrective], frame_q95[frame_corrective], 
                               c='orange', s=100, marker='*', label='Self-Fixing!', zorder=4, 
                               alpha=1.0, edgecolors='darkorange', linewidths=2)
            
            # Current point (large and highlighted) - make it very visible
            current_ok = df.iloc[frame]["ok"]
            current_corrective = df.iloc[frame]["corrective"]
            if current_ok:
                ax1.scatter([beta_N[frame]], [q95[frame]], c='green', s=300, 
                           marker='o', edgecolors='darkgreen', linewidths=4, zorder=10, 
                           label='Current (Safe)', alpha=0.9)
            elif current_corrective:
                # Self-fixing - make it extra prominent!
                ax1.scatter([beta_N[frame]], [q95[frame]], c='orange', s=400, 
                           marker='*', edgecolors='darkorange', linewidths=4, zorder=10, 
                           label='Current (Self-Fixing!)', alpha=1.0)
                # Add a pulsing effect with a larger circle behind
                circle = plt.Circle((beta_N[frame], q95[frame]), 0.15, 
                                   color='orange', alpha=0.2, zorder=9)
                ax1.add_patch(circle)
            else:
                ax1.scatter([beta_N[frame]], [q95[frame]], c='red', s=300, 
                           marker='X', edgecolors='darkred', linewidths=4, zorder=10, 
                           label='Current (Violation)', alpha=0.9)
            
            ax1.set_xlabel('β_N', fontsize=12, fontweight='bold')
            ax1.set_ylabel('q95', fontsize=12, fontweight='bold')
            # Add status to title
            status_emoji = "🟠" if current_corrective else ("🟢" if current_ok else "🔴")
            status_text_title = "Self-Fixing!" if current_corrective else ("Safe" if current_ok else "Violation")
            ax1.set_title(f'Live Trajectory - Step {frame+1}/{len(df)} | {status_emoji} {status_text_title}', 
                         fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            if frame > 0:  # Only show legend if there are points
                ax1.legend(fontsize=8, loc='best', framealpha=0.9)
            
            # Right: 3D view
            ax2.remove()
            ax2 = fig_anim.add_subplot(122, projection='3d')
            
            # Draw safe zone box in 3D
            x_safe = [constraints.BETA_N_MIN, constraints.BETA_N_MAX, constraints.BETA_N_MAX, constraints.BETA_N_MIN, constraints.BETA_N_MIN]
            y_safe = [min(q_min), min(q_min), min(q_min), min(q_min), min(q_min)]
            z_bottom = [constraints.Q95_MIN]*5
            z_top = [constraints.Q95_MAX]*5
            
            # Draw bottom and top faces
            ax2.plot(x_safe, y_safe, z_bottom, 'g--', alpha=0.6, linewidth=2, label='Safe Zone')
            ax2.plot(x_safe, y_safe, z_top, 'g--', alpha=0.6, linewidth=2)
            # Draw vertical edges
            for i in range(4):
                ax2.plot([x_safe[i], x_safe[i]], [y_safe[i], y_safe[i]], 
                        [constraints.Q95_MIN, constraints.Q95_MAX], 'g--', alpha=0.4, linewidth=1)
            
            if frame > 0:
                ax2.plot(beta_N[:frame], q_min[:frame], q95[:frame], 
                        'b-', alpha=0.5, linewidth=2, label='Path')
            
            # Current point in 3D - make it larger and more visible
            color = 'green' if current_ok else ('orange' if current_corrective else 'red')
            marker = 'o' if current_ok else ('*' if current_corrective else 'X')
            ax2.scatter([beta_N[frame]], [q_min[frame]], [q95[frame]], 
                       c=color, s=300, marker=marker, edgecolors='black', linewidths=3)
            
            ax2.set_xlabel('β_N', fontsize=11)
            ax2.set_ylabel('q_min', fontsize=11)
            ax2.set_zlabel('q95', fontsize=11)
            ax2.set_title('3D Shape Space', fontsize=12)
            if frame > 0:
                ax2.legend(fontsize=8, loc='upper left')
            
            plt.tight_layout()
            placeholder.pyplot(fig_anim)
            plt.close(fig_anim)
            
            # Control animation speed
            time.sleep(1.0 / animation_speed)
        
        st.success("✅ Animation complete!")
    
    # Step-by-step table (collapsible)
    with st.expander("📋 Step-by-Step Details (Click to expand)"):
        display_df = df[["step", "beta_N", "q_min", "q95", "ok", "corrective", "severity", "reward", "penalty"]].copy()
        display_df["status"] = display_df.apply(
            lambda row: "🟢 Safe" if row["ok"] else ("🟠 Self-Fixing" if row["corrective"] else "🔴 Violation"),
            axis=1
        )
        display_df = display_df[["step", "status", "beta_N", "q_min", "q95", "severity", "reward", "penalty"]]
        display_df.columns = ["Step", "Status", "β_N", "q_min", "q95", "Severity", "Reward", "Penalty"]
        st.dataframe(display_df.style.format({
            "β_N": "{:.3f}",
            "q_min": "{:.3f}",
            "q95": "{:.3f}",
            "Severity": "{:.3f}",
            "Reward": "{:.3f}",
            "Penalty": "{:.3f}",
        }), width='stretch', height=400)

