"""
Animated visualization of plasma shape trajectory changing in real-time.
Shows the actual movement of the shape through parameter space.
"""
import gymnasium as gym
import gymtorax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from agent import RandomAgent
import importlib.util
from pathlib import Path
import time

# Import shape guard
spec = importlib.util.spec_from_file_location(
    "shape_guard",
    Path(__file__).parent / "optimization-for-constraints" / "shape_guard.py"
)
shape_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape_guard)
shape_violation = shape_guard.shape_violation

def animate_plasma_shape(speed=1.0, num_steps=100):
    """
    Animate the plasma shape trajectory in real-time.
    
    Args:
        speed: Animation speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
        num_steps: Number of steps to animate
    """
    # Setup environment and agent
    env = gym.make("gymtorax/IterHybrid-v0")
    agent = RandomAgent(
        action_space=env.action_space,
        shape_penalty=0.1,
        damp_on_violation=True,
        damp_factor=0.5,
    )
    
    # Collect trajectory data
    trajectory = []
    observation, info = env.reset()
    agent.reset_state(observation)
    prev_severity = float('inf')
    
    for step in range(num_steps):
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        reward = agent.apply_shape_safety(reward, observation)
        
        if agent.last_shape_info:
            shape_info = agent.last_shape_info
            # Check if this is a corrective action (violation but severity decreasing)
            is_corrective = (not shape_info["ok"] and 
                           prev_severity != float('inf') and 
                           shape_info["severity"] < prev_severity)
            
            trajectory.append({
                "step": step,
                "beta_N": shape_info["shape"][0],
                "q_min": shape_info["shape"][1],
                "q95": shape_info["shape"][2],
                "ok": shape_info["ok"],
                "severity": shape_info["severity"],
                "corrective": is_corrective,
            })
            
            prev_severity = shape_info["severity"]
        
        if terminated or truncated:
            break
    
    env.close()
    
    if not trajectory:
        print("No trajectory data collected")
        return
    
    # Extract data
    beta_N = np.array([t["beta_N"] for t in trajectory])
    q95 = np.array([t["q95"] for t in trajectory])
    q_min = np.array([t["q_min"] for t in trajectory])
    ok = np.array([t["ok"] for t in trajectory])
    severity = np.array([t["severity"] for t in trajectory])
    corrective = np.array([t["corrective"] for t in trajectory])
    
    constraints = shape_guard
    
    # Create figure with two views
    fig = plt.figure(figsize=(16, 8))
    
    # Left: 2D trajectory view (β_N vs q95)
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlim(constraints.BETA_N_MIN - 0.5, constraints.BETA_N_MAX + 0.5)
    ax1.set_ylim(constraints.Q95_MIN - 0.5, constraints.Q95_MAX + 0.5)
    ax1.set_xlabel('β_N (Normalized Beta)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('q95 (Edge Safety Factor)', fontsize=12, fontweight='bold')
    ax1.set_title('Plasma Shape Trajectory - Live Movement', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Draw safe operating box
    safe_box = Rectangle(
        (constraints.BETA_N_MIN, constraints.Q95_MIN),
        constraints.BETA_N_MAX - constraints.BETA_N_MIN,
        constraints.Q95_MAX - constraints.Q95_MIN,
        fill=True, facecolor='lightgreen', edgecolor='green', 
        linewidth=3, linestyle='--', alpha=0.3, label='Safe Zone'
    )
    ax1.add_patch(safe_box)
    
    # Initialize trajectory line and point
    line, = ax1.plot([], [], 'b-', alpha=0.4, linewidth=2, label='Path')
    # Make current point much larger and more visible
    current_point, = ax1.plot([], [], 'o', markersize=25, markeredgewidth=3, 
                              zorder=10, label='Current Position', animated=True)
    safe_point, = ax1.plot([], [], 'go', markersize=12, alpha=0.7, zorder=4)
    violation_point, = ax1.plot([], [], 'rx', markersize=15, zorder=4)
    corrective_point, = ax1.plot([], [], '*', markersize=20, color='orange', 
                                 markeredgecolor='darkorange', markeredgewidth=2, 
                                 zorder=9, label='Self-Fixing!', alpha=0.9)
    # Add a trail to show recent positions
    trail, = ax1.plot([], [], 'o', markersize=8, alpha=0.5, color='cyan', zorder=3, label='Recent Trail')
    
    # Add text for current state
    status_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                          fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.legend(loc='upper right', fontsize=9)
    
    # Right: 3D-like view showing all parameters
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    ax2.set_xlabel('β_N', fontsize=11, fontweight='bold')
    ax2.set_ylabel('q_min', fontsize=11, fontweight='bold')
    ax2.set_zlabel('q95', fontsize=11, fontweight='bold')
    ax2.set_title('3D Shape Space Trajectory', fontsize=12, fontweight='bold')
    
    # Set 3D limits
    ax2.set_xlim(constraints.BETA_N_MIN - 0.5, constraints.BETA_N_MAX + 0.5)
    ax2.set_ylim(min(q_min) - 0.5, max(q_min) + 0.5)
    ax2.set_zlim(constraints.Q95_MIN - 0.5, constraints.Q95_MAX + 0.5)
    
    # Draw safe zone box in 3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Create a wireframe box for the safe zone
    x_safe = [constraints.BETA_N_MIN, constraints.BETA_N_MAX, constraints.BETA_N_MAX, constraints.BETA_N_MIN, constraints.BETA_N_MIN]
    y_safe = [min(q_min), min(q_min), min(q_min), min(q_min), min(q_min)]  # Use q_min range
    z_safe = [constraints.Q95_MIN, constraints.Q95_MIN, constraints.Q95_MAX, constraints.Q95_MAX, constraints.Q95_MIN]
    
    # Draw bottom face
    ax2.plot(x_safe, y_safe, [constraints.Q95_MIN]*5, 'g--', alpha=0.5, linewidth=2, label='Safe Zone')
    # Draw top face
    ax2.plot(x_safe, y_safe, [constraints.Q95_MAX]*5, 'g--', alpha=0.5, linewidth=2)
    # Draw vertical edges
    for i in range(4):
        ax2.plot([x_safe[i], x_safe[i]], [y_safe[i], y_safe[i]], 
                [constraints.Q95_MIN, constraints.Q95_MAX], 'g--', alpha=0.5, linewidth=1)
    
    # Initialize 3D trajectory
    line3d, = ax2.plot([], [], [], 'b-', alpha=0.5, linewidth=2, label='Path')
    current_point3d, = ax2.plot([], [], [], 'o', markersize=20, markeredgewidth=2, zorder=5)
    
    # Animation function
    def animate(frame):
        if frame >= len(trajectory):
            return line, current_point, safe_point, violation_point, corrective_point, trail, status_text, line3d, current_point3d
        
        # Update 2D plot
        x_data = beta_N[:frame+1]
        y_data = q95[:frame+1]
        line.set_data(x_data, y_data)
        
        # Current point
        current_beta = beta_N[frame]
        current_q95 = q95[frame]
        current_ok = ok[frame]
        current_corrective = corrective[frame]
        
        # Make current point very visible
        current_point.set_data([current_beta], [current_q95])
        
        # Color based on safety and corrective actions - make it very obvious
        if current_corrective:
            # Self-fixing action - orange star!
            current_point.set_color('orange')
            current_point.set_marker('*')
            current_point.set_markeredgecolor('darkorange')
            current_point.set_markersize(30)  # Even larger for self-fixing
            corrective_point.set_data([current_beta], [current_q95])
            violation_point.set_data([], [])
            safe_point.set_data([], [])
        elif current_ok:
            # Safe state - green circle
            current_point.set_color('green')
            current_point.set_marker('o')
            current_point.set_markeredgecolor('darkgreen')
            current_point.set_markersize(25)
            safe_point.set_data([current_beta], [current_q95])
            violation_point.set_data([], [])
            corrective_point.set_data([], [])
        else:
            # Violation - red X
            current_point.set_color('red')
            current_point.set_marker('X')
            current_point.set_markeredgecolor('darkred')
            current_point.set_markersize(25)
            violation_point.set_data([current_beta], [current_q95])
            safe_point.set_data([], [])
            corrective_point.set_data([], [])
        
        # Show recent trail (last 5 points) to make movement more visible
        trail_length = min(5, frame + 1)
        if trail_length > 0:
            trail_x = beta_N[max(0, frame - trail_length + 1):frame+1]
            trail_y = q95[max(0, frame - trail_length + 1):frame+1]
            trail.set_data(trail_x, trail_y)
        else:
            trail.set_data([], [])
        
        # Update status text
        t = trajectory[frame]
        if t["corrective"]:
            status = "🟠 SELF-FIXING!"
        elif t["ok"]:
            status = "🟢 SAFE"
        else:
            status = "🔴 VIOLATION"
        status_text.set_text(
            f'Step: {frame+1}/{len(trajectory)}\n'
            f'Status: {status}\n'
            f'β_N: {t["beta_N"]:.3f}\n'
            f'q_min: {t["q_min"]:.3f}\n'
            f'q95: {t["q95"]:.3f}\n'
            f'Severity: {t["severity"]:.3f}'
        )
        
        # Update 3D plot
        x3d = beta_N[:frame+1]
        y3d = q_min[:frame+1]
        z3d = q95[:frame+1]
        line3d.set_data_3d(x3d, y3d, z3d)
        current_point3d.set_data_3d([current_beta], [q_min[frame]], [current_q95])
        
        # Color 3D point based on safety and corrective actions
        if current_corrective:
            current_point3d.set_color('orange')
            current_point3d.set_markeredgecolor('darkorange')
            current_point3d.set_marker('*')
        elif current_ok:
            current_point3d.set_color('green')
            current_point3d.set_markeredgecolor('darkgreen')
            current_point3d.set_marker('o')
        else:
            current_point3d.set_color('red')
            current_point3d.set_markeredgecolor('darkred')
            current_point3d.set_marker('X')
        
        return line, current_point, safe_point, violation_point, corrective_point, trail, status_text, line3d, current_point3d
    
    # Create animation
    interval = int(1000 / speed)  # milliseconds per frame (lower = faster)
    anim = FuncAnimation(fig, animate, frames=len(trajectory), 
                        interval=interval, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

if __name__ == "__main__":
    import sys
    
    # Get speed from command line or use default
    speed = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"🎬 Starting animation (speed: {speed}x, steps: {num_steps})")
    print("💡 Close the window to stop the animation")
    
    anim = animate_plasma_shape(speed=speed, num_steps=num_steps)
    
    # Keep animation running
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nAnimation stopped")

