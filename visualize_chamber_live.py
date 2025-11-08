"""
Live tokamak chamber visualization connected to agent.
Shows real-time particle trails based on simulation state.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import gymnasium as gym
import gymtorax
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


class LiveTokamakChamber:
    def __init__(self, ax):
        self.ax = ax
        self.chamber_radius = 10
        self.num_trails = 300
        self.trails = []
        self.time = 0
        self._init_trails()
    
    def _init_trails(self):
        """Initialize particle trails."""
        self.trails = []
        for i in range(self.num_trails):
            angle = np.random.random() * 2 * np.pi
            radius = 3 + np.random.random() * 4
            z = -2 + np.random.random() * 4
            
            self.trails.append({
                'angle': angle,
                'radius': radius,
                'z': z,
                'speed': 0.05 + np.random.random() * 0.1,
                'phase': np.random.random() * 2 * np.pi,
            })
    
    def update_trails(self, beta_N, ok, violation):
        """Update trail colors and behavior based on state."""
        size_factor = 0.7 + (beta_N - 0.5) / (3.0 - 0.5) * 0.3
        
        for trail in self.trails:
            trail['angle'] += trail['speed'] * size_factor
            trail['phase'] += 0.02
            trail['z'] += 0.01 * np.sin(trail['phase'])
            if trail['z'] > 2:
                trail['z'] = -2
            elif trail['z'] < -2:
                trail['z'] = 2
            
            # Update color based on status
            if ok:
                trail['color'] = np.random.choice(['cyan', 'blue', 'green'], p=[0.4, 0.4, 0.2])
            elif violation < 0.5:
                trail['color'] = np.random.choice(['orange', 'yellow'], p=[0.7, 0.3])
            else:
                trail['color'] = np.random.choice(['red', 'magenta', 'pink'], p=[0.5, 0.3, 0.2])
    
    def draw(self, beta_N, q_min, q95, ok, violation):
        """Draw the chamber."""
        self.ax.clear()
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-8, 8)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_facecolor('#0a0a0f')
        
        # Chamber walls
        chamber = Circle((0, 0), self.chamber_radius, 
                        fill=False, edgecolor='#3a3a4a', 
                        linewidth=3, alpha=0.6)
        self.ax.add_patch(chamber)
        
        inner = Circle((0, 0), self.chamber_radius * 0.9, 
                      fill=False, edgecolor='#2a2a3a', 
                      linewidth=1, alpha=0.4, linestyle='--')
        self.ax.add_patch(inner)
        
        # Center column
        col_radius = 1.5 * (1 + 0.2 * np.sin(self.time))
        center_col = Circle((0, 0), col_radius, 
                           fill=True, color='#1a1a2a', 
                           edgecolor='#4a4a5a', linewidth=2, alpha=0.8)
        self.ax.add_patch(center_col)
        
        # Update and draw trails
        self.update_trails(beta_N, ok, violation)
        size_factor = 0.7 + (beta_N - 0.5) / (3.0 - 0.5) * 0.3
        
        for trail in self.trails:
            angle = trail['angle']
            radius = trail['radius'] * size_factor
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Draw trail streak
            for i in range(5):
                t = i / 5
                prev_angle = angle - trail['speed'] * (1 - t) * 10
                prev_radius = radius * (1 - t * 0.1)
                px = prev_radius * np.cos(prev_angle)
                py = prev_radius * np.sin(prev_angle)
                
                alpha = (1 - t) * 0.8
                size = 20 * (1 - t) + 5
                
                self.ax.scatter(px, py, s=size, c=trail['color'], 
                              alpha=alpha, edgecolors='none', zorder=10)
            
            # Main particle
            self.ax.scatter(x, y, s=30, c=trail['color'], 
                          alpha=0.9, edgecolors='white', 
                          linewidths=0.5, zorder=11)
        
        self.time += 0.1
        
        # Status
        status = "🟢 SAFE" if ok else ("🟠 SELF-FIXING" if violation < 0.5 else "🔴 VIOLATION")
        self.ax.text(0, -10, f"{status} | β_N={beta_N:.2f} | q_min={q_min:.2f} | q95={q95:.2f}", 
                    ha='center', va='top', fontsize=12, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))


def run_live_visualization():
    """Run live visualization connected to agent."""
    # Setup environment
    env = gym.make("gymtorax/IterHybrid-v0")
    agent = RandomAgent(
        action_space=env.action_space,
        shape_penalty=0.1,
        damp_on_violation=True,
        damp_factor=0.5,
    )
    
    observation, info = env.reset()
    agent.reset_state(observation)
    
    # Setup visualization
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0a0a0f')
    chamber = LiveTokamakChamber(ax)
    
    step_count = 0
    
    def animate(frame):
        nonlocal observation, step_count
        
        # Get action and step
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        reward = agent.apply_shape_safety(reward, observation)
        
        # Get shape data
        if agent.last_shape_info:
            shape_info = agent.last_shape_info
            beta_N = float(shape_info["shape"][0])
            q_min = float(shape_info["shape"][1])
            q95 = float(shape_info["shape"][2])
            violation = float(shape_info["severity"])
            ok = shape_info["ok"]
            
            chamber.draw(beta_N, q_min, q95, ok, violation)
            
            if step_count % 10 == 0:
                status = "🟢 SAFE" if ok else ("🟠 SELF-FIXING" if violation < 0.5 else "🔴 VIOLATION")
                print(f"Step {step_count}: {status} | β_N={beta_N:.2f} | violation={violation:.3f}")
        
        step_count += 1
        
        # Reset if episode ends
        if terminated or truncated:
            observation, info = env.reset()
            agent.reset_state(observation)
            step_count = 0
        
        return []
    
    anim = FuncAnimation(fig, animate, interval=100, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()
    
    env.close()


if __name__ == "__main__":
    print("Starting live tokamak chamber visualization...")
    print("Close the window to stop.")
    try:
        run_live_visualization()
    except KeyboardInterrupt:
        print("\nStopped.")

