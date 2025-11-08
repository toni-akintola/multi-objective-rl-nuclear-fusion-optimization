"""
Futuristic tokamak chamber visualization with swirling particle trails.
Inspired by particle accelerator/fusion reactor aesthetics.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import time

class TokamakChamber:
    def __init__(self, beta_N=1.5, q_min=1.5, q95=4.0, ok=True, violation=0.0):
        self.beta_N = beta_N
        self.q_min = q_min
        self.q95 = q95
        self.ok = ok
        self.violation = violation
        self.time = 0
        
        # Chamber dimensions
        self.chamber_radius = 10
        self.chamber_height = 12
        self.center_column_radius = 1.5
        
        # Particle trail parameters
        self.num_trails = 200
        self.trail_length = 30
        self.trails = []
        self._init_trails()
    
    def _init_trails(self):
        """Initialize particle trails with random starting positions."""
        self.trails = []
        for i in range(self.num_trails):
            # Random starting angle and radius
            angle = np.random.random() * 2 * np.pi
            radius = 3 + np.random.random() * 4  # Between 3 and 7
            z = -2 + np.random.random() * 4  # Vertical position
            
            # Color based on status
            if self.ok:
                color = np.random.choice(['cyan', 'blue', 'green'], p=[0.4, 0.4, 0.2])
            elif self.violation < 0.5:
                color = np.random.choice(['orange', 'yellow'], p=[0.7, 0.3])
            else:
                color = np.random.choice(['red', 'magenta', 'pink'], p=[0.5, 0.3, 0.2])
            
            self.trails.append({
                'angle': angle,
                'radius': radius,
                'z': z,
                'color': color,
                'speed': 0.05 + np.random.random() * 0.1,
                'phase': np.random.random() * 2 * np.pi,
            })
    
    def update(self, dt=0.1):
        """Update particle positions."""
        self.time += dt
        
        # Update plasma size based on beta_N
        size_factor = 0.7 + (self.beta_N - 0.5) / (3.0 - 0.5) * 0.3
        
        for trail in self.trails:
            # Toroidal motion - particles swirl around
            trail['angle'] += trail['speed'] * size_factor
            trail['phase'] += 0.02
            
            # Add some vertical oscillation
            trail['z'] += 0.01 * np.sin(trail['phase'])
            if trail['z'] > 2:
                trail['z'] = -2
            elif trail['z'] < -2:
                trail['z'] = 2
    
    def draw(self, ax):
        """Draw the chamber and particle trails."""
        ax.clear()
        ax.set_xlim(-12, 12)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('#0a0a0f')
        
        # Draw chamber walls (circular)
        chamber_circle = Circle((0, 0), self.chamber_radius, 
                               fill=False, edgecolor='#3a3a4a', 
                               linewidth=3, alpha=0.6)
        ax.add_patch(chamber_circle)
        
        # Draw inner chamber ring
        inner_circle = Circle((0, 0), self.chamber_radius * 0.9, 
                             fill=False, edgecolor='#2a2a3a', 
                             linewidth=1, alpha=0.4, linestyle='--')
        ax.add_patch(inner_circle)
        
        # Draw center column (top view - appears as circle)
        col_radius = self.center_column_radius * (1 + 0.2 * np.sin(self.time))
        center_col = Circle((0, 0), col_radius, 
                           fill=True, color='#1a1a2a', 
                           edgecolor='#4a4a5a', linewidth=2, alpha=0.8)
        ax.add_patch(center_col)
        
        # Draw particle trails
        size_factor = 0.7 + (self.beta_N - 0.5) / (3.0 - 0.5) * 0.3
        
        for trail in self.trails:
            angle = trail['angle']
            radius = trail['radius'] * size_factor
            z = trail['z']
            
            # Convert 3D position to 2D (top view)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Draw trail as a streak
            trail_points = 5
            for i in range(trail_points):
                t = i / trail_points
                prev_angle = angle - trail['speed'] * (1 - t) * 10
                prev_radius = radius * (1 - t * 0.1)
                px = prev_radius * np.cos(prev_angle)
                py = prev_radius * np.sin(prev_angle)
                
                alpha = (1 - t) * 0.8
                size = 20 * (1 - t) + 5
                
                ax.scatter(px, py, s=size, c=trail['color'], 
                         alpha=alpha, edgecolors='none', zorder=10)
            
            # Main particle
            ax.scatter(x, y, s=30, c=trail['color'], 
                      alpha=0.9, edgecolors='white', 
                      linewidths=0.5, zorder=11)
        
        # Status text
        status = "🟢 SAFE" if self.ok else ("🟠 SELF-FIXING" if self.violation < 0.5 else "🔴 VIOLATION")
        ax.text(0, -10, f"{status} | β_N={self.beta_N:.2f} | q95={self.q95:.2f}", 
               ha='center', va='top', fontsize=12, color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))


def animate_chamber(beta_N=1.5, q_min=1.5, q95=4.0, ok=True, violation=0.0, duration=10):
    """Animate the tokamak chamber."""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0a0a0f')
    
    chamber = TokamakChamber(beta_N, q_min, q95, ok, violation)
    
    def animate(frame):
        chamber.update(dt=0.1)
        chamber.draw(ax)
        return []
    
    anim = FuncAnimation(fig, animate, interval=50, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()
    return anim


if __name__ == "__main__":
    # Test with different states
    print("Creating tokamak chamber visualization...")
    print("Close the window to exit.")
    
    # Start with safe state
    anim = animate_chamber(beta_N=1.8, q_min=1.5, q95=4.0, ok=True, violation=0.0)
    
    # Keep running
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\nStopped.")

