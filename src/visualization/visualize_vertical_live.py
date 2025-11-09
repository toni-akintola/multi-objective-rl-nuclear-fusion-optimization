"""
Live vertical guard visualization with interactive coil controls.
Shows real-time vertical position and self-correction as user adjusts coil parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
import gymnasium as gym
import gymtorax
from agent import RandomAgent
from src.environments.iter_hybrid_shape_guard_env import make_iter_hybrid_shape_guard_env
from src.utils.vertical_guard import Z_MAX, MAX_DZ
from collections import deque


class LiveVerticalChamber:
    def __init__(self, ax_main, ax_history, ax_status):
        self.ax_main = ax_main  # Main tokamak side view
        self.ax_history = ax_history  # Position history plot
        self.ax_status = ax_status  # Status panel
        
        # Tokamak dimensions
        self.chamber_radius = 8
        self.center_column_radius = 1.5
        
        # Coil parameters (will be controlled by sliders)
        self.coil_upper_current = 1.0
        self.coil_lower_current = 1.0
        self.coil_center_current = 1.0
        
        # Vertical position tracking
        self.z_history = deque(maxlen=200)
        self.time_history = deque(maxlen=200)
        self.violation_history = deque(maxlen=200)
        self.status_history = deque(maxlen=200)
        
        # Plasma particles - fewer for cleaner look
        self.num_particles = 50
        self.particles = []
        self._init_particles()
        
        self.time = 0
        self.prev_z = 0.0
        self.prev_severity = float('inf')
        
    def _init_particles(self):
        """Initialize plasma particles."""
        self.particles = []
        for i in range(self.num_particles):
            angle = np.random.random() * 2 * np.pi
            radius = 2 + np.random.random() * 3
            z = np.random.random() * 2 - 1  # Start near center
            
            self.particles.append({
                'angle': angle,
                'radius': radius,
                'z': z,
                'speed': 0.03 + np.random.random() * 0.05,
                'phase': np.random.random() * 2 * np.pi,
            })
    
    def update_coil_currents(self, upper, lower, center):
        """Update coil current values (called by sliders)."""
        self.coil_upper_current = upper
        self.coil_lower_current = lower
        self.coil_center_current = center
    
    def update_particles(self, z_cm, status):
        """Update particle positions based on vertical position."""
        z_viz = z_cm / 10.0  # Scale for visualization
        
        for particle in self.particles:
            particle['angle'] += particle['speed']
            particle['phase'] += 0.02
            
            # Vertical drift based on z_cm
            target_z = z_viz
            particle['z'] += (target_z - particle['z']) * 0.1
            
            # Add random motion
            particle['z'] += 0.05 * np.sin(particle['phase'])
            
            # Update color based on status
            if status == 'SAFE':
                particle['color'] = np.random.choice(['cyan', 'blue', 'green'], p=[0.4, 0.4, 0.2])
            elif status == 'SELF-FIXING':
                particle['color'] = np.random.choice(['orange', 'yellow', 'lime'], p=[0.5, 0.3, 0.2])
            else:  # VIOLATION
                particle['color'] = np.random.choice(['red', 'magenta', 'pink'], p=[0.5, 0.3, 0.2])
    
    def draw_main_view(self, z_cm, status, severity, dz, in_band, smooth):
        """Draw tokamak from side view - beautiful, elegant design."""
        self.ax_main.clear()
        self.ax_main.set_xlim(-11, 11)
        self.ax_main.set_ylim(-11, 7)
        self.ax_main.set_aspect('equal')
        self.ax_main.axis('off')
        self.ax_main.set_facecolor('#0f0f0f')
        
        # Simple chamber - just one ring
        chamber_ring = Circle((0, 0), self.chamber_radius, fill=False,
                             edgecolor='#444455', linewidth=1.5, alpha=0.4, linestyle='-')
        self.ax_main.add_patch(chamber_ring)
        
        # Center column - simple
        center_col = Circle((0, 0), self.center_column_radius, fill=True,
                           facecolor='#1a1a2a', edgecolor='#444455', linewidth=1.5)
        self.ax_main.add_patch(center_col)
        
        # Coils - simple circles
        coil_positions = {
            'upper': {'x': 0, 'y': 6, 'radius': 0.8, 'current': self.coil_upper_current},
            'lower': {'x': 0, 'y': -6, 'radius': 0.8, 'current': self.coil_lower_current},
            'center': {'x': 0, 'y': 0, 'radius': 1.4, 'current': self.coil_center_current},
        }
        
        coil_colors = {
            'upper': '#4a88ff',
            'lower': '#4aff88',
            'center': '#ff8844'
        }
        
        for coil_name, coil in coil_positions.items():
            base_color = coil_colors[coil_name]
            
            # Simple coil ring
            coil_ring = Circle((coil['x'], coil['y']), coil['radius'], 
                             fill=False, edgecolor=base_color, 
                             linewidth=2, alpha=0.6)
            self.ax_main.add_patch(coil_ring)
        
        # Safe zone - very subtle
        safe_band = Rectangle((-self.chamber_radius, -Z_MAX/10), 
                             2*self.chamber_radius, 2*Z_MAX/10,
                             fill=True, facecolor='#00ff88', alpha=0.05,
                             edgecolor='#00ff88', linewidth=0.5, linestyle='-')
        self.ax_main.add_patch(safe_band)
        
        # Vertical position visualization
        z_viz = z_cm / 10.0
        
        # Status colors - elegant palette
        if status == 'SAFE':
            plasma_color = '#00ffaa'
            glow_color = '#00ff66'
        elif status == 'SELF-FIXING':
            plasma_color = '#ffaa00'
            glow_color = '#ff8800'
        else:
            plasma_color = '#ff4444'
            glow_color = '#ff2222'
        
        # Plasma particles - minimal, no trails
        for particle in self.particles:
            x = particle['radius'] * np.cos(particle['angle'])
            y = particle['z'] + z_viz
            
            # Color based on status
            if status == 'SAFE':
                p_color = '#00ffff'
            elif status == 'SELF-FIXING':
                p_color = '#ffaa00'
            else:
                p_color = '#ff6666'
            
            # Just the particle, no trails
            self.ax_main.scatter(x, y, s=20, c=p_color, alpha=0.6, 
                               edgecolors='none')
        
        # Plasma boundary - simple ring
        plasma_ring = Circle((0, z_viz), 3.0, fill=False,
                            edgecolor=plasma_color, linewidth=2, alpha=0.7)
        self.ax_main.add_patch(plasma_ring)
        
        # Simple vertical line
        if abs(z_viz) > 0.05:
            self.ax_main.plot([0, 0], [0, z_viz], color=plasma_color, 
                            linewidth=2.5, alpha=0.6)
        
        # Position marker - simple
        marker = Circle((0, z_viz), 0.3, fill=True, 
                      facecolor=plasma_color, edgecolor='white', linewidth=1.5)
        self.ax_main.add_patch(marker)
    
    def draw_history(self):
        """Draw vertical position over time - clean website style."""
        self.ax_history.clear()
        self.ax_history.set_facecolor('#0f0f0f')
        
        if len(self.time_history) < 2:
            return
        
        times = np.array(self.time_history)
        z_values = np.array(self.z_history)
        violations = np.array(self.violation_history)
        statuses = list(self.status_history)
        
        # Draw safe band with gradient
        self.ax_history.axhspan(-Z_MAX, Z_MAX, alpha=0.15, color='#00ff88', label='Safe Zone')
        self.ax_history.axhline(Z_MAX, color='#00ff88', linestyle='-', linewidth=1.5, alpha=0.6)
        self.ax_history.axhline(-Z_MAX, color='#00ff88', linestyle='-', linewidth=1.5, alpha=0.6)
        
        # Simple line plot
        self.ax_history.plot(times, z_values, color='#00ffff', 
                           linewidth=2, alpha=0.8)
        
        self.ax_history.set_xlabel('Time', color='#888888', fontsize=9)
        self.ax_history.set_ylabel('Position (cm)', color='#888888', fontsize=9)
        self.ax_history.grid(True, alpha=0.1, color='#333333', linestyle='-')
        self.ax_history.tick_params(colors='#666666', labelsize=8)
        self.ax_history.set_facecolor('#0f0f0f')
    
    def draw_status_panel(self, z_cm, dz, severity, status, in_band, smooth):
        """Draw status panel - clean website style."""
        self.ax_status.clear()
        self.ax_status.set_xlim(0, 10)
        self.ax_status.set_ylim(0, 10)
        self.ax_status.axis('off')
        self.ax_status.set_facecolor('#0f0f0f')
        
        y_pos = 9
        
        # Status - minimal
        status_colors = {
            'SAFE': '#00ffaa',
            'SELF-FIXING': '#ffaa00',
            'VIOLATION': '#ff4444'
        }
        status_color = status_colors.get(status, '#ffffff')
        self.ax_status.text(5, y_pos, status, ha='center', va='top',
                           fontsize=12, weight='bold', color=status_color)
        y_pos -= 1.5
        
        # Position
        self.ax_status.text(5, y_pos, f"{z_cm:.2f} cm", ha='center', va='top',
                           fontsize=11, color='#00ffff')
        y_pos -= 1.2
        
        # Safety checks - minimal
        check_color = '#00ffaa' if in_band else '#ff4444'
        self.ax_status.text(1, y_pos, "In Band" if in_band else "Out", 
                           ha='left', va='center', fontsize=9, color=check_color)
        y_pos -= 0.8
        
        smooth_color = '#00ffaa' if smooth else '#ff4444'
        self.ax_status.text(1, y_pos, "Smooth" if smooth else "Rapid", 
                           ha='left', va='center', fontsize=9, color=smooth_color)
        y_pos -= 0.8
        
        # Severity
        self.ax_status.text(1, y_pos, f"{severity:.2f}", 
                           ha='left', va='center', fontsize=9, 
                           color='#ffaa00' if severity > 0 else '#00ffaa')
    
    def update(self, z_cm, dz, severity, status, in_band, smooth):
        """Update all visualizations."""
        self.z_history.append(z_cm)
        self.time_history.append(self.time)
        self.violation_history.append(severity)
        self.status_history.append(status)
        self.time += 1
        
        self.update_particles(z_cm, status)
        self.draw_main_view(z_cm, status, severity, dz, in_band, smooth)
        self.draw_history()
        self.draw_status_panel(z_cm, dz, severity, status, in_band, smooth)
        
        self.prev_z = z_cm
        self.prev_severity = severity


def run_live_vertical_visualization():
    """Run live vertical visualization with interactive coil controls."""
    print("Initializing live vertical guard visualization...")
    
    # Create environment with vertical guard
    env = make_iter_hybrid_shape_guard_env(
        shape_penalty=0.05,
        vertical_penalty=0.05,
        enable_shape_guard=True,
        enable_vertical_guard=True
    )
    
    agent = RandomAgent(env.action_space, shape_penalty=0.0)
    observation, info = env.reset()
    
    # Setup figure with subplots - clean website aesthetic
    fig = plt.figure(figsize=(14, 8), facecolor='#0f0f0f')
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25, 
                         left=0.08, right=0.95, top=0.95, bottom=0.20)
    
    ax_main = fig.add_subplot(gs[:, 0], facecolor='#0f0f0f')  # Main view (left half)
    ax_history = fig.add_subplot(gs[0, 1:], facecolor='#0f0f0f')  # History (top right)
    ax_status = fig.add_subplot(gs[1, 1:], facecolor='#0f0f0f')  # Status (bottom right)
    
    chamber = LiveVerticalChamber(ax_main, ax_history, ax_status)
    
    # Create sliders for coil control - clean website style
    ax_upper = plt.axes([0.1, 0.06, 0.25, 0.03], facecolor='#1a1a1a')
    ax_lower = plt.axes([0.4, 0.06, 0.25, 0.03], facecolor='#1a1a1a')
    ax_center = plt.axes([0.7, 0.06, 0.25, 0.03], facecolor='#1a1a1a')
    
    # Make sliders - clean website style
    slider_upper = Slider(ax_upper, 'Upper', -2.0, 2.0, valinit=1.0, 
                         valstep=0.1, 
                         facecolor='#4a88ff', 
                         track_color='#2a4a88',
                         handle_style={'size': 10})
    slider_lower = Slider(ax_lower, 'Lower', -2.0, 2.0, valinit=1.0, 
                         valstep=0.1, 
                         facecolor='#4aff88',
                         track_color='#2a8844',
                         handle_style={'size': 10})
    slider_center = Slider(ax_center, 'Center', -2.0, 2.0, valinit=1.0, 
                          valstep=0.1, 
                          facecolor='#ff8844',
                          track_color='#884422',
                          handle_style={'size': 10})
    
    # Clean slider labels
    for slider in [slider_upper, slider_lower, slider_center]:
        slider.label.set_color('#aaaaaa')
        slider.label.set_fontsize(10)
        slider.valtext.set_color('#ffffff')
        slider.valtext.set_fontsize(9)
    
    def update_coils(val):
        """Update coil currents when sliders change."""
        chamber.update_coil_currents(
            slider_upper.val,
            slider_lower.val,
            slider_center.val
        )
    
    slider_upper.on_changed(update_coils)
    slider_lower.on_changed(update_coils)
    slider_center.on_changed(update_coils)
    
    step_count = 0
    prev_vertical_info = None
    
    # Store coil influence on vertical position
    coil_vertical_influence = 0.0
    
    def modify_action_with_coils(action):
        """Modify action to reflect coil current changes."""
        # Calculate vertical influence from coil current imbalance
        # Upper coil pushes plasma up (positive z), lower coil pushes down (negative z)
        # Center coil affects overall stability
        upper_influence = (chamber.coil_upper_current - 1.0) * 2.0  # Scale factor
        lower_influence = (chamber.coil_lower_current - 1.0) * -2.0  # Negative (pushes down)
        center_influence = (chamber.coil_center_current - 1.0) * 0.5
        
        # Combined influence on vertical position (in cm)
        nonlocal coil_vertical_influence
        coil_vertical_influence = upper_influence + lower_influence + center_influence
        
        # Modify action if it's a dict
        if isinstance(action, dict):
            modified = action.copy()
            # Try to find and modify coil-related parameters
            # Common keys might be: 'coil_currents', 'I_coil', 'coil', etc.
            for key in ['coil_currents', 'I_coil', 'coil', 'coils', 'magnetic_coils']:
                if key in modified:
                    if isinstance(modified[key], (list, np.ndarray)):
                        # Modify coil currents if available
                        if len(modified[key]) >= 3:
                            modified[key][0] = chamber.coil_upper_current
                            modified[key][1] = chamber.coil_lower_current
                            modified[key][2] = chamber.coil_center_current
                    elif isinstance(modified[key], dict):
                        # Nested dict structure
                        if 'upper' in modified[key]:
                            modified[key]['upper'] = chamber.coil_upper_current
                        if 'lower' in modified[key]:
                            modified[key]['lower'] = chamber.coil_lower_current
                        if 'center' in modified[key]:
                            modified[key]['center'] = chamber.coil_center_current
            return modified
        elif isinstance(action, np.ndarray):
            # If action is an array, we can't directly modify coil currents
            # but we'll apply the influence through the vertical position
            return action
        return action
    
    def animate(frame):
        nonlocal observation, step_count, prev_vertical_info
        
        # Get action
        action = agent.act(observation)
        
        # Modify action based on coil settings
        # Note: This is a simplified approach - actual coil control would
        # require understanding the action space structure
        action = modify_action_with_coils(action)
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Get vertical info
        if hasattr(env, 'vertical_info') and env.vertical_info:
            vertical_info = env.vertical_info.copy()  # Make a copy to modify
            
            # Apply coil current influence directly to vertical position
            # This simulates how coil currents affect the magnetic field and plasma position
            base_z_cm = vertical_info['z']
            
            # Add coil influence (scaled to be visible but realistic)
            # The influence decays over time to show self-correction
            coil_vertical_influence = (chamber.coil_upper_current - 1.0) * 3.0 + \
                                    (chamber.coil_lower_current - 1.0) * -3.0 + \
                                    (chamber.coil_center_current - 1.0) * 1.0
            
            # Apply influence with some smoothing
            if not hasattr(chamber, 'target_z_influence'):
                chamber.target_z_influence = 0.0
            chamber.target_z_influence += (coil_vertical_influence - chamber.target_z_influence) * 0.1
            
            # Modify the vertical position to show coil effect
            modified_z_cm = base_z_cm + chamber.target_z_influence
            
            # Update vertical_info with modified position
            prev_z = chamber.prev_z if hasattr(chamber, 'prev_z') else base_z_cm
            vertical_info['z'] = modified_z_cm
            vertical_info['dz'] = modified_z_cm - prev_z
            
            # Recalculate safety checks with modified position
            vertical_info['in_band'] = abs(modified_z_cm) <= Z_MAX
            vertical_info['smooth'] = abs(vertical_info['dz']) <= MAX_DZ
            vertical_info['ok'] = vertical_info['in_band'] and vertical_info['smooth']
            
            # Recalculate severity
            z_severity = abs(modified_z_cm) / Z_MAX if not vertical_info['in_band'] else 0.0
            dz_severity = abs(vertical_info['dz']) / MAX_DZ if not vertical_info['smooth'] else 0.0
            vertical_info['severity'] = z_severity + dz_severity
            
            z_cm = modified_z_cm
            dz = vertical_info['dz']
            severity = vertical_info['severity']
            in_band = vertical_info['in_band']
            smooth = vertical_info['smooth']
            ok = vertical_info['ok']
            
            # Determine status
            if ok:
                status = 'SAFE'
            elif prev_vertical_info and not prev_vertical_info['ok'] and severity < prev_vertical_info['severity']:
                status = 'SELF-FIXING'
            else:
                status = 'VIOLATION'
            
            # Update visualization
            chamber.update(z_cm, dz, severity, status, in_band, smooth)
            
            # Store previous z for next iteration
            chamber.prev_z = modified_z_cm
            prev_vertical_info = vertical_info
        
        step_count += 1
        
        # Reset if episode ends
        if terminated or truncated:
            observation, info = env.reset()
            agent.reset_state(observation)
            prev_vertical_info = None
    
    # Create animation
    anim = FuncAnimation(fig, animate, interval=50, blit=False, repeat=True)
    
    # Clean website aesthetic - no title, no instruction text
    # Ensure all axes have clean background
    for ax in [ax_main, ax_history, ax_status]:
        ax.set_facecolor('#0f0f0f')
        # Hide all spines
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    plt.show()
    
    env.close()


if __name__ == "__main__":
    run_live_vertical_visualization()

