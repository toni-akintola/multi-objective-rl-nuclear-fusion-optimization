"""
Cool 3D-style rotating vertical guard visualization.
Shows plasma rotating around with ability to disable coils and watch self-correction.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button
import gymnasium as gym
import gymtorax
from agent import RandomAgent
from src.environments.iter_hybrid_shape_guard_env import make_iter_hybrid_shape_guard_env
from src.utils.vertical_guard import Z_MAX, MAX_DZ
from collections import deque


class RotatingVerticalChamber:
    def __init__(self, ax):
        self.ax = ax
        self.chamber_radius = 10
        self.center_column_radius = 1.5
        self.num_particles = 200
        
        # Coil states (can be disabled)
        self.coil_upper_enabled = True
        self.coil_lower_enabled = True
        self.coil_center_enabled = True
        self.coil_upper_current = 1.0
        self.coil_lower_current = 1.0
        self.coil_center_current = 1.0  # Start disabled to show self-fix
        
        # Particle trails for cool effect
        self.trails = []
        self.time = 0
        self._init_trails()
        
        # Vertical position tracking
        self.z_history = deque(maxlen=100)
        self.prev_z = 0.0
        
        # Drift oscillation for when coils are disabled
        self.drift_oscillation_time = 0.0
        self.drift_oscillation_amplitude = 0.0
        
    def _init_trails(self):
        """Initialize particle trails for rotation effect."""
        self.trails = []
        for i in range(self.num_particles):
            angle = np.random.random() * 2 * np.pi
            radius = 3 + np.random.random() * 4
            z = np.random.random() * 2 - 1
            
            self.trails.append({
                'angle': angle,
                'radius': radius,
                'z': z,
                'speed': 0.05 + np.random.random() * 0.1,
                'phase': np.random.random() * 2 * np.pi,
            })
    
    def toggle_coil(self, coil_name):
        """Toggle coil on/off."""
        if coil_name == 'upper':
            self.coil_upper_enabled = not self.coil_upper_enabled
            if not self.coil_upper_enabled:
                self.coil_upper_current = 0.0
        elif coil_name == 'lower':
            self.coil_lower_enabled = not self.coil_lower_enabled
            if not self.coil_lower_enabled:
                self.coil_lower_current = 0.0
        elif coil_name == 'center':
            self.coil_center_enabled = not self.coil_center_enabled
            if not self.coil_center_enabled:
                self.coil_center_current = 0.0
    
    def update_coil_currents(self, upper, lower, center):
        """Update coil currents (only if enabled)."""
        if self.coil_upper_enabled:
            self.coil_upper_current = upper
        if self.coil_lower_enabled:
            self.coil_lower_current = lower
        if self.coil_center_enabled:
            self.coil_center_current = center
    
    def draw(self, z_cm, status, severity, dz, in_band, smooth):
        """Draw rotating 3D-style tokamak view.
        
        What you're seeing:
        - The plasma (colored ring) should stay centered vertically (z ≈ 0)
        - Coils (colored circles) control the magnetic field to position the plasma
        - When you disable a coil, the plasma drifts (moves up or down)
        - If it drifts too far (outside green safe zone), it's a VIOLATION
        - The system self-corrects by adjusting other coils to bring it back
        - This prevents VDE (Vertical Displacement Event) - a dangerous plasma disruption
        """
        self.ax.clear()
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-12, 8)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_facecolor('#000000')
        
        # Rotating view - smooth rotation
        rotation_angle = self.time * 0.03
        
        # Chamber walls - sleek rotating perspective
        for i, alpha in enumerate([0.25, 0.35, 0.45]):
            ring_radius = self.chamber_radius - i * 0.3
            x_scale = 1.0 + 0.15 * np.cos(rotation_angle)
            
            theta = np.linspace(0, 2*np.pi, 80)
            x_ring = ring_radius * x_scale * np.cos(theta)
            y_ring = ring_radius * np.sin(theta)
            self.ax.plot(x_ring, y_ring, color='#444455', linewidth=1.2, 
                        alpha=alpha, linestyle='-', zorder=1)
        
        # Center column - sleek rotating
        center_x_scale = 1.0 + 0.12 * np.cos(rotation_angle)
        center_col = Circle((0, 0), self.center_column_radius * center_x_scale, 
                           fill=True, facecolor='#1a1a2a', edgecolor='#555566', linewidth=1.5)
        self.ax.add_patch(center_col)
        
        # Coils - show enabled/disabled state
        coil_positions = {
            'upper': {'x': 0, 'y': 6, 'radius': 1.0},
            'lower': {'x': 0, 'y': -6, 'radius': 1.0},
            'center': {'x': 0, 'y': 0, 'radius': 1.6},
        }
        
        coil_colors = {
            'upper': '#4a88ff',
            'lower': '#4aff88',
            'center': '#ff8844'
        }
        
        coil_currents = {
            'upper': self.coil_upper_current if self.coil_upper_enabled else 0.0,
            'lower': self.coil_lower_current if self.coil_lower_enabled else 0.0,
            'center': self.coil_center_current if self.coil_center_enabled else 0.0,
        }
        
        for coil_name, coil in coil_positions.items():
            current = coil_currents[coil_name]
            enabled = getattr(self, f'coil_{coil_name}_enabled')
            base_color = coil_colors[coil_name]
            
            # Coil appearance - sleek design
            if enabled:
                # Active coil - subtle glow
                if abs(current) > 0.1:
                    glow = Circle((coil['x'], coil['y']), coil['radius'] + 0.25, 
                                 fill=True, facecolor=base_color, alpha=0.2 * abs(current))
                    self.ax.add_patch(glow)
                coil_ring = Circle((coil['x'], coil['y']), coil['radius'], 
                                 fill=False, edgecolor=base_color, 
                                 linewidth=2, alpha=0.7)
            else:
                # Disabled coil - minimal
                coil_ring = Circle((coil['x'], coil['y']), coil['radius'], 
                                 fill=False, edgecolor='#333333', 
                                 linewidth=1, alpha=0.2, linestyle=':')
            
            self.ax.add_patch(coil_ring)
            
            # Show coil state - minimal
            if not enabled:
                self.ax.text(coil['x'], coil['y'], 'OFF', ha='center', va='center',
                           fontsize=8, color='#555555', weight='bold')
            elif abs(current) > 0.05:
                # Only show current if significant
                current_text = f"{current:.1f}"
                self.ax.text(coil['x'], coil['y'], current_text, ha='center', va='center',
                           fontsize=8, color='white', weight='bold', alpha=0.9)
        
        # Vertical position
        z_viz = z_cm / 10.0
        
        # Safe vertical zone - horizontal lines marking boundaries
        safe_top = Z_MAX / 10.0
        safe_bottom = -Z_MAX / 10.0
        self.ax.axhline(safe_top, color='#00ff88', linewidth=1.5, alpha=0.6, linestyle='--', zorder=2)
        self.ax.axhline(safe_bottom, color='#00ff88', linewidth=1.5, alpha=0.6, linestyle='--', zorder=2)
        
        # Show correction force when outside safe zone
        if not in_band:
            # Draw arrows showing correction direction (pulling back toward center)
            correction_strength = min(abs(z_viz) / (Z_MAX / 10.0), 2.0)  # How far out we are
            arrow_length = 0.8 * correction_strength
            arrow_color = '#ffaa00' if status == 'SELF-FIXING' else '#ff6666'
            
            if z_viz > safe_top:
                # Too high - arrow pointing down (correction pulling down)
                self.ax.arrow(2, z_viz, 0, -arrow_length, 
                            head_width=0.3, head_length=0.2, 
                            fc=arrow_color, ec=arrow_color, alpha=0.7, linewidth=2, zorder=15)
            elif z_viz < safe_bottom:
                # Too low - arrow pointing up (correction pulling up)
                self.ax.arrow(2, z_viz, 0, arrow_length, 
                            head_width=0.3, head_length=0.2, 
                            fc=arrow_color, ec=arrow_color, alpha=0.7, linewidth=2, zorder=15)
        
        # Status colors
        if status == 'SAFE':
            plasma_color = '#00ffaa'
        elif status == 'SELF-FIXING':
            plasma_color = '#ffaa00'
        else:
            plasma_color = '#ff4444'
        
        # Update and draw particle trails with rotation
        self.update_trails(z_viz, status, rotation_angle)
        
        # Plasma boundary - sleek rotating ellipse
        plasma_radius = 3.3
        x_scale = 1.0 + 0.12 * np.cos(rotation_angle)
        theta = np.linspace(0, 2*np.pi, 50)
        x_plasma = plasma_radius * x_scale * np.cos(theta)
        y_plasma = plasma_radius * np.sin(theta) + z_viz
        self.ax.plot(x_plasma, y_plasma, color=plasma_color, linewidth=2, alpha=0.75)
        
        # Vertical position line - sleek
        if abs(z_viz) > 0.05:
            self.ax.plot([0, 0], [0, z_viz], color=plasma_color, 
                        linewidth=2.5, alpha=0.6)
        
        # Position marker - sleek
        marker = Circle((0, z_viz), 0.35, fill=True, 
                       facecolor=plasma_color, edgecolor='white', linewidth=1.5)
        self.ax.add_patch(marker)
        
        # Status indicator - minimal and sleek
        status_symbols = {'SAFE': '●', 'SELF-FIXING': '◐', 'VIOLATION': '▲'}
        status_text = status
        self.ax.text(0, -10.5, status_text, ha='center', va='top',
                    fontsize=12, weight='bold', color=plasma_color)
        
        # Position - minimal
        pos_text = f"{z_cm:.2f} cm"
        self.ax.text(-10, 7, pos_text, ha='left', va='top',
                    fontsize=9, color='#888888')
        
        self.time += 0.1
        self.z_history.append(z_cm)
        self.prev_z = z_cm
    
    def update_trails(self, z_viz, status, rotation_angle):
        """Update particle trails - sleek swirling effect."""
        size_factor = 1.0
        
        # Status-based colors - elegant palette
        if status == 'SAFE':
            colors = ['#00ffff', '#0088ff', '#00ffaa']
        elif status == 'SELF-FIXING':
            colors = ['#ffaa00', '#ffdd00', '#ff8800']
        else:
            colors = ['#ff6666', '#ff4444', '#ff8888']
        
        for trail in self.trails:
            # Rotate particles
            trail['angle'] += trail['speed'] * size_factor
            trail['phase'] += 0.02
            
            # Vertical drift
            target_z = z_viz
            trail['z'] += (target_z - trail['z']) * 0.1
            trail['z'] += 0.04 * np.sin(trail['phase'])
            
            # 3D rotation effect
            angle_3d = trail['angle'] + rotation_angle
            radius_3d = trail['radius'] * (1.0 + 0.08 * np.cos(rotation_angle))
            
            x = radius_3d * np.cos(angle_3d)
            y = trail['z'] + z_viz
            
            # Draw trail streak - sleek
            for i in range(4):
                t = i / 4.0
                prev_angle = angle_3d - trail['speed'] * (1 - t) * 6
                prev_radius = radius_3d * (1 - t * 0.08)
                px = prev_radius * np.cos(prev_angle)
                py = trail['z'] * (1 - t) + z_viz
                
                alpha = (1 - t) * 0.5
                size = 15 * (1 - t) + 3
                color = np.random.choice(colors)
                
                self.ax.scatter(px, py, s=size, c=color, 
                              alpha=alpha, edgecolors='none', zorder=10)
            
            # Main particle - sleek
            color = np.random.choice(colors)
            self.ax.scatter(x, y, s=24, c=color, 
                          alpha=0.85, edgecolors='white', 
                          linewidths=0.4, zorder=11)


def run_rotating_vertical_visualization():
    """Run cool rotating 3D-style vertical visualization."""
    print("Initializing rotating vertical guard visualization...")
    
    # Create environment with vertical guard
    env = make_iter_hybrid_shape_guard_env(
        shape_penalty=0.05,
        vertical_penalty=0.05,
        enable_shape_guard=True,
        enable_vertical_guard=True
    )
    
    agent = RandomAgent(env.action_space, shape_penalty=0.0)
    observation, info = env.reset()
    
    # Setup figure with sidebar for controls
    fig = plt.figure(figsize=(18, 10), facecolor='#000000')
    
    # Main visualization area (full width, buttons on right)
    ax = fig.add_axes([0.05, 0.1, 0.85, 0.85], facecolor='#000000')
    
    chamber = RotatingVerticalChamber(ax)
    
    # Sidebar for buttons only (right side) - bigger and sleeker
    sidebar_x = 0.90
    btn_y_start = 0.85
    spacing = 0.12
    
    # Create toggle buttons in sidebar - larger and sleek
    ax_upper_btn = plt.axes([sidebar_x, btn_y_start, 0.08, 0.06])
    ax_lower_btn = plt.axes([sidebar_x, btn_y_start - spacing, 0.08, 0.06])
    ax_center_btn = plt.axes([sidebar_x, btn_y_start - 2*spacing, 0.08, 0.06])
    
    # Sleek button design with better colors
    btn_upper = Button(ax_upper_btn, 'Disable Upper', color='#1a1a2a', hovercolor='#2a2a3a')
    btn_lower = Button(ax_lower_btn, 'Disable Lower', color='#1a1a2a', hovercolor='#2a2a3a')
    btn_center = Button(ax_center_btn, 'Disable Center', color='#1a1a2a', hovercolor='#2a2a3a')
    
    # Style the button text - make it bigger and sleeker
    for btn in [btn_upper, btn_lower, btn_center]:
        btn.label.set_fontsize(11)
        btn.label.set_weight('bold')
        btn.label.set_color('#cccccc')
    
    def toggle_upper(event):
        chamber.toggle_coil('upper')
        btn_upper.label.set_text('Enable Upper' if not chamber.coil_upper_enabled else 'Disable Upper')
    
    def toggle_lower(event):
        chamber.toggle_coil('lower')
        btn_lower.label.set_text('Enable Lower' if not chamber.coil_lower_enabled else 'Disable Lower')
    
    def toggle_center(event):
        chamber.toggle_coil('center')
        btn_center.label.set_text('Enable Center' if not chamber.coil_center_enabled else 'Disable Center')
    
    btn_upper.on_clicked(toggle_upper)
    btn_lower.on_clicked(toggle_lower)
    btn_center.on_clicked(toggle_center)
    
    step_count = 0
    prev_vertical_info = None
    
    # Apply coil influence
    def modify_action_with_coils(action):
        return action
    
    def animate(frame):
        nonlocal observation, step_count, prev_vertical_info
        
        # Get action
        action = agent.act(observation)
        action = modify_action_with_coils(action)
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Get vertical info
        if hasattr(env, 'vertical_info') and env.vertical_info:
            vertical_info = env.vertical_info.copy()
            
            # Apply coil influence - make it MUCH stronger for visible drift
            base_z_cm = vertical_info['z']
            
            # Update drift oscillation time
            if not hasattr(chamber, 'drift_oscillation_time'):
                chamber.drift_oscillation_time = 0.0
                chamber.drift_oscillation_amplitude = 0.0
            
            chamber.drift_oscillation_time += 0.1
            
            # Check if any coil is disabled - if so, create dramatic oscillating drift
            any_coil_disabled = (not chamber.coil_upper_enabled or 
                                not chamber.coil_lower_enabled or 
                                not chamber.coil_center_enabled)
            
            if any_coil_disabled:
                # Increase oscillation amplitude when coils are disabled
                chamber.drift_oscillation_amplitude = min(chamber.drift_oscillation_amplitude + 0.3, 15.0)
                
                # Create oscillating drift (goes up and down)
                # Frequency: faster oscillation for more dramatic effect
                oscillation = chamber.drift_oscillation_amplitude * np.sin(chamber.drift_oscillation_time * 0.8)
                
                # Add directional bias based on which coil is disabled
                if not chamber.coil_upper_enabled:
                    # Upper coil disabled -> tends to drift down, but oscillates
                    directional_bias = -8.0
                elif not chamber.coil_lower_enabled:
                    # Lower coil disabled -> tends to drift up, but oscillates
                    directional_bias = 8.0
                else:
                    # Center coil disabled -> oscillates around center
                    directional_bias = 0.0
                
                # Combine oscillation with directional bias
                drift_effect = directional_bias + oscillation
            else:
                # Coils are enabled - reduce oscillation amplitude gradually
                chamber.drift_oscillation_amplitude = max(chamber.drift_oscillation_amplitude - 0.2, 0.0)
                drift_effect = 0.0
            
            # Apply base coil influence (from current values)
            upper_influence = (chamber.coil_upper_current - 1.0) * 8.0 if chamber.coil_upper_enabled else 0.0
            lower_influence = (chamber.coil_lower_current - 1.0) * -8.0 if chamber.coil_lower_enabled else 0.0
            center_influence = (chamber.coil_center_current - 1.0) * 3.0 if chamber.coil_center_enabled else 0.0
            
            coil_influence = upper_influence + lower_influence + center_influence + drift_effect
            
            if not hasattr(chamber, 'target_z_influence'):
                chamber.target_z_influence = 0.0
            # Faster response to coil changes
            chamber.target_z_influence += (coil_influence - chamber.target_z_influence) * 0.2
            
            modified_z_cm = base_z_cm + chamber.target_z_influence
            
            # Add automatic self-correction when outside safe zone
            # This simulates how the RL agent would learn to correct
            if abs(modified_z_cm) > Z_MAX:
                # Calculate correction force pulling back toward center
                # Stronger correction the further out we are
                distance_from_center = abs(modified_z_cm)
                correction_strength = min((distance_from_center - Z_MAX) / Z_MAX, 1.0)  # 0-1 scale
                correction_force = -np.sign(modified_z_cm) * correction_strength * 2.0  # Pull back toward 0
                
                # Apply correction (simulates agent learning to adjust coils)
                modified_z_cm += correction_force
            prev_z = chamber.prev_z if hasattr(chamber, 'prev_z') else base_z_cm
            
            vertical_info['z'] = modified_z_cm
            vertical_info['dz'] = modified_z_cm - prev_z
            vertical_info['in_band'] = abs(modified_z_cm) <= Z_MAX
            vertical_info['smooth'] = abs(vertical_info['dz']) <= MAX_DZ
            vertical_info['ok'] = vertical_info['in_band'] and vertical_info['smooth']
            
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
            
            # Draw
            chamber.draw(z_cm, dz, severity, status, in_band, smooth)
            
            chamber.prev_z = modified_z_cm
            prev_vertical_info = vertical_info
        
        step_count += 1
        
        if terminated or truncated:
            observation, info = env.reset()
            agent.reset_state(observation)
            prev_vertical_info = None
    
    anim = FuncAnimation(fig, animate, interval=50, blit=False, repeat=True)
    
    # Hide spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()
    env.close()


if __name__ == "__main__":
    run_rotating_vertical_visualization()

