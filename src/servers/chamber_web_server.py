"""
Web server to serve the matplotlib chamber visualization.
Embed this in the Next.js page using an iframe.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import gymnasium as gym
import gymtorax
from agent import RandomAgent
import importlib.util
from pathlib import Path
from flask import Flask, Response
import io
import base64
import threading
import time

# Import shape guard
spec = importlib.util.spec_from_file_location(
    "shape_guard",
    Path(__file__).parent / "optimization-for-constraints" / "shape_guard.py"
)
shape_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape_guard)
shape_violation = shape_guard.shape_violation

app = Flask(__name__)

# Global state
current_frame = None
frame_lock = threading.Lock()

class LiveTokamakChamber:
    def __init__(self, ax):
        self.ax = ax
        self.chamber_radius = 10
        self.num_trails = 300
        self.trails = []
        self.time = 0
        self.prev_severity = float('inf')
        self.is_corrective = False
        self.violation_history = []
        self.prev_beta_N = None
        self.prev_q_min = None
        self.prev_q95 = None
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
    
    def update_trails(self, beta_N, ok, violation, is_corrective=False):
        """Update trail colors and behavior based on state."""
        size_factor = 0.7 + (beta_N - 0.5) / (3.0 - 0.5) * 0.3
        
        if violation < self.prev_severity and not ok:
            is_corrective = True
        self.prev_severity = violation
        self.is_corrective = is_corrective
        
        self.violation_history.append(violation)
        if len(self.violation_history) > 50:
            self.violation_history.pop(0)
        
        for trail in self.trails:
            trail['angle'] += trail['speed'] * size_factor
            trail['phase'] += 0.02
            trail['z'] += 0.01 * np.sin(trail['phase'])
            if trail['z'] > 2:
                trail['z'] = -2
            elif trail['z'] < -2:
                trail['z'] = 2
            
            if ok:
                trail['color'] = np.random.choice(['cyan', 'blue', 'green'], p=[0.4, 0.4, 0.2])
            elif is_corrective:
                trail['color'] = np.random.choice(['orange', 'yellow', 'lime'], p=[0.5, 0.3, 0.2])
            elif violation < 0.5:
                trail['color'] = np.random.choice(['orange', 'yellow'], p=[0.7, 0.3])
            else:
                trail['color'] = np.random.choice(['red', 'magenta', 'pink'], p=[0.5, 0.3, 0.2])
    
    def draw(self, beta_N, q_min, q95, ok, violation, in_box, smooth):
        """Draw the chamber."""
        self.ax.clear()
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-12, 8)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_facecolor('#0a0a0f')
        
        # Chamber walls
        chamber = Circle((0, 0), self.chamber_radius, 
                        fill=False, edgecolor='#3a3a4a', 
                        linewidth=3, alpha=0.6, zorder=1)
        self.ax.add_patch(chamber)
        
        inner = Circle((0, 0), self.chamber_radius * 0.9, 
                      fill=False, edgecolor='#2a2a3a', 
                      linewidth=1, alpha=0.4, linestyle='--', zorder=1)
        self.ax.add_patch(inner)
        
        # Center column
        col_radius = 1.5 * (1 + 0.2 * np.sin(self.time))
        center_col = Circle((0, 0), col_radius, 
                           fill=True, color='#1a1a2a', 
                           edgecolor='#4a4a5a', linewidth=2, alpha=0.8)
        self.ax.add_patch(center_col)
        
        # Update and draw trails
        self.update_trails(beta_N, ok, violation, self.is_corrective)
        size_factor = 0.7 + (beta_N - 0.5) / (3.0 - 0.5) * 0.3
        
        for trail in self.trails:
            angle = trail['angle']
            radius = trail['radius'] * size_factor
            
            plasma_boundary_radius = 3 + (beta_N - 0.5) / (3.0 - 0.5) * 4
            if radius > plasma_boundary_radius * 0.95:
                radius = plasma_boundary_radius * 0.95
            
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
        
        # Status (using text instead of emoji to avoid font issues)
        if ok:
            status = "[SAFE]"
            status_color = 'green'
        elif self.is_corrective:
            status = "[SELF-FIXING! (Severity ↓)]"
            status_color = 'orange'
        else:
            status = "[VIOLATION]"
            status_color = 'red'
        
        # Parameter bars
        safe_bounds = {
            'β_N': (0.5, 3.0),
            'q_min': (1.0, float('inf')),
            'q95': (3.0, 5.0)
        }
        
        bar_x = 10.5
        bar_width = 0.8
        bar_spacing = 1.2
        
        params = [
            ('beta_N', beta_N, safe_bounds['β_N'][0], safe_bounds['β_N'][1], 'cyan', 'β_N: Pressure'),
            ('q_min', q_min, safe_bounds['q_min'][0], None, 'green', 'q_min: Internal stability'),
            ('q95', q95, safe_bounds['q95'][0], safe_bounds['q95'][1], 'blue', 'q95: Edge stability')
        ]
        
        for i, (name, value, min_val, max_val, color, label) in enumerate(params):
            bar_y = -8 + i * bar_spacing
            
            in_bounds = True
            if max_val is None:
                in_bounds = value >= min_val
            else:
                in_bounds = min_val <= value <= max_val
            
            bar_color = '#00ff00' if in_bounds else '#ff0000'
            
            if max_val is None:
                safe_range = 2.0
                safe_start = min_val
            else:
                safe_range = max_val - min_val
                safe_start = min_val
            
            if max_val is None:
                display_val = (value - min_val) / 1.0
                display_val = min(display_val, 2.0)
            else:
                display_val = (value - min_val) / safe_range
                display_val = max(0, min(display_val, 1.5))
            
            safe_height = 0.3
            self.ax.add_patch(plt.Rectangle((bar_x, bar_y - safe_height/2), bar_width, safe_height,
                                          facecolor='#004400', edgecolor='#00aa00', linewidth=1, alpha=0.5, zorder=20))
            
            indicator_y = bar_y - safe_height/2 + (display_val / 1.5) * safe_height
            indicator_color = bar_color
            self.ax.scatter(bar_x + bar_width/2, indicator_y, s=200, c=indicator_color, 
                          edgecolors='white', linewidths=2, zorder=21, marker='o')
            
            prev_attr_map = {'beta_N': 'prev_beta_N', 'q_min': 'prev_q_min', 'q95': 'prev_q95'}
            prev_attr = prev_attr_map.get(name)
            if prev_attr and hasattr(self, prev_attr):
                prev_val = getattr(self, prev_attr)
                if prev_val is not None:
                    if value > prev_val:
                        arrow_color = '#ff0000' if not in_bounds else '#00ff00'
                        self.ax.arrow(bar_x + bar_width + 0.2, indicator_y, 0.3, 0, 
                                    head_width=0.1, head_length=0.1, fc=arrow_color, ec=arrow_color, zorder=22)
                    elif value < prev_val:
                        arrow_color = '#00ff00' if not in_bounds else '#ff0000'
                        self.ax.arrow(bar_x + bar_width + 0.2, indicator_y, -0.3, 0, 
                                    head_width=0.1, head_length=0.1, fc=arrow_color, ec=arrow_color, zorder=22)
            
            if prev_attr:
                setattr(self, prev_attr, value)
            
            self.ax.text(bar_x + bar_width/2, bar_y - 0.5, label, 
                        ha='center', va='top', fontsize=9, color='white', weight='bold', zorder=23)
        
        # Status circle
        status_circle_size = 80 if ok else (100 if self.is_corrective else 60)
        status_circle_color = '#00ff00' if ok else ('#ff8800' if self.is_corrective else '#ff0000')
        self.ax.scatter(-10.5, 0, s=status_circle_size, facecolor=status_circle_color, 
                       alpha=0.8, edgecolors='white', linewidths=3, zorder=20)
        
        if ok:
            symbol = 'OK'
        elif self.is_corrective:
            symbol = 'FIX'
        else:
            symbol = 'X'
        
        self.ax.text(-10.5, 0, symbol, ha='center', va='center', fontsize=20, 
                    color='white', weight='bold', zorder=21)
        
        # Main status text
        status_text = f"{status} | β_N={beta_N:.2f} | q_min={q_min:.2f} | q95={q95:.2f}"
        self.ax.text(0, -9, status_text, 
                    ha='center', va='top', fontsize=12, color=status_color, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor=status_color, linewidth=2))
        
        # Plasma boundary
        base_radius = 3 + (beta_N - 0.5) / (3.0 - 0.5) * 4
        elongation = 1.0 + (q95 - 3.0) / (5.0 - 3.0) * 0.3
        triangularity = 0.0 + (1.0 - q_min) / (1.0 - 0.5) * 0.2
        
        theta = np.linspace(0, 2 * np.pi, 100)
        R0 = 0.0
        a = base_radius
        R_plasma = R0 + a * (np.cos(theta) + triangularity * np.cos(2*theta))
        Z_plasma = a * elongation * np.sin(theta)
        
        self.ax.plot(R_plasma, Z_plasma, color=status_color, 
                    linewidth=3, alpha=0.9, linestyle='-', zorder=3)
        
        # Violation severity indicator ring
        if len(self.violation_history) > 5:
            n_points = len(self.violation_history)
            angles = np.linspace(0, 2 * np.pi, n_points)
            
            plasma_boundary_radius = 3 + (beta_N - 0.5) / (3.0 - 0.5) * 4
            base_radius = plasma_boundary_radius * 0.6
            radii = base_radius + np.array(self.violation_history) * 1.2
            radii = np.clip(radii, base_radius * 0.4, plasma_boundary_radius * 0.9)
            
            trend_x = radii * np.cos(angles)
            trend_y = radii * np.sin(angles)
            
            self.ax.plot(trend_x, trend_y, color=status_color, linewidth=2, alpha=0.6, zorder=2)
            
            base_x = base_radius * np.cos(angles)
            base_y = base_radius * np.sin(angles)
            for i in range(len(angles) - 1):
                self.ax.fill([base_x[i], base_x[i+1], trend_x[i+1], trend_x[i]], 
                           [base_y[i], base_y[i+1], trend_y[i+1], trend_y[i]], 
                           color=status_color, alpha=0.1, zorder=2)
            
            if len(self.violation_history) > 10:
                recent_avg = np.mean(self.violation_history[-5:])
                older_avg = np.mean(self.violation_history[-15:-5])
                if recent_avg < older_avg:
                    self.ax.text(0, -11, "↓ Improving", fontsize=9, color='lime', 
                               weight='bold', ha='center', 
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
                elif recent_avg > older_avg:
                    self.ax.text(0, -11, "↑ Worsening", fontsize=9, color='red', 
                               weight='bold', ha='center',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))


# Global simulation state
env = None
agent = None
chamber = None
fig = None
ax = None
observation = None
step_count = 0
simulation_lock = threading.Lock()

def init_simulation():
    """Initialize the simulation."""
    global env, agent, chamber, fig, ax, observation, step_count
    
    env = gym.make("gymtorax/IterHybrid-v0")
    agent = RandomAgent(
        action_space=env.action_space,
        shape_penalty=0.1,
        damp_on_violation=True,
        damp_factor=0.5,
    )
    
    observation, _ = env.reset()
    agent.reset_state(observation)
    
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0a0a0f')
    chamber = LiveTokamakChamber(ax)
    step_count = 0

def update_frame():
    """Update one frame of the simulation."""
    global observation, step_count, current_frame
    
    if env is None or agent is None or fig is None:
        return
    
    with simulation_lock:
        action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        reward = agent.apply_shape_safety(reward, observation)
        
        if agent.last_shape_info:
            shape_info = agent.last_shape_info
            beta_N = float(shape_info["shape"][0])
            q_min = float(shape_info["shape"][1])
            q95 = float(shape_info["shape"][2])
            violation = float(shape_info["severity"])
            ok = shape_info["ok"]
            in_box = shape_info["in_box"]
            smooth = shape_info["smooth"]
            
            was_in_violation = (hasattr(chamber, 'prev_severity') and 
                              chamber.prev_severity != float('inf') and 
                              not (ok and chamber.prev_severity == 0))
            is_corrective = False
            if was_in_violation and not ok:
                is_corrective = violation < chamber.prev_severity
            
            chamber.draw(beta_N, q_min, q95, ok, violation, in_box, smooth)
            
            # Save frame to buffer - do this while holding the lock
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a0f')
                buf.seek(0)
                
                with frame_lock:
                    current_frame = buf.read()
            except Exception as e:
                print(f"Error saving frame: {e}")
                # Create a blank frame on error
                buf = io.BytesIO()
                fig2, ax2 = plt.subplots(figsize=(14, 10), facecolor='#0a0a0f')
                ax2.text(0.5, 0.5, 'Error rendering frame', ha='center', va='center', 
                       color='white', fontsize=20, transform=ax2.transAxes)
                fig2.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a0f')
                plt.close(fig2)
                buf.seek(0)
                with frame_lock:
                    current_frame = buf.read()
        
        step_count += 1
        
        if terminated or truncated:
            observation, _ = env.reset()
            agent.reset_state(observation)
            step_count = 0

@app.route('/')
def index():
    """Serve the visualization page."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chamber Visualization</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                background: #0a0a0f;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            img {
                max-width: 100%;
                max-height: 100%;
            }
        </style>
    </head>
    <body>
        <img id="frame" src="/frame" />
        <script>
            setInterval(() => {
                document.getElementById('frame').src = '/frame?t=' + Date.now();
            }, 100);
        </script>
    </body>
    </html>
    '''

@app.route('/frame')
def get_frame():
    """Get the current frame."""
    try:
        update_frame()
        
        with frame_lock:
            if current_frame:
                return Response(current_frame, mimetype='image/png')
            else:
                # Return a blank frame
                buf = io.BytesIO()
                fig2, ax2 = plt.subplots(figsize=(14, 10), facecolor='#0a0a0f')
                ax2.text(0.5, 0.5, 'Initializing...', ha='center', va='center', 
                       color='white', fontsize=20, transform=ax2.transAxes)
                fig2.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a0f')
                plt.close(fig2)
                buf.seek(0)
                return Response(buf.read(), mimetype='image/png')
    except Exception as e:
        print(f"Error in get_frame: {e}")
        # Return error frame
        buf = io.BytesIO()
        fig2, ax2 = plt.subplots(figsize=(14, 10), facecolor='#0a0a0f')
        ax2.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', 
               color='red', fontsize=16, transform=ax2.transAxes)
        fig2.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a0f')
        plt.close(fig2)
        buf.seek(0)
        return Response(buf.read(), mimetype='image/png')

if __name__ == '__main__':
    print("Initializing simulation...")
    init_simulation()
    print("Starting web server on http://localhost:5001")
    print("Embed this in your Next.js page with: <iframe src='http://localhost:5001'></iframe>")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)

