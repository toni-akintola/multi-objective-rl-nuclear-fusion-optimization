"use client"

import { useEffect, useRef, useState } from "react"

declare global {
  interface Window {
    loadPyodide: any
  }
}

export function ChamberPython() {
  const containerRef = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState("Loading Pyodide...")
  const [error, setError] = useState<string | null>(null)
  const pyodideRef = useRef<any>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let mounted = true

    const initPyodide = async () => {
      try {
        setStatus("Loading Pyodide runtime...")
        
        // Load Pyodide
        if (!window.loadPyodide) {
          const script = document.createElement("script")
          script.src = "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"
          script.async = true
          await new Promise((resolve, reject) => {
            script.onload = resolve
            script.onerror = reject
            document.head.appendChild(script)
          })
        }

        if (!mounted) return

        setStatus("Initializing Python environment...")
        const pyodide = await window.loadPyodide({
          indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
        })
        
        if (!mounted) return

        pyodideRef.current = pyodide

        setStatus("Installing packages...")
        await pyodide.loadPackage(["numpy", "matplotlib"])

        if (!mounted) return

        setStatus("Loading visualization code...")
        
        // Use inline Python code (exact copy from visualize_chamber_live.py)
        const pythonCode = `
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import io
import base64
import js
from pyodide.ffi import create_proxy
import asyncio

# Set up matplotlib figure
fig = plt.figure(figsize=(14, 10), facecolor='#0a0a0f')
ax = fig.add_subplot(111)

# Copy the exact LiveTokamakChamber class from visualize_chamber_live.py
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
                           fill=True, facecolor='#1a1a2a', 
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
        
        # Status
        if ok:
            status = "🟢 SAFE"
            status_color = 'green'
        elif self.is_corrective:
            status = "🟠 SELF-FIXING! (Severity ↓)"
            status_color = 'orange'
        else:
            status = "🔴 VIOLATION"
            status_color = 'red'
        
        # Parameter bars (simplified for now)
        # ... (rest of draw method from visualize_chamber_live.py)
        
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
        
        # Status text
        status_text = f"{status} | β_N={beta_N:.2f} | q_min={q_min:.2f} | q95={q95:.2f}"
        self.ax.text(0, -9, status_text, 
                    ha='center', va='top', fontsize=12, color=status_color, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor=status_color, linewidth=2))

# Initialize
chamber = LiveTokamakChamber(ax)

# Update frame function - renders to PNG and displays
def update_frame(beta_N=1.5, q_min=2.0, q95=4.0, ok=True, violation=0.0, in_box=True, smooth=True):
    try:
        chamber.draw(beta_N, q_min, q95, ok, violation, in_box, smooth)
        
        # Render to PNG buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a0f')
        buf.seek(0)
        
        # Convert to base64 data URL
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        data_url = f"data:image/png;base64,{img_data}"
        
        # Update image element
        img = js.document.getElementById('chamber-img')
        if img:
            img.src = data_url
    except Exception as e:
        js.console.log(f"Error in update_frame: {e}")

# Draw initial frame
js.console.log("Drawing initial frame...")
update_frame()
js.console.log("✅ Initial frame drawn")

# Connect to WebSocket for real-time data
import json
from pyodide.ffi import create_proxy

def connect_websocket():
    try:
        ws = js.WebSocket.new("ws://localhost:8765")
        
        def on_message(event):
            try:
                data = json.loads(event.data)
                if data.get('type') == 'chamber_data':
                    d = data['data']
                    update_frame(
                        beta_N=d['beta_N'],
                        q_min=d['q_min'],
                        q95=d['q95'],
                        ok=d['ok'],
                        violation=d['violation'],
                        in_box=d['in_box'],
                        smooth=d.get('smooth', True)
                    )
            except Exception as e:
                js.console.log(f"Error processing message: {e}")
        
        ws.onmessage = create_proxy(on_message)
        ws.onopen = create_proxy(lambda e: js.console.log("✅ Connected to WebSocket"))
        ws.onerror = create_proxy(lambda e: js.console.log("WebSocket error"))
        ws.onclose = create_proxy(lambda e: js.console.log("WebSocket closed"))
        
        return ws
    except Exception as e:
        js.console.log(f"WebSocket not available: {e}")
        return None

# Try to connect, fallback to mock data
ws = connect_websocket()
if not ws:
    # Use mock data if WebSocket unavailable
    import asyncio
    step = [0]  # Use list to allow modification in nested function
    
    async def mock_loop():
        while True:
            step[0] += 1
            beta_N = 0.5 + (np.sin(step[0] * 0.1) + 1) * 1.25
            q_min = 1.0 + np.abs(np.sin(step[0] * 0.15)) * 2.0
            q95 = 3.0 + (np.cos(step[0] * 0.12) + 1) * 1.0
            ok = (0.5 <= beta_N <= 3.0 and q_min >= 1.0 and 3.0 <= q95 <= 5.0)
            violation = 0.0 if ok else 0.5 + np.random.random() * 0.5
            update_frame(beta_N, q_min, q95, ok, violation, ok, True)
            await asyncio.sleep(0.1)
    
    asyncio.create_task(mock_loop())
`

        await pyodide.runPython(pythonCode)

        if (!mounted) return

        // Give it a moment to render the first frame
        await new Promise(resolve => setTimeout(resolve, 500))
        
        setStatus("Running...")
        setError(null)
      } catch (err: any) {
        if (mounted) {
          setError(err.message || "Failed to initialize Pyodide")
          setStatus("Error")
          console.error("Pyodide error:", err)
        }
      }
    }

    initPyodide()

    return () => {
      mounted = false
    }
  }, [])

  return (
    <div className="relative w-full">
      <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-foreground/20 bg-foreground/5 backdrop-blur-md">
        {error ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <p className="text-red-500">{error}</p>
              <p className="mt-2 text-sm text-foreground/50">
                Pyodide requires a modern browser with WebAssembly support
              </p>
            </div>
          </div>
        ) : (
          <div ref={containerRef} className="h-full w-full">
            <canvas
              id="chamber-canvas"
              ref={canvasRef}
              className="h-full w-full"
              style={{ display: status === "Running..." ? "block" : "none" }}
            />
            <img
              id="chamber-img"
              alt="Chamber Visualization"
              className="h-full w-full object-contain"
              style={{ display: status === "Running..." ? "block" : "none" }}
            />
            {status !== "Running..." && (
              <div className="flex h-full items-center justify-center">
                <div className="text-center">
                  <div className="mb-4 inline-block h-8 w-8 animate-spin rounded-full border-4 border-foreground/20 border-t-foreground"></div>
                  <p className="font-mono text-sm text-foreground/70">{status}</p>
                  <p className="mt-2 font-mono text-xs text-foreground/50">
                    This may take a minute on first load...
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

