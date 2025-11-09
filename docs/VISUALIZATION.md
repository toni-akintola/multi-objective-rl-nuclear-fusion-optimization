# Visualization Guide

This guide covers all visualization tools and demos in the project.

## Available Visualizations

### 1. Streamlit App (Interactive Dashboard)
The main interactive dashboard for exploring plasma shape dynamics.

```bash
streamlit run app.py
```

Features:
- Real-time shape trajectory visualization
- Interactive parameter controls
- Shape guard diagnostics
- 2D and 3D trajectory views
- Self-fixing action tracking

### 2. Animated Shape Trajectory
Watch the plasma shape evolve in real-time with animated plots.

```bash
# Default speed (1.0x)
python src/visualization/animate_shape.py

# Faster animation (2x speed)
python src/visualization/animate_shape.py 2.0

# Custom speed and steps
python src/visualization/animate_shape.py 1.5 200
```

Shows:
- Live 2D trajectory (β_N vs q95)
- 3D shape space evolution
- Safety zone boundaries
- Self-fixing actions highlighted

### 3. Chamber Visualization (3D)
3D visualization of the tokamak chamber and plasma shape.

```bash
# Live 3D chamber view
python src/visualization/visualize_chamber_live.py

# Vertical cross-section
python src/visualization/visualize_vertical_live.py

# 3D vertical view
python src/visualization/visualize_vertical_3d.py
```

### 4. Web-Based Visualizations

#### Chamber Web Server
```bash
python src/servers/chamber_web_server.py
```
Then open: http://localhost:8000

#### Chamber WebSocket Server (Real-time)
```bash
python src/servers/chamber_websocket_server.py
```

#### API Server (REST API)
```bash
python src/servers/api_server.py
```
API available at: http://localhost:8000/docs

### 5. Launch Visualization Helper
Quick launcher for all visualization options:

```bash
python scripts/launch_visualization.py
```

## Visualization Components

### Shape Parameters Tracked
- **β_N** (Normalized Beta): Plasma pressure
- **q_min** (Minimum Safety Factor): MHD stability
- **q95** (Edge Safety Factor): Confinement quality

### Safety Zones
- Green: Safe operating region
- Red: Violation (outside safe bounds)
- Orange: Self-fixing (corrective action reducing severity)

### Trajectory Features
- **Path**: Blue line showing historical trajectory
- **Current Position**: Large marker (green/red/orange)
- **Recent Trail**: Cyan dots showing last 5 positions
- **Safe Box**: Green dashed rectangle (2D) or box (3D)

## Tips

1. **Performance**: For smoother animations, reduce the number of steps or increase animation speed
2. **Interactive Mode**: Use Streamlit app for best interactivity
3. **Recording**: Screenshots are automatically saved as PNG files
4. **3D Views**: Use mouse to rotate 3D visualizations

## Troubleshooting

**Animation too slow?**
```bash
python src/visualization/animate_shape.py 2.0  # 2x speed
```

**Port already in use?**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

**Import errors?**
Make sure you're in the project root and using the virtual environment:
```bash
source .venv/bin/activate  # or: uv sync
```
