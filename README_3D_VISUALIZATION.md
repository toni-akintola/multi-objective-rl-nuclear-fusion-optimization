# How to Run the 3D Tokamak Visualization

## Setup

1. **Install websockets:**
   ```bash
   pip install websockets
   # or
   uv sync
   ```

2. **Start the WebSocket server:**
   ```bash
   python websocket_server.py
   ```

3. **Open the visualization:**
   - Open `index.html` in your browser
   - Or serve it: `python -m http.server 8000` then go to `http://localhost:8000`

## What You'll See

- **3D Tokamak chamber** with metallic walls
- **Swirling plasma filaments** that change color based on status:
  - Cyan/Green = Safe
  - Orange = Self-fixing or rough
  - Red = Violation
- **Real-time updates** from your agent
- **Interactive camera** (drag to rotate, scroll to zoom)

## Connection

The WebSocket server (`websocket_server.py`) connects your `agent.py` to the `index.html` visualization:
- Runs the simulation using your agent
- Streams β_N, q_min, q95, violation severity
- Updates the 3D visualization in real-time

## Features

- **Connected to agent.py** - Uses your actual agent code
- **Real-time streaming** - Updates 30 times per second
- **3D visualization** - Much cooler than 2D plots!
- **Interactive** - Rotate and zoom to see different angles

