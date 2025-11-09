# Chamber Visualization Setup

## Quick Start

To run the live chamber visualization in the Next.js demo website:

### 1. Start the Next.js Development Server

```bash
cd demo
npm run dev
```

The website will be available at: http://localhost:3000

### 2. Start the Python WebSocket Server

In a **separate terminal**, run:

```bash
python chamber_websocket_server.py
```

The server will start on `ws://localhost:8765` and stream simulation data.

### 3. Open the Chamber Visualization

Navigate to: **http://localhost:3000/chamber**

Or click the "Live Chamber Visualization" button on the fusion page.

## What You'll See

- **Real-time tokamak chamber visualization** with:
  - Swirling particle trails (300 particles)
  - Color-coded status:
    - 🟢 **Green/Cyan/Blue** = Safe operation
    - 🟠 **Orange/Yellow** = Self-fixing (recovering from violation)
    - 🔴 **Red/Magenta** = Violation (parameters out of bounds)
  - Plasma boundary that changes shape based on β_N, q_min, q95
  - Severity ring that expands/contracts with violation severity
  - Real-time parameter display

## Features

- **WebSocket Connection**: Real-time data streaming from Python simulation
- **Canvas Rendering**: Smooth 60 FPS animation
- **Status Indicators**: Visual feedback for safe/violation/self-fixing states
- **Parameter Display**: Shows β_N, q_min, q95 values in real-time

## Troubleshooting

### Connection Error

If you see "Failed to connect to visualization server":
1. Make sure `chamber_websocket_server.py` is running
2. Check that port 8765 is not in use
3. Verify the Python environment has all dependencies installed

### No Data Displaying

- Check the browser console for WebSocket errors
- Verify the Python server is receiving connections (you'll see "✅ Client connected!")
- Make sure the simulation is running (you should see step updates in the terminal)

## Architecture

- **Frontend**: Next.js React component with Canvas API
- **Backend**: Python WebSocket server running gymtorax simulation
- **Communication**: WebSocket protocol (ws://localhost:8765)
- **Data Format**: JSON messages with chamber state data

## Next Steps

- Add controls to pause/resume simulation
- Add parameter history graphs
- Add multiple visualization modes
- Integrate with other visualizations (vertical guard, etc.)

