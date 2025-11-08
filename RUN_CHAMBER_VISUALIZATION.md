# Tokamak Chamber Visualization

## Quick Start

### Option 1: Static Test Visualization
```bash
python visualize_chamber.py
```

Shows a beautiful swirling particle chamber with:
- Dense particle trails in cyan/blue/green (safe)
- Orange/yellow trails (self-fixing)
- Red/magenta trails (violation)
- Real-time swirling motion
- Status display

### Option 2: Live Visualization (Connected to Agent)
```bash
python visualize_chamber_live.py
```

Shows real-time visualization connected to your agent:
- Updates based on actual simulation state
- Changes colors based on safety status
- Shows β_N, q_min, q95 values
- Particle trails respond to plasma parameters

## Features

- **Futuristic Chamber Design**: Dark industrial aesthetic with metallic walls
- **Swirling Particle Trails**: 200-300 particles creating dense, energetic patterns
- **Color-Coded Status**:
  - Cyan/Blue/Green = Safe operation
  - Orange/Yellow = Self-fixing
  - Red/Magenta = Violation
- **Dynamic Scaling**: Plasma size changes with β_N
- **Real-time Updates**: Connected to your agent simulation

## Controls

- Close window to stop
- Animation runs automatically
- Status updates in real-time

