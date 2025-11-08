# How to Run the Animation

## 🎬 Option 1: Standalone Animation (Recommended)

### Basic Usage:
```bash
python animate_shape.py
```

### With Custom Speed:
```bash
# Slow motion (half speed)
python animate_shape.py 0.5

# Normal speed
python animate_shape.py 1.0

# Fast (double speed)
python animate_shape.py 2.0

# Very slow (quarter speed)
python animate_shape.py 0.25
```

### With Custom Number of Steps:
```bash
# 50 steps at normal speed
python animate_shape.py 1.0 50

# 200 steps at half speed
python animate_shape.py 0.5 200
```

**What happens:**
- Opens a matplotlib window
- Shows 2D and 3D views side-by-side
- Animates the trajectory in real-time
- You can see the point moving through space
- Close the window to stop

## 🌐 Option 2: Streamlit Web App

### Run the app:
```bash
streamlit run app.py
```

### Then:
1. Browser opens automatically (or go to `http://localhost:8501`)
2. Adjust parameters in the sidebar
3. Click "🚀 Run Simulation"
4. Scroll down to "🎬 Live Trajectory Animation"
5. Adjust speed slider (0.1 = very slow, 5.0 = fast)
6. Click "▶️ Play Animation"
7. Watch the trajectory animate step-by-step

**Note:** Keep the terminal running - closing it stops the web server.

## 🎯 Quick Start

**Easiest way to see the animation:**
```bash
python animate_shape.py 0.5
```

This will:
- Run the simulation
- Show you the trajectory moving in slow motion
- Display both 2D and 3D views
- Update in real-time

## 💡 Tips

- **Slow it down** (0.1-0.5) to see each step clearly
- **Speed it up** (2.0-5.0) for quick overview
- The animation loops automatically
- Close the window or press Ctrl+C to stop

