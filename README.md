# Multi-Objective RL for Nuclear Fusion Optimization

A reinforcement learning project for optimizing plasma shape control in nuclear fusion reactors using shape constraints and self-fixing mechanisms.

## 🚀 Quick Start

### Installation
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Run the Main Demo
```bash
# Interactive mode with shape guard visualization
python main.py
```

### Web Interface (Streamlit)
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser to:
- Adjust parameters (shape penalty, damping, episodes)
- Run simulations interactively
- View real-time visualizations and animations
- Analyze step-by-step data

### Animation
```bash
# Standalone animation with custom speed
python src/visualization/animate_shape.py 0.5  # Slow motion (0.5x speed)
python src/visualization/animate_shape.py 2.0  # Fast forward (2x speed)
python src/visualization/animate_shape.py 1.0 200  # Normal speed, 200 steps
```

## 📁 Project Structure

```
nuclear/
├── src/                        # Source code
│   ├── agents/                 # Agent implementations
│   │   └── agent.py
│   ├── environments/           # Custom environments
│   │   └── iter_hybrid_shape_guard_env.py
│   ├── visualization/          # Visualization tools
│   │   ├── animate_shape.py
│   │   ├── visualize_chamber_live.py
│   │   ├── visualize_vertical_3d.py
│   │   └── visualize_vertical_live.py
│   ├── servers/                # Web servers & APIs
│   │   ├── api_server.py
│   │   ├── chamber_web_server.py
│   │   └── chamber_websocket_server.py
│   └── utils/                  # Utilities
│       └── shape_guard.py
│
├── scripts/                    # Executable scripts
│   ├── build_dataset.py
│   ├── launch_visualization.py
│   └── test_multi_env.py
│
├── train/                      # Training scripts
│   ├── offline_train.py
│   └── sac_train.py
│
├── eval/                       # Evaluation scripts
│   ├── cql_online_eval.py
│   ├── rand_eval.py
│   ├── sac_eval.py
│   └── surrogate_eval.py
│
├── tests/                      # Test files
│   └── test_surrogate_inference.py
│
├── docs/                       # Documentation
│   ├── ANALYSIS.md
│   ├── TRAINING.md
│   ├── VISUALIZATION.md
│   ├── SURROGATE_MODEL.md
│   └── CHANGELOG.md
│
├── data/                       # Data processing
├── models/                     # Saved models
├── demo/                       # Frontend demo
├── logs/                       # Training logs
│
├── app.py                      # Streamlit app
├── main.py                     # Main entry point
└── modal_*.py                  # Modal deployment scripts
```

## 🎯 Features

### Shape Guard System
The shape guard monitors and corrects plasma shape violations:
- **β_N (Normalized Beta)**: Plasma pressure constraint
- **q_min (Minimum Safety Factor)**: Stability constraint
- **q95 (Edge Safety Factor)**: Edge stability constraint

### Self-Fixing Mechanism
- Detects constraint violations in real-time
- Applies corrective actions to reduce severity
- Tracks and visualizes recovery progress

### Visualization Tools
1. **Interactive Terminal Mode**: Step-by-step constraint monitoring
2. **Web Interface**: Full-featured Streamlit dashboard
3. **Animated Trajectories**: Real-time 2D/3D shape evolution
4. **Comprehensive Plots**: Multi-panel analysis with severity tracking

## 🔧 Usage Examples

### Training SAC Agent
```bash
python train/sac_train.py
```

### Evaluating Agents
```bash
# Random agent
python eval/rand_eval.py

# SAC agent
python eval/sac_eval.py
```

### Running on Modal (Cloud)
```bash
modal run modal_run.py
```

## 📊 What You'll See

### Terminal Output
```
Step 1:
  Shape: β_N=2.345, q_min=1.234, q95=4.567
  Status: 🟢 SAFE
  Severity: 0.000
  Reward: 0.150 → 0.150 (no penalty)

Step 2:
  Shape: β_N=3.456, q_min=0.890, q95=5.678
  Status: 🔴 VIOLATION
  Severity: 2.345 ↑ (was 0.000)
  Reward: 0.120 → -0.125 (penalty: -0.245)

Step 3:
  Status: 🟠 SELF-FIXING!
  Severity: 1.800 ↓ (was 2.345)
  ⭐ Corrective action! Severity reduced
```

### Generated Visualizations
- `random_baseline.png` - Baseline performance without shape guard
- `shape_self_fixing_with_shape_guard.png` - Multi-panel analysis showing:
  - Shape parameters over time
  - Severity reduction timeline
  - 2D/3D trajectory plots
  - Recovery statistics

## ⚙️ Configuration

### Shape Guard Parameters
Adjust in your scripts:
```python
agent = RandomAgent(
    action_space=env.action_space,
    shape_penalty=0.1,        # Penalty coefficient (0.0 = off)
    damp_on_violation=True,   # Reduce action magnitude on violations
    damp_factor=0.5,          # Damping strength (0.0-1.0)
)
```

### Constraint Tuning
Edit `src/utils/shape_guard.py` to adjust:
- `BETA_N_MIN`, `BETA_N_MAX`
- `QMIN_MIN`
- `Q95_MIN`, `Q95_MAX`

## 🐛 Troubleshooting

**Rewards too negative?**
- Reduce `shape_penalty` (try 0.01 or 0.05)
- Check if initial state violates constraints

**No corrective actions shown?**
- Verify violations are decreasing in severity
- Check corrective logic in `agent.py`

**Streamlit app issues?**
- Ensure dependencies installed: `uv sync`
- Check that imports are working correctly

## 📚 Dependencies

Core requirements:
- `gymtorax>=0.1.1` - Fusion reactor environment
- `torax==1.0.3` - Physics simulation
- `matplotlib>=3.8.0` - Visualization
- `numpy>=1.26.0` - Numerical computing
- `streamlit>=1.28.0` - Web interface
- `ray>=2.51.1` - Distributed computing
- `modal>=1.2.1` - Cloud deployment

See `pyproject.toml` for complete list.

## 🎓 Research Context

This project explores multi-objective reinforcement learning for nuclear fusion plasma control, focusing on:
- Safe exploration with hard constraints
- Self-correcting control policies
- Real-time constraint satisfaction
- Interpretable safety mechanisms

## 📝 License

See project repository for license information.
