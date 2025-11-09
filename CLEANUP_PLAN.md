# Repository Cleanup Plan

## Current Issues
1. **Root directory clutter** - Too many files in root
2. **Scattered documentation** - 9 different markdown files
3. **Mixed visualization files** - Multiple similar visualization scripts
4. **Unclear organization** - Hard to find what you need
5. **Generated files in repo** - PNG files should be in .gitignore

## Proposed Structure

```
nuclear/
├── docs/                          # All documentation
│   ├── README.md                  # Main readme (symlink to root)
│   ├── SETUP.md                   # Setup instructions
│   ├── TRAINING.md                # Training guides
│   ├── VISUALIZATION.md           # Visualization guides
│   └── CHANGELOG.md               # Change history
│
├── src/                           # Source code
│   ├── agents/                    # Agent implementations
│   │   └── agent.py
│   ├── environments/              # Custom environments
│   │   └── iter_hybrid_shape_guard_env.py
│   ├── visualization/             # All visualization scripts
│   │   ├── animate_shape.py
│   │   ├── visualize_chamber_live.py
│   │   ├── visualize_vertical_3d.py
│   │   └── visualize_vertical_live.py
│   ├── servers/                   # Web servers
│   │   ├── api_server.py
│   │   ├── chamber_web_server.py
│   │   └── chamber_websocket_server.py
│   └── utils/                     # Utilities
│       └── shape_guard.py (from optimization-for-constraints/)
│
├── scripts/                       # Executable scripts
│   ├── build_dataset.py
│   ├── launch_visualization.py
│   └── test_multi_env.py
│
├── train/                         # Training scripts (keep as is)
│   ├── offline_train.py
│   └── sac_train.py
│
├── eval/                          # Evaluation scripts (keep as is)
│   ├── cql_online_eval.py
│   ├── rand_eval.py
│   ├── sac_eval.py
│   └── surrogate_eval.py
│
├── data/                          # Data processing
│   ├── batch_convert_nc.py
│   └── nc2csv.py
│
├── models/                        # Saved models
│   └── surrogate_best.pt
│
├── demo/                          # Frontend demo (keep as is)
│
├── notebooks/                     # Jupyter notebooks (if any)
│
├── tests/                         # Test files
│   └── test_surrogate_inference.py
│
├── .github/                       # GitHub workflows (future)
│
├── app.py                         # Streamlit app (keep in root for easy access)
├── main.py                        # Main entry point
├── pyproject.toml                 # Dependencies
├── uv.lock                        # Lock file
├── .gitignore                     # Git ignore
└── README.md                      # Main readme

```

## Files to Move/Organize

### Documentation → docs/
- ANALYSIS_AND_RECOMMENDATIONS.md → docs/ANALYSIS.md
- CHAMBER_VISUALIZATION_SETUP.md → docs/VISUALIZATION.md (merge)
- GPU_TRAINING_GUIDE.md → docs/TRAINING.md (merge)
- HOW_TO_RUN_ANIMATION.md → docs/VISUALIZATION.md (merge)
- NEXT_STEPS.md → docs/ROADMAP.md
- README_STREAMLIT.md → docs/STREAMLIT.md
- SURROGATE_MODEL_GUIDE.md → docs/SURROGATE_MODEL.md
- CHANGELOG.md → docs/CHANGELOG.md

### Source Code → src/
- agent.py → src/agents/agent.py
- iter_hybrid_shape_guard_env.py → src/environments/
- animate_shape.py → src/visualization/
- visualize_*.py → src/visualization/
- api_server.py → src/servers/
- chamber_*.py → src/servers/
- optimization-for-constraints/shape_guard.py → src/utils/

### Scripts → scripts/
- build_dataset.py → scripts/
- launch_visualization.py → scripts/
- test_multi_env.py → scripts/

### Tests → tests/
- test_surrogate_inference.py → tests/

### Modal Scripts (keep in root or move to scripts/)
- modal_run.py
- modal_surrogate.py

### Files to Delete
- agent_baseline.png (generated output)
- shape_self_fixing_random_with_shape_guard.png (generated output)
- sac_shape_guard.py (duplicate/old?)

### Update .gitignore
- Add *.png to ignore generated plots
- Add notebooks/.ipynb_checkpoints/

## Benefits
1. ✅ Clear separation of concerns
2. ✅ Easy to navigate
3. ✅ Professional structure
4. ✅ Scalable for future growth
5. ✅ Follows Python best practices
