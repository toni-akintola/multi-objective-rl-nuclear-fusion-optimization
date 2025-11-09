# Surrogate Model Guide

## Overview

The surrogate model is a fast neural network approximation of the TORAX plasma simulator. It predicts next states and rewards 50-100x faster than running full TORAX simulations, making it ideal for:
- Real-time demos and visualizations
- Fast policy evaluation
- Model-based reinforcement learning
- Rapid prototyping

## Model Architecture

- **Type**: Feed-forward neural network
- **Input**: State (60 dims) + Action (4 dims)
- **Output**: Next state (60 dims) + Reward (1 dim)
- **Hidden layers**: [512, 512, 256] with LayerNorm, ReLU, and Dropout
- **Parameters**: 445,501
- **Training**: 50 epochs on 3.8M transitions from offline dataset

## Performance Metrics

### Inference Speed
- **Single prediction**: ~0.5ms
- **Batch (1000)**: ~0.003ms per prediction
- **Throughput**: ~400K predictions/sec (CPU)
- **10-step rollout**: ~1.6ms total

### Accuracy (on 10K test samples)
- **Reward prediction**:
  - MSE: 0.000134
  - MAE: 0.011116
  - RMSE: 0.011561
  
- **State prediction**:
  - Note: Large MSE values indicate the model may need better normalization for extreme state values
  - The model still provides useful relative predictions for policy evaluation

## Files

### Training
- `modal_surrogate.py` - Train surrogate model on Modal GPU
  - Uses A100 GPU for fast training
  - Saves model to Modal volume
  - ~5 minutes for 50 epochs

### Evaluation
- `eval/surrogate_eval.py` - Comprehensive evaluation script
  - Computes MSE, MAE, RMSE, R² metrics
  - Tests on held-out data
  - Measures inference speed
  - Memory-efficient batched evaluation

### Inference
- `test_surrogate_inference.py` - Quick inference testing
  - Load and use the model
  - Single and batch predictions
  - Multi-step rollout simulation
  - Performance benchmarks

### Model Files
- `models/surrogate_best.pt` - Trained model checkpoint
  - Contains model weights
  - Normalization statistics (mean/std for states, actions, rewards)
  - Model configuration

## Usage

### 1. Download Model (if not already done)
```bash
uv run modal volume get cql-models models/surrogate/surrogate_best.pt ./models/
```

### 2. Run Evaluation
```bash
# Full evaluation with 10K test samples
uv run python eval/surrogate_eval.py --dataset-path data/offline_dataset.pkl --max-test-samples 10000

# Quick inference test only
uv run python eval/surrogate_eval.py --test-only
```

### 3. Test Inference
```bash
uv run python test_surrogate_inference.py
```

### 4. Use in Your Code

```python
import torch
import torch.nn as nn
import numpy as np

# Load model
from test_surrogate_inference import load_surrogate_model, predict_transition

model, checkpoint = load_surrogate_model('models/surrogate_best.pt')

# Single prediction
state = np.random.randn(60).astype(np.float32)  # Your state
action = np.random.randn(4).astype(np.float32)  # Your action

next_state, reward = predict_transition(model, checkpoint, state, action)

print(f"Predicted reward: {reward:.6f}")
print(f"Next state shape: {next_state.shape}")
```

### 5. Batch Predictions (faster)

```python
# For multiple predictions at once
states = np.random.randn(100, 60).astype(np.float32)
actions = np.random.randn(100, 4).astype(np.float32)

with torch.no_grad():
    states_t = torch.tensor(states)
    actions_t = torch.tensor(actions)
    
    # Normalize
    state_mean = checkpoint['state_mean']
    state_std = checkpoint['state_std']
    action_mean = checkpoint['action_mean']
    action_std = checkpoint['action_std']
    reward_mean = checkpoint['reward_mean']
    reward_std = checkpoint['reward_std']
    
    states_norm = (states_t - state_mean) / (state_std + 1e-8)
    actions_norm = (actions_t - action_mean) / (action_std + 1e-8)
    
    # Predict
    next_states_norm, rewards_norm = model(states_norm, actions_norm)
    
    # Denormalize
    next_states = next_states_norm * (state_std + 1e-8) + state_mean
    rewards = rewards_norm * (reward_std + 1e-8) + reward_mean
    
    next_states = next_states.numpy()
    rewards = rewards.squeeze(-1).numpy()
```

## Integration with RL Training

The surrogate model can be used for:

1. **Model-based RL**: Use predicted transitions for planning
2. **Fast evaluation**: Quickly evaluate policies without running TORAX
3. **Warm-starting**: Initialize policies using surrogate rollouts
4. **Debugging**: Test policy behavior in fast simulation

Example integration:
```python
# In your RL training loop
if use_surrogate:
    next_state, reward = predict_transition(model, checkpoint, state, action)
else:
    next_state, reward = env.step(action)  # Real TORAX
```

## Retraining

To retrain with new data:

```bash
# 1. Upload new dataset to Modal
uv run modal volume put cql-models data/new_dataset.pkl offline_dataset.pkl

# 2. Train on Modal GPU
uv run modal run modal_surrogate.py --epochs 50 --batch-size 2048

# 3. Download new model
uv run modal volume get cql-models models/surrogate/surrogate_best.pt ./models/
```

## Tips

1. **Normalization is critical**: The model uses z-score normalization. Always use the stored mean/std from the checkpoint.

2. **Batch predictions are faster**: For multiple predictions, use batching (100-1000 samples) for best throughput.

3. **GPU acceleration**: For very large batches, load model on GPU:
   ```python
   model, checkpoint = load_surrogate_model('models/surrogate_best.pt', device='cuda')
   ```

4. **State prediction accuracy**: The model is most accurate for reward prediction. State predictions may have larger errors but are still useful for relative comparisons.

5. **Multi-step rollouts**: For long rollouts, errors accumulate. Best used for short-horizon planning (5-20 steps).

## Next Steps

- [ ] Use surrogate for fast policy visualization
- [ ] Integrate with CQL evaluation for faster testing
- [ ] Create model-based RL agent using surrogate
- [ ] Fine-tune on specific regions of state space
- [ ] Add uncertainty estimation for out-of-distribution detection
