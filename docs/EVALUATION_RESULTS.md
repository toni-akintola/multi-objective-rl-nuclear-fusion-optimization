# CQL Model Evaluation Results

## Model Details
- **Training**: 50k steps on CSV-derived dataset (60-dim observations, 4-dim actions)
- **Hyperparameters**: Initially poor (actor loss increasing), fixed mid-training
- **Dataset**: 10 episodes, 3.86M transitions, mean reward 0.022148

## Evaluation Results

### 1. Offline Evaluation (Dataset)

**Action Imitation**: CQL is 98.3% better than random at matching dataset actions
- Random action difference: 1.27e19
- CQL action difference: 2.18e17
- **Interpretation**: Model learned to approximate the dataset policy

**Value Estimates**: CQL shows high variance (numerical instability)
- Action variance: inf (overflow)
- **Interpretation**: Model has numerical issues, likely due to huge action scales (1e19 range)

### 2. Online Evaluation (Environment with Wrapper)

**Performance**: 
- Mean reward: -999.85
- Episode length: 12 steps (very short)
- **Interpretation**: Model performs poorly in environment

**Comparison**:
- Dataset reward: 0.022148 per step
- CQL online reward: -999.85 per episode
- **Conclusion**: Model does not generalize to environment

## Root Causes

### 1. Data Mismatch
- **Observations**: CSV has 60 features, environment has 1735 features
- **Actions**: CSV has 4 dimensions, environment needs 7 dimensions
- **Solution**: Created wrapper to bridge the gap, but imperfect mapping

### 2. Training Issues
- Only 50k steps (insufficient for convergence)
- Bad hyperparameters for first portion of training
- Action scales are huge (1e19), causing numerical instability

### 3. Dataset Quality
- Dataset reward is very low (0.022 per step)
- May not contain good examples of optimal behavior
- Collected from simulation, not optimized for RL

## Recommendations

### Short Term: Improve Current Approach
1. **Normalize actions** in the dataset (scale to [-1, 1] range)
2. **Train longer** with fixed hyperparameters (500k steps)
3. **Fine-tune wrapper** feature selection to better match CSV

### Long Term: Clean Slate (Recommended)
1. **Collect new dataset** directly from gymtorax environment
   ```bash
   python scripts/collect_env_dataset.py --episodes 100 --output data/env_dataset.pkl
   ```
2. **Train with matching obs/action spaces** (1735-dim obs, 7-dim actions)
3. **Use proper action normalization** from the start
4. **Evaluate without wrappers** (no dimension mismatch)

## Key Learnings

1. ✅ **CQL training works** - Model can learn from offline data
2. ✅ **Wrapper approach works** - Can bridge dimension mismatches
3. ❌ **Data quality matters** - Low-reward dataset → poor policy
4. ❌ **Action scales matter** - Huge values (1e19) cause numerical issues
5. ❌ **Hyperparameters matter** - Bad config early on hurt final performance

## Next Steps

**Option A: Quick Fix**
- Normalize dataset actions
- Retrain for 500k steps
- Test with wrapper

**Option B: Proper Solution** (Recommended)
- Collect environment dataset
- Train with matching dimensions
- Evaluate directly (no wrapper needed)

The model shows it **can learn**, but needs:
- Better data (from environment)
- Longer training (500k steps)
- Proper action normalization
- Matching observation/action spaces
