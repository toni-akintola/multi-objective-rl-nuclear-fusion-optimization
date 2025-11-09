# TORAX Data Analysis & Training Recommendations

## Current Data Status

### ✅ What We Have: Raw TORAX Simulation Data
- **Location**: `./sim_data/csv/` (10 files)
- **Format**: NetCDF → CSV conversions
- **Content**: Physics simulation outputs (temperature, density, safety factor, etc.)
- **Size**: 3.86M data points, 1.88 GB total
- **Structure**: 
  - 22 timesteps per simulation
  - 17,550 spatial points per timestep
  - 61 physics variables per point

### ❌ What We're Missing: RL Training Data
- **Observations**: State vectors from gym environment
- **Actions**: Control inputs (heating power, current drive, etc.)
- **Rewards**: Performance metrics
- **Transitions**: (s_t, a_t, r_t, s_{t+1}) tuples

## Key Findings

### Physics Variables (from sim_data)
```
Ion Temperature (T_i):     1.00 - 17.82 keV  (mean: 9.10 keV)
Electron Temperature (T_e): 1.00 - 23.17 keV  (mean: 10.93 keV)
Safety Factor (q):         1.17 - 3.49       (mean: 1.80)
Density (n_e):             8e19 - 1.2e20 m^-3
Simulation Duration:       21 seconds each
```

### Critical Insight
The CSV data is **raw TORAX output**, not **RL trajectory data**. We need to run the data collection script to get:
- Gym environment observations
- Agent actions
- Rewards
- Episode structure

---

## Recommended Path Forward

### Phase 1: Collect RL Training Data (TODAY)

**Step 1: Test data collection with 1 episode**
```bash
python data_collection/collect_torax_data.py \
    --policy random \
    --num-episodes 1 \
    --seed 42
```

**Expected output**: `data/torax_random_1eps_seed42.h5`

**Step 2: Collect small training dataset (30 min)**
```bash
python data_collection/collect_torax_data.py \
    --policy random \
    --num-episodes 100 \
    --num-workers 8 \
    --checkpoint-every 25 \
    --seed 42
```

**Step 3: Analyze collected RL data**
```python
import h5py
import numpy as np

with h5py.File('data/torax_random_100eps_seed42.h5', 'r') as f:
    print("Keys:", list(f.keys()))
    print("Observations shape:", f['observations'].shape)
    print("Actions shape:", f['actions'].shape)
    print("Rewards:", f['rewards'][:100])
```

---

### Phase 2: Build Offline RL Training Pipeline (2-3 DAYS)

#### Option A: Conservative Q-Learning (CQL) - RECOMMENDED
**Best for**: Limited data, avoiding out-of-distribution actions

```python
# Pseudocode structure
class CQLTrainer:
    def __init__(self, dataset, obs_dim, action_dim):
        self.q_network = QNetwork(obs_dim, action_dim)
        self.policy = GaussianPolicy(obs_dim, action_dim)
        
    def train_step(self, batch):
        # Standard Q-learning loss
        q_loss = td_error(batch)
        
        # CQL penalty: penalize Q-values for unseen actions
        cql_loss = log_sum_exp(Q(s, a_random)) - Q(s, a_data)
        
        total_loss = q_loss + alpha * cql_loss
        return total_loss
```

**Implementation**: Use `d3rlpy` library (easiest) or `stable-baselines3-contrib`

#### Option B: Implicit Q-Learning (IQL)
**Best for**: Avoiding distributional shift, simpler than CQL

**Implementation**: Also available in `d3rlpy`

---

### Phase 3: Training Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE RL PIPELINE                      │
└─────────────────────────────────────────────────────────────┘

1. DATA COLLECTION (collect_torax_data.py)
   ├─ Random Policy: 500 episodes
   ├─ SAC Policy: 500 episodes (if available)
   └─ Output: data/torax_mixed_1000eps.h5
        ├─ observations: (N, obs_dim)
        ├─ actions: (N, action_dim)
        ├─ rewards: (N,)
        └─ next_observations: (N, obs_dim)

2. PREPROCESSING (train/preprocess_data.py)
   ├─ Normalize observations & actions
   ├─ Compute returns & advantages
   ├─ Train/val split (90/10)
   └─ Output: data/torax_normalized.h5

3. TRAINING (train/offline_rl_train.py)
   ├─ Algorithm: CQL or IQL
   ├─ Network: MLP (256-256-256)
   ├─ Batch size: 256
   ├─ Epochs: 100-500
   └─ Output: logs/cql_model.zip

4. EVALUATION (eval/offline_eval.py)
   ├─ Load trained policy
   ├─ Run on real TORAX env
   ├─ Compare to random baseline
   └─ Track shape violations
```

---

## Concrete Next Steps (Priority Order)

### 🔴 CRITICAL: Test Data Collection (5 min)
```bash
# Make sure the data collection script works
python data_collection/collect_torax_data.py \
    --policy random \
    --num-episodes 1 \
    --seed 42
```

### 🟡 HIGH: Collect Training Data (30 min - 4 hours)
```bash
# Small dataset for testing
python data_collection/collect_torax_data.py \
    --policy random \
    --num-episodes 100 \
    --num-workers 8 \
    --checkpoint-every 25

# Full dataset for training (if small works)
python data_collection/collect_torax_data.py \
    --policy random \
    --num-episodes 1000 \
    --num-workers 8 \
    --checkpoint-every 100
```

### 🟢 MEDIUM: Build Training Script (1-2 days)

**File**: `train/offline_rl_train.py`

**Key components**:
1. Data loader for HDF5 files
2. CQL/IQL implementation (use `d3rlpy`)
3. Training loop with checkpointing
4. Tensorboard logging
5. Validation evaluation

### 🔵 LOW: Build Evaluation Script (1 day)

**File**: `eval/offline_eval.py`

**Key components**:
1. Load trained policy
2. Run episodes on TORAX env
3. Compute metrics (return, shape violations, etc.)
4. Visualization

---

## Recommended Libraries

### For Offline RL:
1. **d3rlpy** (EASIEST) ⭐
   - Pre-built CQL, IQL, BCQ implementations
   - Works with HDF5 data
   - Good documentation
   ```bash
   pip install d3rlpy
   ```

2. **stable-baselines3-contrib**
   - Has offline RL algorithms
   - Integrates with SB3 ecosystem
   ```bash
   pip install sb3-contrib
   ```

3. **CORL** (if you want state-of-the-art)
   - Latest offline RL algorithms
   - More complex setup

---

## Expected Performance Metrics

### Baseline (Random Policy)
- Mean return: ~-500 to 0 (depends on reward function)
- Shape violation rate: 40-60%
- Episode length: ~150 steps

### Target (Trained CQL)
- Mean return: +200 to +500 (2-3x improvement)
- Shape violation rate: <20%
- Episode length: Full episodes (no early termination)

### Stretch Goal (Trained + Fine-tuned)
- Mean return: +500 to +1000
- Shape violation rate: <10%
- Stable plasma control

---

## Timeline Estimate

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1 | Test data collection (1 episode) | 5 min | ⏳ NEXT |
| 1 | Collect 100 episodes | 30 min | ⏳ |
| 1 | Analyze collected data | 15 min | ⏳ |
| 2 | Build CQL training script | 1 day | ⏳ |
| 2 | Train on 100 episodes | 1 hour | ⏳ |
| 2 | Evaluate trained model | 30 min | ⏳ |
| 3 | Collect 1000 episodes | 4 hours | ⏳ |
| 3 | Train on full dataset | 4 hours | ⏳ |
| 3 | Final evaluation | 1 hour | ⏳ |
| **TOTAL** | | **2-3 days** | |

---

## Questions to Answer Before Training

1. **What is the observation space structure?**
   - Run 1 episode and inspect the HDF5 file
   - Determine obs_dim

2. **What is the action space structure?**
   - Check action_dim from collected data
   - Understand action bounds

3. **What is the reward function?**
   - Inspect rewards from collected data
   - Understand reward scale and distribution

4. **Do we have shape constraint data?**
   - Check if shape_violations are tracked
   - Use for evaluation metrics

---

## Decision Point

**BEFORE BUILDING TRAINING PIPELINE:**

Run this command and share the output:
```bash
python data_collection/collect_torax_data.py \
    --policy random \
    --num-episodes 1 \
    --seed 42

# Then inspect:
python -c "
import h5py
with h5py.File('data/torax_random_1eps_seed42.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    for key in f.keys():
        print(f'{key}: {f[key].shape}, dtype={f[key].dtype}')
"
```

This will tell us:
- Exact data structure
- Observation/action dimensions
- Reward distribution
- Whether we're ready to build the training pipeline

---

## Summary

**Current Status**: ❌ No RL training data collected yet  
**Next Action**: ✅ Run data collection script (1 episode test)  
**Goal**: Build offline RL pipeline in 2-3 days  
**Approach**: CQL/IQL with d3rlpy library
