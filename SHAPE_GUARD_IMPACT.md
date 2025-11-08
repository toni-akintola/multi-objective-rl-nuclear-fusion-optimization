# Shape Guard Impact Analysis

## Current State

**`shape_guard.py` is imported in `agent.py` but NOT currently active in `main.py`**

### Integration Flow

```
main.py
  └─> Creates Agent (RandomAgent)
       └─> Agent.__init__() sets up shape_penalty (default: 0.0 = OFF)
            └─> shape_guard.py imported via importlib
                 └─> shape_violation() function available
```

### What `shape_guard.py` Does

1. **Monitors Plasma Shape Parameters:**
   - `beta_N` (normalized beta): Must be between 0.5 and 3.0
   - `q_min` (minimum safety factor): Must be ≥ 1.0
   - `q95` (edge safety factor): Must be between 3.0 and 5.0

2. **Checks Two Safety Conditions:**
   - **Safe Box**: State parameters within acceptable ranges
   - **Smooth Transitions**: Limits how fast parameters can change per step:
     - `beta_N` change ≤ 0.2
     - `q_min` change ≤ 0.15
     - `q95` change ≤ 0.4

3. **Returns Violation Info:**
   ```python
   {
       "ok": bool,           # True if both safe box AND smooth
       "in_box": bool,       # Within safe parameter ranges
       "smooth": bool,       # No sudden jumps
       "severity": float,    # 0 = perfect, >0 = worse
       "shape": np.array     # [beta_N, q_min, q95]
   }
   ```

## Current Impact: **NONE** (Not Active)

In `main.py`, the shape safety methods are **never called**:
- ❌ `agent.reset_state()` is NOT called at `env.reset()`
- ❌ `agent.apply_shape_safety()` is NOT called after `env.step()`
- ❌ `shape_penalty=0.0` by default (turns off penalties even if called)

## How to Activate Shape Guard

### Option 1: Add Shape Penalty to Rewards

Modify `main.py` to call `apply_shape_safety()`:

```python
# After env.step()
observation, reward, terminated, truncated, info = env.step(action)
reward = agent.apply_shape_safety(reward, observation)  # Apply penalty
episode_reward += reward
```

### Option 2: Initialize Agent with Shape Penalty

```python
from agent import RandomAgent

agent = RandomAgent(
    action_space=env.action_space,
    shape_penalty=1.0,  # Penalty coefficient (higher = stricter)
    damp_on_violation=True,  # Reduce action magnitude on violations
    damp_factor=0.5  # Scale factor for actions
)
```

### Option 3: Reset State Tracking

```python
observation, info = env.reset()
agent.reset_state(observation)  # Initialize shape tracking
```

## Expected Impact When Activated

### With `shape_penalty > 0`:
- **Reward Modification**: Rewards are reduced when plasma shape violates safety constraints
- **Penalty Formula**: `penalty = shape_penalty * (1.0 + severity)`
- **Effect**: Agent learns to avoid unsafe plasma configurations

### With `damp_on_violation=True`:
- **Action Damping**: When shape violation detected, actions are scaled down by `damp_factor`
- **Effect**: Agent becomes more conservative after violations

## Example Integration

See `main.py` for a complete example with shape guard enabled.

