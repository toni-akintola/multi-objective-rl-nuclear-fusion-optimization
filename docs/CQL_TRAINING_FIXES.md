# CQL Training Configuration Fixes

## Problem Identified (50k step run)

Looking at the training metrics from the 50k step run:

```
Epoch 1: actor_loss=95.4,  temp=1.17,  alpha=0.62
Epoch 5: actor_loss=257,   temp=3.97,  alpha=0.0165
```

**Issues:**
1. ❌ Actor loss **increased** from 95 → 257 (should decrease)
2. ❌ Temperature **grew** from 1.17 → 3.97 (making policy too stochastic)
3. ❌ Alpha dropped too quickly (0.62 → 0.0165)

This indicates the actor couldn't learn effectively due to:
- Learning rate too low (3e-5)
- Temperature growing unchecked
- Conservative penalty too aggressive (default 5.0)

## Configuration Changes

### Before (Problematic)
```python
actor_learning_rate=3e-5      # Too slow
critic_learning_rate=3e-4
temp_learning_rate=3e-5       # Too slow to control temp
initial_temperature=1.0       # Too high
conservative_weight=5.0       # Default, too aggressive
```

### After (Fixed)
```python
actor_learning_rate=1e-4      # 3.3x faster - actor can keep up with critic
critic_learning_rate=3e-4     # Unchanged
temp_learning_rate=1e-4       # 3.3x faster - better temp control
initial_temperature=0.1       # 10x lower - less stochastic initially
conservative_weight=1.0       # 5x lower - less penalty on actor
```

## Expected Improvements

With these changes, you should see:
- ✅ Actor loss **decreasing** over time
- ✅ Temperature staying **controlled** (< 2.0)
- ✅ Better balance between critic and actor learning
- ✅ More stable training overall

## Next Steps

1. **Run another 50k test**:
   ```bash
   modal run modal_run.py --n-steps 50000 --save-interval 10000
   ```

2. **Check metrics** - Look for:
   - Actor loss trending down
   - Temperature staying < 2.0
   - Conservative loss stabilizing

3. **If good, run full training**:
   ```bash
   modal run modal_run.py --n-steps 500000 --save-interval 10000
   ```

## Monitoring Tips

Watch these metrics during training:
- `actor_loss`: Should decrease over time
- `temp`: Should stabilize around 0.5-1.5
- `alpha`: CQL penalty weight, should stabilize
- `conservative_loss`: Should be negative but not too large
- `td_error`: Should decrease (better value estimates)
