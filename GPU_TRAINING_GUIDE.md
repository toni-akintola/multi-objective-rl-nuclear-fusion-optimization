# GPU Training Guide for CQL on Modal

## Quick Start

```bash
# 1. Make sure you have the dataset
python build_dataset.py

# 2. Run training on Modal GPU
modal run modal_run.py

# 3. Download trained model
modal volume get cql-models /models/cql_torax/cql_final.d3 ./logs/
```

## Configuration

### Recommended Settings for A100 GPU

- **Training steps**: 500,000 (default)
  - Takes ~30-60 minutes on A100
  - Good balance of performance and cost
  
- **Batch size**: 256 (default)
  - Optimal for A100 memory
  - Can increase to 512 if needed

- **Resources**:
  - GPU: Single A100 (40GB VRAM)
  - CPU: 8 cores
  - RAM: 32 GB
  - Timeout: 4 hours

### Custom Training

```bash
# Train for 1M steps with larger batch
modal run modal_run.py --n-steps 1000000 --batch-size 512

# Quick test run (100k steps)
modal run modal_run.py --n-steps 100000
```

## Cost Estimate

**A100 GPU on Modal**: ~$1.10/hour

- 500k steps: ~$0.55-1.10 (30-60 min)
- 1M steps: ~$1.10-2.20 (1-2 hours)

## What Happens During Training

1. **Upload**: Dataset (~3GB) uploaded to Modal
2. **Setup**: Docker container built with dependencies
3. **Training**: CQL agent trains on GPU
   - Saves checkpoints every 50k steps
   - Tracks TD error for monitoring
4. **Save**: Final model saved to Modal volume
5. **Download**: Retrieve model to local machine

## Monitoring

Training logs show:
- `critic_loss`: Q-function learning progress
- `conservative_loss`: CQL penalty (should stabilize)
- `actor_loss`: Policy improvement
- `td_error`: Prediction accuracy (should decrease)

## Troubleshooting

**"Dataset not found"**
```bash
python build_dataset.py
```

**"Modal not authenticated"**
```bash
modal token new
```

**Training too slow**
- Increase batch size: `--batch-size 512`
- Reduce steps: `--n-steps 250000`

**Out of memory**
- Reduce batch size: `--batch-size 128`
- Use smaller GPU: Change `gpu="A100"` to `gpu="T4"`

## Next Steps

After training completes:
1. Download model: `modal volume get cql-models /models/cql_torax/cql_final.d3 ./logs/`
2. Evaluate in environment (create eval script)
3. Compare with baseline (random agent)
