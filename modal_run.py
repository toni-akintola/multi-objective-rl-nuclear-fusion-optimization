"""
Train CQL agent on Modal with GPU acceleration.

Run with:
    modal run modal_run.py
"""

import modal
from pathlib import Path
import d3rlpy

# Define Modal image with all dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "d3rlpy>=2.0.0",
    "torch>=2.0.0",
    "numpy>=1.26.0",
    "pandas>=2.0.0",
)

# Create Modal app
app = modal.App("cql-torax-training", image=image)

# Create volume for storing models and data
volume = modal.Volume.from_name("cql-models", create_if_missing=True)

# Mount path - single mount point for both data and models
VOLUME_DIR = "/cql-models"


@app.function(
    gpu="A100:2",  # Single A100 GPU
    cpu=16,  # 16 CPU cores for better data loading
    memory=32768,  # 32 GB RAM
    timeout=3600 * 4,  # 4 hour timeout
    volumes={VOLUME_DIR: volume},
)
def train_cql_remote(
    dataset_filename: str = "offline_dataset.pkl",
    n_steps: int = 500000,
    batch_size: int = 1024,  # Increased to 8192 for better GPU utilization
    save_interval: int = 50000,
):
    """
    Train CQL agent on GPU.

    Args:
        dataset_path: Path to local dataset file (will be uploaded)
        n_steps: Number of training steps (500k recommended for A100)
        batch_size: Batch size (256 is good for A100)
        save_interval: Save model every N steps
    """
    import pickle
    import torch
    from d3rlpy.algos import CQLConfig
    from d3rlpy.metrics import TDErrorEvaluator, AverageValueEstimationEvaluator
    from d3rlpy.dataset import MDPDataset
    from pathlib import Path

    print("=" * 60)
    print("CQL Training on Modal GPU")
    print("=" * 60)

    # Check GPU availability
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Load dataset from Modal volume (mounted at VOLUME_DIR)
    dataset_path = Path(VOLUME_DIR) / dataset_filename
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        dataset: MDPDataset = pickle.load(f)

    print(f"Dataset loaded successfully!")
    print(f"Total episodes: {len(dataset.episodes)}")

    # Use first 10% of episodes for evaluation metrics
    n_test_episodes = max(1, len(dataset.episodes) // 10)
    test_episodes = dataset.episodes[:n_test_episodes]
    print(f"Using {n_test_episodes} episodes for evaluation metrics")

    # Create CQL agent with GPU
    print("\nCreating CQL agent on GPU...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

    # Enable TF32 for faster training on A100
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for faster A100 training")

    # Add scalers to prevent numerical instability
    from d3rlpy.preprocessing import (
        StandardObservationScaler,
        MinMaxActionScaler,
        StandardRewardScaler,
    )

    # Use CQLConfig with proper scaling and stability constraints - d3rlpy 2.x API
    cql = CQLConfig(
        batch_size=batch_size,
        observation_scaler=StandardObservationScaler(),  # Normalize observations
        action_scaler=MinMaxActionScaler(),  # Scale actions to [-1, 1]
        reward_scaler=StandardRewardScaler(),  # Normalize rewards
        # Stability parameters to prevent divergence
        actor_learning_rate=3e-5,  # Slower actor updates (was 1e-4)
        critic_learning_rate=3e-4,  # Keep critic learning rate
        temp_learning_rate=3e-5,  # Slower temperature updates (was 1e-4)
        initial_temperature=1.0,  # Start with lower temperature
    ).create(device=device)

    print(f"CQL agent created on device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Training steps: {n_steps}")

    # Setup save directory
    save_dir = Path(VOLUME_DIR) / "models" / "cql_torax"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train the agent
    print(f"\nStarting training for {n_steps} steps...")
    print("=" * 60)

    try:
        # Train with d3rlpy 2.x API with checkpointing
        cql.fit(
            dataset,
            n_steps=n_steps,
            save_interval=save_interval,  # Save every N steps
            experiment_name=str(save_dir / "cql_experiment"),
            evaluators={
                "td_error": TDErrorEvaluator(test_episodes),
                "value_scale": AverageValueEstimationEvaluator(test_episodes),
            },
        )

        # Save final model
        final_model_path = save_dir / "cql_final.d3"
        print(f"\nSaving final model to {final_model_path}...")
        cql.save(final_model_path)

        # Also save a copy with timestamp for safety
        import time

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = save_dir / f"cql_backup_{timestamp}.d3"
        cql.save(backup_path)
        print(f"Backup saved to {backup_path}")

        # Commit volume to persist the model
        volume.commit()

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"Model saved to: {final_model_path}")
        print(f"To download: modal volume get cql-models {final_model_path}")

        return str(final_model_path)

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise


@app.local_entrypoint()
def main(
    dataset_filename: str = "offline_dataset.pkl",
    n_steps: int = 500000,
    batch_size: int = 256,  # Increased for A100
):
    """
    Local entrypoint that triggers remote training.

    Dataset should already be uploaded to Modal volume at: data/offline_dataset.pkl

    Usage:
        modal run modal_run.py --n-steps 500000

    To upload dataset first:
        modal volume put cql-models data/offline_dataset.pkl data/offline_dataset.pkl
    """
    print(f"Starting CQL training on Modal GPU...")
    print(f"\nTraining configuration:")
    print(f"  Dataset: data/{dataset_filename}")
    print(f"  Steps: {n_steps:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  GPU: 1x A100")
    print(f"  CPU cores: 16")
    print()

    # Train on remote GPU
    model_path = train_cql_remote.remote(
        dataset_filename=dataset_filename,
        n_steps=n_steps,
        batch_size=batch_size,
    )

    print(f"\nTraining completed! Model saved at: {model_path}")
    print(f"\nTo download the model:")
    print(f"  modal volume get cql-models models/cql_torax/cql_final.d3 ./logs/")
    print(f"\nTo list all files in volume:")
    print(f"  modal volume ls cql-models")
