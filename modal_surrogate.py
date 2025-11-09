"""
Train surrogate model on Modal GPU.

Usage:
    modal run modal_surrogate.py --epochs 50
"""

import modal
from pathlib import Path

# Define Modal image with dependencies
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.0.0", "numpy>=1.26.0", "tqdm>=4.65.0", "d3rlpy>=2.0.0"
)

# Create Modal app
app = modal.App("surrogate-training", image=image)

# Create volume for storing models
volume = modal.Volume.from_name("cql-models", create_if_missing=True)

# Mount path
VOLUME_DIR = "/cql-models"


@app.function(
    gpu="A100",  # Single A100 GPU
    cpu=4,
    memory=16384,  # 16 GB RAM
    timeout=3600,  # 1 hour timeout
    volumes={VOLUME_DIR: volume},
)
def train_surrogate_remote(
    dataset_filename: str = "offline_dataset.pkl",
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 1e-3,
):
    """
    Train surrogate model on GPU.

    Args:
        dataset_filename: Dataset filename in Modal volume
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    import torch
    import torch.nn as nn
    import pickle
    from pathlib import Path

    # Define model architecture
    class PlasmaTransitionModel(nn.Module):
        def __init__(self, state_dim=60, action_dim=4, hidden_dims=[512, 512, 256]):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim

            input_dim = state_dim + action_dim
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                    ]
                )
                prev_dim = hidden_dim

            self.backbone = nn.Sequential(*layers)
            self.state_head = nn.Linear(prev_dim, state_dim)
            self.reward_head = nn.Linear(prev_dim, 1)

        def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            features = self.backbone(x)
            next_state = self.state_head(features)
            reward = self.reward_head(features)
            return next_state, reward

    print("=" * 60)
    print("Surrogate Model Training on Modal GPU")
    print("=" * 60)

    # Check GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Load dataset
    dataset_path = Path(VOLUME_DIR) / dataset_filename
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Extract data from episodes (d3rlpy 2.x API)
    all_observations = []
    all_actions = []
    all_rewards = []

    for episode in dataset.episodes:
        all_observations.append(episode.observations)
        all_actions.append(episode.actions)
        all_rewards.append(episode.rewards)

    # Concatenate all episodes
    import numpy as np

    observations = np.concatenate(all_observations, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)

    # Create state transitions (s_t, a_t) -> (s_{t+1}, r_t)
    states = torch.tensor(observations[:-1], dtype=torch.float32)
    actions = torch.tensor(actions[:-1], dtype=torch.float32)
    next_states = torch.tensor(observations[1:], dtype=torch.float32)
    rewards = torch.tensor(rewards[:-1], dtype=torch.float32).unsqueeze(-1)

    print(f"Dataset loaded!")
    print(f"  Transitions: {len(states):,}")
    print(f"  State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")

    # Compute normalization stats
    state_mean = states.mean(dim=0)
    state_std = states.std(dim=0)
    action_mean = actions.mean(dim=0)
    action_std = actions.std(dim=0)
    reward_mean = rewards.mean()
    reward_std = rewards.std()

    print(f"\nNormalization stats:")
    print(f"  State mean: {state_mean.mean():.4f}, std: {state_std.mean():.4f}")
    print(f"  Action mean: {action_mean.mean():.4f}, std: {action_std.mean():.4f}")
    print(f"  Reward mean: {reward_mean:.4f}, std: {reward_std:.4f}")

    # Create model
    model = PlasmaTransitionModel(
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
    ).to(device)

    print(f"\nModel created on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("=" * 60)

    best_loss = float("inf")
    save_dir = Path(VOLUME_DIR) / "models" / "surrogate"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Shuffle data
        indices = torch.randperm(len(states))

        epoch_loss = 0
        epoch_state_loss = 0
        epoch_reward_loss = 0
        n_batches = 0

        model.train()

        for i in range(0, len(states), batch_size):
            batch_idx = indices[i : i + batch_size]

            # Get batch
            batch_states = states[batch_idx].to(device)
            batch_actions = actions[batch_idx].to(device)
            batch_next_states = next_states[batch_idx].to(device)
            batch_rewards = rewards[batch_idx].to(device)

            # Normalize
            batch_states_norm = (batch_states - state_mean.to(device)) / (
                state_std.to(device) + 1e-8
            )
            batch_actions_norm = (batch_actions - action_mean.to(device)) / (
                action_std.to(device) + 1e-8
            )
            batch_next_states_norm = (batch_next_states - state_mean.to(device)) / (
                state_std.to(device) + 1e-8
            )
            batch_rewards_norm = (batch_rewards - reward_mean.to(device)) / (
                reward_std.to(device) + 1e-8
            )

            # Forward pass
            pred_next_state, pred_reward = model(batch_states_norm, batch_actions_norm)

            # Compute losses
            state_loss = nn.MSELoss()(pred_next_state, batch_next_states_norm)
            reward_loss = nn.MSELoss()(pred_reward, batch_rewards_norm)

            # Total loss
            loss = state_loss + 0.1 * reward_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_state_loss += state_loss.item()
            epoch_reward_loss += reward_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_state_loss = epoch_state_loss / n_batches
        avg_reward_loss = epoch_reward_loss / n_batches

        scheduler.step(avg_loss)

        print(
            f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f} "
            f"(State: {avg_state_loss:.6f}, Reward: {avg_reward_loss:.6f})"
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"  → New best model! Saving...")

            model_path = save_dir / "surrogate_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "state_dim": states.shape[1],
                    "action_dim": actions.shape[1],
                    "hidden_dims": [512, 512, 256],
                    "state_mean": state_mean,
                    "state_std": state_std,
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "epoch": epoch,
                    "loss": avg_loss,
                },
                model_path,
            )

            # Commit to volume
            volume.commit()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Model saved to: {model_path}")
    print(f"\nTo download:")
    print(f"  modal volume get cql-models models/surrogate/surrogate_best.pt ./models/")

    return str(model_path)


@app.local_entrypoint()
def main(
    dataset_filename: str = "offline_dataset.pkl",
    epochs: int = 50,
    batch_size: int = 2048,
    lr: float = 1e-3,
):
    """
    Local entrypoint to trigger remote surrogate training.

    Usage:
        modal run modal_surrogate.py --epochs 50 --batch-size 2048
    """
    print("Starting surrogate model training on Modal GPU...")
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_filename}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  GPU: A100")
    print()

    # Train on remote GPU
    model_path = train_surrogate_remote.remote(
        dataset_filename=dataset_filename,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    print(f"\nTraining completed! Model saved at: {model_path}")
    print(f"\nNext steps:")
    print(
        f"  1. Download: modal volume get cql-models models/surrogate/surrogate_best.pt ./models/"
    )
    print(f"  2. Use for fast inference in demos (50-100x faster than TORAX!)")
