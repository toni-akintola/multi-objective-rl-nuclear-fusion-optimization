"""
Quick test script for surrogate model inference.

Usage:
    python test_surrogate_inference.py
"""

import torch
import torch.nn as nn
import numpy as np
import time


class PlasmaTransitionModel(nn.Module):
    """Surrogate model for plasma state transitions."""

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


def load_surrogate_model(
    model_path: str = "models/surrogate_best.pt", device: str = "cpu"
):
    """
    Load the surrogate model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to run on

    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict with normalization stats
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model
    model = PlasmaTransitionModel(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dims=checkpoint.get("hidden_dims", [512, 512, 256]),
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_transition(
    model, checkpoint, state: np.ndarray, action: np.ndarray, device: str = "cpu"
):
    """
    Predict next state and reward given current state and action.

    Args:
        model: Surrogate model
        checkpoint: Checkpoint dict with normalization stats
        state: Current state (state_dim,)
        action: Action to take (action_dim,)
        device: Device to run on

    Returns:
        next_state: Predicted next state (state_dim,)
        reward: Predicted reward (scalar)
    """
    with torch.no_grad():
        # Convert to tensors
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

        # Get normalization stats
        state_mean = checkpoint["state_mean"].to(device)
        state_std = checkpoint["state_std"].to(device)
        action_mean = checkpoint["action_mean"].to(device)
        action_std = checkpoint["action_std"].to(device)
        reward_mean = checkpoint["reward_mean"].to(device)
        reward_std = checkpoint["reward_std"].to(device)

        # Normalize inputs
        state_norm = (state_t - state_mean) / (state_std + 1e-8)
        action_norm = (action_t - action_mean) / (action_std + 1e-8)

        # Predict
        next_state_norm, reward_norm = model(state_norm, action_norm)

        # Denormalize outputs
        next_state = next_state_norm * (state_std + 1e-8) + state_mean
        reward = reward_norm * (reward_std + 1e-8) + reward_mean

        # Convert to numpy
        next_state = next_state.squeeze(0).cpu().numpy()
        reward = reward.squeeze().cpu().item()

    return next_state, reward


def main():
    """Test surrogate model inference."""
    print("=" * 60)
    print("Surrogate Model Inference Test")
    print("=" * 60)

    # Load model
    print("\n📦 Loading model...")
    model, checkpoint = load_surrogate_model()

    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]

    print(f"✓ Model loaded successfully!")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test 1: Single prediction with random input
    print("\n" + "=" * 60)
    print("Test 1: Single Random Prediction")
    print("=" * 60)

    state = np.random.randn(state_dim).astype(np.float32)
    action = np.random.randn(action_dim).astype(np.float32)

    print(f"\nInput:")
    print(f"  State shape: {state.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  State sample: [{state[0]:.4f}, {state[1]:.4f}, {state[2]:.4f}, ...]")
    print(
        f"  Action: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}, {action[3]:.4f}]"
    )

    start = time.time()
    next_state, reward = predict_transition(model, checkpoint, state, action)
    elapsed = time.time() - start

    print(f"\nOutput:")
    print(f"  Next state shape: {next_state.shape}")
    print(
        f"  Next state sample: [{next_state[0]:.4f}, {next_state[1]:.4f}, {next_state[2]:.4f}, ...]"
    )
    print(f"  Reward: {reward:.6f}")
    print(f"  Inference time: {elapsed*1000:.3f}ms")

    # Test 2: Batch predictions
    print("\n" + "=" * 60)
    print("Test 2: Batch Predictions")
    print("=" * 60)

    batch_sizes = [10, 100, 1000]

    for batch_size in batch_sizes:
        states = np.random.randn(batch_size, state_dim).astype(np.float32)
        actions = np.random.randn(batch_size, action_dim).astype(np.float32)

        # Convert to tensors for batch processing
        with torch.no_grad():
            states_t = torch.tensor(states, dtype=torch.float32)
            actions_t = torch.tensor(actions, dtype=torch.float32)

            # Normalize
            state_mean = checkpoint["state_mean"]
            state_std = checkpoint["state_std"]
            action_mean = checkpoint["action_mean"]
            action_std = checkpoint["action_std"]
            reward_mean = checkpoint["reward_mean"]
            reward_std = checkpoint["reward_std"]

            states_norm = (states_t - state_mean) / (state_std + 1e-8)
            actions_norm = (actions_t - action_mean) / (action_std + 1e-8)

            # Time batch prediction
            start = time.time()
            next_states_norm, rewards_norm = model(states_norm, actions_norm)
            elapsed = time.time() - start

            # Denormalize
            next_states = next_states_norm * (state_std + 1e-8) + state_mean
            rewards = rewards_norm * (reward_std + 1e-8) + reward_mean

        throughput = batch_size / elapsed
        per_pred = elapsed / batch_size * 1000

        print(f"\nBatch size: {batch_size}")
        print(f"  Total time: {elapsed*1000:.3f}ms")
        print(f"  Per prediction: {per_pred:.3f}ms")
        print(f"  Throughput: {throughput:.1f} predictions/sec")

    # Test 3: Rollout simulation
    print("\n" + "=" * 60)
    print("Test 3: Multi-Step Rollout")
    print("=" * 60)

    n_steps = 10
    state = np.random.randn(state_dim).astype(np.float32)

    print(f"\nSimulating {n_steps}-step rollout...")
    print(f"Initial state: [{state[0]:.4f}, {state[1]:.4f}, {state[2]:.4f}, ...]")

    total_reward = 0.0
    start = time.time()

    for step in range(n_steps):
        # Random action
        action = np.random.randn(action_dim).astype(np.float32) * 0.1

        # Predict next state
        state, reward = predict_transition(model, checkpoint, state, action)
        total_reward += reward

        if step < 3 or step == n_steps - 1:
            print(
                f"  Step {step+1}: reward={reward:.6f}, state=[{state[0]:.4f}, {state[1]:.4f}, ...]"
            )

    elapsed = time.time() - start

    print(f"\nRollout complete!")
    print(f"  Total reward: {total_reward:.6f}")
    print(f"  Average reward: {total_reward/n_steps:.6f}")
    print(f"  Total time: {elapsed*1000:.3f}ms")
    print(f"  Time per step: {elapsed/n_steps*1000:.3f}ms")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print("\n💡 Usage in your code:")
    print("```python")
    print(
        "from test_surrogate_inference import load_surrogate_model, predict_transition"
    )
    print("")
    print("# Load model once")
    print("model, checkpoint = load_surrogate_model('models/surrogate_best.pt')")
    print("")
    print("# Use for predictions")
    print("next_state, reward = predict_transition(model, checkpoint, state, action)")
    print("```")


if __name__ == "__main__":
    main()
