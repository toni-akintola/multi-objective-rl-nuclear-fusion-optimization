"""
Evaluate and test the surrogate model.

Usage:
    python eval/surrogate_eval.py --model-path models/surrogate_best.pt
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict
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


class SurrogateModel:
    """Wrapper for surrogate model with normalization and inference."""

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Load surrogate model from checkpoint.

        Args:
            model_path: Path to model checkpoint
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Extract model config
        self.state_dim = checkpoint["state_dim"]
        self.action_dim = checkpoint["action_dim"]
        hidden_dims = checkpoint.get("hidden_dims", [512, 512, 256])

        # Load normalization stats
        self.state_mean = checkpoint["state_mean"].to(device)
        self.state_std = checkpoint["state_std"].to(device)
        self.action_mean = checkpoint["action_mean"].to(device)
        self.action_std = checkpoint["action_std"].to(device)
        self.reward_mean = checkpoint["reward_mean"].to(device)
        self.reward_std = checkpoint["reward_std"].to(device)

        # Create and load model
        self.model = PlasmaTransitionModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"✓ Loaded surrogate model from {model_path}")
        print(f"  State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Device: {device}")

    def predict(
        self, state: np.ndarray, action: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Predict next state and reward.

        Args:
            state: Current state (state_dim,)
            action: Action to take (action_dim,)

        Returns:
            next_state: Predicted next state (state_dim,)
            reward: Predicted reward (scalar)
        """
        with torch.no_grad():
            # Convert to tensors
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            action_t = torch.tensor(action, dtype=torch.float32, device=self.device)

            # Add batch dimension if needed
            if state_t.dim() == 1:
                state_t = state_t.unsqueeze(0)
            if action_t.dim() == 1:
                action_t = action_t.unsqueeze(0)

            # Normalize
            state_norm = (state_t - self.state_mean) / (self.state_std + 1e-8)
            action_norm = (action_t - self.action_mean) / (self.action_std + 1e-8)

            # Predict
            next_state_norm, reward_norm = self.model(state_norm, action_norm)

            # Denormalize
            next_state = next_state_norm * (self.state_std + 1e-8) + self.state_mean
            reward = reward_norm * (self.reward_std + 1e-8) + self.reward_mean

            # Convert back to numpy
            next_state = next_state.squeeze(0).cpu().numpy()
            reward = reward.squeeze().cpu().item()

        return next_state, reward

    def predict_batch(
        self, states: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next states and rewards for a batch.

        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size, action_dim)

        Returns:
            next_states: Predicted next states (batch_size, state_dim)
            rewards: Predicted rewards (batch_size,)
        """
        with torch.no_grad():
            # Convert to tensors
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)

            # Normalize
            states_norm = (states_t - self.state_mean) / (self.state_std + 1e-8)
            actions_norm = (actions_t - self.action_mean) / (self.action_std + 1e-8)

            # Predict
            next_states_norm, rewards_norm = self.model(states_norm, actions_norm)

            # Denormalize
            next_states = next_states_norm * (self.state_std + 1e-8) + self.state_mean
            rewards = rewards_norm * (self.reward_std + 1e-8) + self.reward_mean

            # Convert back to numpy
            next_states = next_states.cpu().numpy()
            rewards = rewards.squeeze(-1).cpu().numpy()

        return next_states, rewards


def evaluate_model(
    model: SurrogateModel,
    dataset_path: str,
    test_fraction: float = 0.2,
    max_test_samples: int = 50000,
    eval_batch_size: int = 4096,
) -> Dict[str, float]:
    """
    Evaluate surrogate model on test data.

    Args:
        model: Surrogate model
        dataset_path: Path to dataset pickle file
        test_fraction: Fraction of data to use for testing
        max_test_samples: Maximum number of test samples to use
        eval_batch_size: Batch size for evaluation

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Evaluating Surrogate Model")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Extract data from episodes
    all_observations = []
    all_actions = []
    all_rewards = []

    for episode in dataset.episodes:
        all_observations.append(episode.observations)
        all_actions.append(episode.actions)
        all_rewards.append(episode.rewards)

    # Concatenate all episodes
    observations = np.concatenate(all_observations, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)

    # Create state transitions
    states = observations[:-1]
    actions = actions[:-1]
    next_states = observations[1:]
    rewards = rewards[:-1]

    # Split into train/test (limit size for memory efficiency)
    n_test = min(int(len(states) * test_fraction), max_test_samples)
    test_indices = np.random.choice(len(states), n_test, replace=False)

    test_states = states[test_indices]
    test_actions = actions[test_indices]
    test_next_states = next_states[test_indices]
    test_rewards = rewards[test_indices]

    print(f"Test set size: {n_test:,} transitions")

    # Predict on test set in batches for memory efficiency
    print(f"\nRunning predictions (batch size: {eval_batch_size})...")
    start_time = time.time()

    pred_next_states_list = []
    pred_rewards_list = []

    for i in range(0, n_test, eval_batch_size):
        batch_states = test_states[i : i + eval_batch_size]
        batch_actions = test_actions[i : i + eval_batch_size]

        batch_pred_next_states, batch_pred_rewards = model.predict_batch(
            batch_states, batch_actions
        )

        pred_next_states_list.append(batch_pred_next_states)
        pred_rewards_list.append(batch_pred_rewards)

    pred_next_states = np.concatenate(pred_next_states_list, axis=0)
    pred_rewards = np.concatenate(pred_rewards_list, axis=0)

    inference_time = time.time() - start_time

    # Compute metrics
    state_mse = np.mean((pred_next_states - test_next_states) ** 2)
    state_mae = np.mean(np.abs(pred_next_states - test_next_states))
    state_rmse = np.sqrt(state_mse)

    reward_mse = np.mean((pred_rewards - test_rewards) ** 2)
    reward_mae = np.mean(np.abs(pred_rewards - test_rewards))
    reward_rmse = np.sqrt(reward_mse)

    # Per-dimension state errors
    state_mse_per_dim = np.mean((pred_next_states - test_next_states) ** 2, axis=0)
    worst_dims = np.argsort(state_mse_per_dim)[-5:]

    # Compute R² scores
    state_r2 = 1 - (
        np.sum((test_next_states - pred_next_states) ** 2)
        / np.sum((test_next_states - test_next_states.mean(axis=0)) ** 2)
    )
    reward_r2 = 1 - (
        np.sum((test_rewards - pred_rewards) ** 2)
        / np.sum((test_rewards - test_rewards.mean()) ** 2)
    )

    metrics = {
        "state_mse": state_mse,
        "state_mae": state_mae,
        "state_rmse": state_rmse,
        "state_r2": state_r2,
        "reward_mse": reward_mse,
        "reward_mae": reward_mae,
        "reward_rmse": reward_rmse,
        "reward_r2": reward_r2,
        "inference_time": inference_time,
        "throughput": n_test / inference_time,
    }

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print("\n📊 State Prediction:")
    print(f"  MSE:  {state_mse:.6f}")
    print(f"  MAE:  {state_mae:.6f}")
    print(f"  RMSE: {state_rmse:.6f}")
    print(f"  R²:   {state_r2:.6f}")

    print("\n💰 Reward Prediction:")
    print(f"  MSE:  {reward_mse:.6f}")
    print(f"  MAE:  {reward_mae:.6f}")
    print(f"  RMSE: {reward_rmse:.6f}")
    print(f"  R²:   {reward_r2:.6f}")

    print("\n⚡ Performance:")
    print(f"  Inference time: {inference_time:.3f}s for {n_test:,} predictions")
    print(f"  Throughput: {metrics['throughput']:.1f} predictions/sec")
    print(f"  Per-prediction: {inference_time/n_test*1000:.3f}ms")

    print("\n🔍 Worst State Dimensions (by MSE):")
    for i, dim_idx in enumerate(worst_dims[::-1], 1):
        print(f"  {i}. Dimension {dim_idx}: MSE = {state_mse_per_dim[dim_idx]:.6f}")

    return metrics


def test_inference():
    """Test inference with random inputs."""
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)

    # Load model
    model_path = "models/surrogate_best.pt"
    model = SurrogateModel(model_path, device="cpu")

    # Create random test inputs
    state = np.random.randn(model.state_dim).astype(np.float32)
    action = np.random.randn(model.action_dim).astype(np.float32)

    print("\n🧪 Single Prediction Test:")
    print(f"  Input state shape: {state.shape}")
    print(f"  Input action shape: {action.shape}")

    # Time single prediction
    start = time.time()
    next_state, reward = model.predict(state, action)
    single_time = time.time() - start

    print(f"\n✓ Prediction successful!")
    print(f"  Output state shape: {next_state.shape}")
    print(f"  Output reward: {reward:.4f}")
    print(f"  Inference time: {single_time*1000:.3f}ms")

    # Test batch prediction
    batch_size = 1000
    states = np.random.randn(batch_size, model.state_dim).astype(np.float32)
    actions = np.random.randn(batch_size, model.action_dim).astype(np.float32)

    print(f"\n🧪 Batch Prediction Test (n={batch_size}):")
    start = time.time()
    next_states, rewards = model.predict_batch(states, actions)
    batch_time = time.time() - start

    print(f"\n✓ Batch prediction successful!")
    print(f"  Output states shape: {next_states.shape}")
    print(f"  Output rewards shape: {rewards.shape}")
    print(f"  Batch inference time: {batch_time*1000:.3f}ms")
    print(f"  Per-prediction: {batch_time/batch_size*1000:.3f}ms")
    print(f"  Throughput: {batch_size/batch_time:.1f} predictions/sec")

    return model


def main():
    """Main evaluation script."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate surrogate model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/surrogate_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/offline_dataset.pkl",
        help="Path to dataset",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=50000,
        help="Maximum number of test samples",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=4096, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run inference test, skip evaluation",
    )

    args = parser.parse_args()

    if args.test_only:
        # Just test inference
        test_inference()
    else:
        # Full evaluation
        model = SurrogateModel(args.model_path, device=args.device)
        metrics = evaluate_model(
            model,
            args.dataset_path,
            test_fraction=args.test_fraction,
            max_test_samples=args.max_test_samples,
            eval_batch_size=args.eval_batch_size,
        )

        # Also run inference test
        print("\n")
        test_inference()

        print("\n" + "=" * 60)
        print("✓ Evaluation Complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
