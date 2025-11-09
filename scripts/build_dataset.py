"""
Build d3rlpy dataset from CSV files containing TORAX simulation data.

This script:
1. Loads all CSV files from sim_data/csv/
2. Extracts observations (all state columns)
3. Extracts actions (Ip, NBI, ECRH control parameters)
4. Computes rewards and terminal flags
5. Creates a d3rlpy MDPDataset for offline RL
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
import d3rlpy


def load_csv_files(data_dir: Path) -> List[pd.DataFrame]:
    """Load all CSV files from the data directory."""
    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(f"Loaded {csv_file.name}: {len(df)} rows, {len(df.columns)} columns")
        dataframes.append(df)

    return dataframes


def extract_observations(df: pd.DataFrame) -> np.ndarray:
    """
    Extract observations from the dataframe.
    Observations are all the state columns (plasma parameters).
    """
    # Exclude time column and action-related columns
    # Keep all physical state variables
    obs_columns = [col for col in df.columns if col != "time"]

    observations = df[obs_columns].values

    # Handle NaN and inf values
    observations = np.nan_to_num(observations, nan=0.0, posinf=1e6, neginf=-1e6)

    return observations


def extract_actions(df: pd.DataFrame) -> np.ndarray:
    """
    Extract actions from the dataframe.

    Actions control:
    - Plasma current (Ip): related to j_total, Ip_profile, j_external
    - Neutral Beam Injection (NBI): related to s_generic_particle
    - Electron Cyclotron Resonance Heating (ECRH): related to p_generic_heat_e

    We'll use the columns that represent control inputs:
    - j_external: external current drive (related to Ip control)
    - p_generic_heat_i: ion heating power (NBI-like)
    - p_generic_heat_e: electron heating power (ECRH)
    - s_generic_particle: particle source (NBI)
    """
    action_columns = [
        "j_external",  # Plasma current control
        "s_generic_particle",  # NBI particle injection
        "p_generic_heat_e",  # ECRH electron heating
        "p_generic_heat_i",  # NBI ion heating (optional, for 4D action space)
    ]

    # Check if all columns exist
    available_columns = [col for col in action_columns if col in df.columns]

    if len(available_columns) < 3:
        print(
            f"Warning: Only found {len(available_columns)} action columns: {available_columns}"
        )
        print("Using first 3-4 columns as proxy actions")
        # Fallback: use first few columns as proxy
        actions = df.iloc[:, 1:4].values  # Skip time column
    else:
        actions = df[available_columns].values

    # Handle NaN and inf values
    actions = np.nan_to_num(actions, nan=0.0, posinf=1e6, neginf=-1e6)

    return actions


def compute_rewards(df: pd.DataFrame) -> np.ndarray:
    """
    Compute rewards based on plasma performance metrics.

    Reward components:
    - Fusion gain (Q): ratio of fusion power to input power
    - H98: confinement quality factor
    - q_min: minimum safety factor (stability)
    - q_95: edge safety factor (stability)

    Note: This is a simplified version since we don't have access to the
    full state dict or reward module. We'll approximate using available columns.
    """
    weight_list = [1, 1, 1, 1]
    n_steps = len(df)
    rewards = np.zeros(n_steps)

    # Extract relevant columns
    T_e = df["T_e"].values if "T_e" in df.columns else np.zeros(n_steps)
    T_i = df["T_i"].values if "T_i" in df.columns else np.zeros(n_steps)
    q = df["q"].values if "q" in df.columns else np.ones(n_steps) * 2.0

    for i in range(n_steps):
        # Check H-mode condition: T_e[0] > 10 and T_i[0] > 10 keV
        is_h_mode = (T_e[i] > 10) and (T_i[i] > 10)

        # r_fusion_gain: Approximate fusion gain
        # In real implementation, this would use reward.get_fusion_gain(next_state)
        # For now, use temperature as proxy (fusion rate ~ T^2)
        fusion_gain_proxy = (T_e[i] * T_i[i]) / 100  # Normalize
        fusion_gain_proxy = fusion_gain_proxy / 10  # Normalize with ITER target
        r_fusion_gain = fusion_gain_proxy if is_h_mode else 0

        # r_h98: Confinement quality factor
        # In real implementation, this would use reward.get_h98(next_state)
        # For now, use pressure/temperature ratio as proxy
        h98_proxy = min(1.0, T_e[i] / 20)  # Simplified proxy
        r_h98 = h98_proxy if is_h_mode else 0

        # r_q_min: Minimum safety factor
        # In real implementation, this would use reward.get_q_min(next_state)
        q_min = q[i]  # Use q as proxy for q_min
        if q_min <= 1:
            r_q_min = q_min
        else:
            r_q_min = 1

        # r_q_95: Edge safety factor
        # In real implementation, this would use reward.get_q95(next_state)
        q_95 = q[i] * 1.5  # Approximate q95 as 1.5 * q_min
        if q_95 / 3 <= 1:
            r_q_95 = q_95 / 3
        else:
            r_q_95 = 1

        # Calculate total reward with weights
        r_fusion_gain_weighted = weight_list[0] * r_fusion_gain / 50
        r_h98_weighted = weight_list[1] * r_h98 / 50
        r_q_min_weighted = weight_list[2] * r_q_min / 150
        r_q_95_weighted = weight_list[3] * r_q_95 / 150

        rewards[i] = (
            r_fusion_gain_weighted + r_h98_weighted + r_q_min_weighted + r_q_95_weighted
        )

    # Handle NaN and inf
    rewards = np.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)

    return rewards


def compute_terminals(df: pd.DataFrame, episode_length: int = None) -> np.ndarray:
    """
    Compute terminal flags randomly.

    Terminals are randomly assigned to create variable-length episodes.
    """
    n_steps = len(df)
    terminals = np.zeros(n_steps, dtype=np.int32)

    # Generate random terminals
    # Use a probability to determine terminal states (e.g., 1% chance per step)
    terminal_probability = 0.01
    random_terminals = np.random.random(n_steps) < terminal_probability
    terminals[random_terminals] = 1

    # Always mark the last step as terminal
    terminals[-1] = 1

    return terminals


def build_dataset_from_csvs(
    data_dir: str = "sim_data/csv",
    episode_length: int = None,
    save_path: str = "data/offline_dataset.h5",
) -> d3rlpy.dataset.MDPDataset:
    """
    Build a d3rlpy MDPDataset from CSV files.

    Each CSV file represents a separate episode/simulation run.

    Args:
        data_dir: Directory containing CSV files
        episode_length: Length of each episode (None to auto-detect)
        save_path: Path to save the dataset

    Returns:
        d3rlpy.dataset.MDPDataset
    """
    data_dir = Path(data_dir)

    # Load all CSV files
    dataframes = load_csv_files(data_dir)

    if not dataframes:
        raise ValueError(f"No CSV files found in {data_dir}")

    # Process each episode separately
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []

    print("\nProcessing episodes...")
    for i, df in enumerate(dataframes):
        print(f"\nEpisode {i+1}/{len(dataframes)}: {len(df)} steps")

        # Extract components for this episode
        obs = extract_observations(df)
        acts = extract_actions(df)
        rews = compute_rewards(df)
        terms = np.zeros(len(df), dtype=np.int32)

        # Mark the last step of each episode as terminal
        terms[-1] = 1

        all_observations.append(obs)
        all_actions.append(acts)
        all_rewards.append(rews)
        all_terminals.append(terms)

        print(f"  Obs shape: {obs.shape}, Actions shape: {acts.shape}")
        print(
            f"  Reward: mean={rews.mean():.4f}, std={rews.std():.4f}, min={rews.min():.4f}, max={rews.max():.4f}"
        )

    # Concatenate all episodes
    print("\nConcatenating all episodes...")
    observations = np.concatenate(all_observations, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    terminals = np.concatenate(all_terminals, axis=0)

    print(f"\nTotal dataset statistics:")
    print(f"  Total steps: {len(observations)}")
    print(f"  Number of episodes: {len(dataframes)}")
    print(f"  Observations shape: {observations.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(
        f"  Overall reward stats: mean={rewards.mean():.4f}, std={rewards.std():.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}"
    )

    # Create d3rlpy dataset
    print("\nCreating d3rlpy MDPDataset...")
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    print(f"\nDataset created successfully!")

    # Save dataset
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving dataset to {save_path}...")
        # d3rlpy datasets can be saved using pickle or h5
        import pickle

        with open(save_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(dataset, f)
        print("Dataset saved!")

    return dataset


def main():
    """Main function to build and save the dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Build d3rlpy dataset from CSV files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="sim_data/csv",
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Length of each episode (None to auto-detect)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="data/offline_dataset.pkl",
        help="Path to save the dataset",
    )

    args = parser.parse_args()

    # Build dataset
    dataset = build_dataset_from_csvs(
        data_dir=args.data_dir,
        episode_length=args.episode_length,
        save_path=args.save_path,
    )

    print("\n" + "=" * 60)
    print("Dataset building complete!")
    print("=" * 60)
    print(f"\nTo load the dataset later:")
    print(f"  import pickle")
    print(f"  with open('{args.save_path}', 'rb') as f:")
    print(f"      dataset = pickle.load(f)")
    print("\nTo use with d3rlpy:")
    print(f"  from d3rlpy.algos import CQLConfig")
    print(f"  cql = CQLConfig().create()")
    print(f"  cql.fit(dataset, n_steps=100000)")


if __name__ == "__main__":
    main()
