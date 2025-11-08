import gymnasium as gym
import numpy as np
import gymtorax
import time

def benchmark_parallel_envs():
    print("\nParallel envs")
    start = time.time()
    envs = gym.make_vec("gymtorax/IterHybrid-v0", vectorization_mode="sync", num_envs=8)
    print(f"Make time: {time.time() - start}")
    _ = envs.reset(seed=42)
    print(f"Reset time: {time.time() - start}")
    _ = envs.step(envs.action_space.sample())
    print(f"Step time: {time.time() - start}")
    envs.close()

def benchmark_single_env():
    print("\nSingle env")
    start = time.time()
    env = gym.make("gymtorax/IterHybrid-v0")
    print(f"Make time: {time.time() - start}")
    _ = env.reset(seed=42)
    print(f"Reset time: {time.time() - start}")
    _ = env.step(env.action_space.sample())
    print(f"Step time: {time.time() - start}")
    env.close()


if __name__ == "__main__":
    benchmark_parallel_envs()

