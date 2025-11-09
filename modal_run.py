import modal
import numpy as np

# Define a Modal Image with all dependencies and pre-warm JAX
def warmup_environment():
    """Pre-compile JAX and initialize environment to cache compilation"""
    import os
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16'
    
    import jax
    import jax.numpy as jnp
    import gymnasium as gym
    import gymtorax
    
    print("Warming up JAX and environment...")
    
    # Trigger JAX compilation with dummy operations
    x = jnp.array([1.0, 2.0, 3.0])
    _ = jnp.square(x)
    _ = jnp.sum(x)
    
    # Create and initialize environment once to trigger any lazy loading
    env = gym.make('gymtorax/IterHybrid-v0')
    obs, info = env.reset()
    action = env.action_space.sample()
    _ = env.step(action)
    env.close()
    
    print("Environment warmed up successfully!")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "jax[cuda12]",
        "gymnasium",
        "gymtorax",
        "numpy",
        "psutil",
        "stable-baselines3",
        "torch",
    )
    .run_function(warmup_environment, gpu="A100", cpu=16.0)
)

# 2) Attach the image
app = modal.App("example-custom-container", image=image)


@app.function(gpu="A100:1", cpu=16.0)
def check_cuda_version():
    """Check CUDA driver and runtime versions in the container"""
    import subprocess
    
    # Check CUDA driver version via nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("=== nvidia-smi output ===")
        print(result.stdout)
    except Exception as e:
        print(f"nvidia-smi error: {e}")
    
    # Check CUDA runtime version
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        print("\n=== nvcc --version output ===")
        print(result.stdout)
    except Exception as e:
        print(f"nvcc error: {e}")
    
    # Check via JAX
    try:
        import jax
        import jax.numpy as jnp
        print("\n=== JAX CUDA info ===")
        print(f"JAX version: {jax.__version__}")
        print(f"Devices: {jax.devices()}")
        print(f"Default backend: {jax.default_backend()}")
        x = 2
        print(f"The square of {x} is {jnp.square(x)}")
    except Exception as e:
        print(f"JAX error: {e}")


@app.function(gpu="A100", cpu=16.0, memory=16384, timeout=3600)
def run_vectorized_episodes(n_envs: int = 2, episodes_per_env: int = 2, seed: int = None):
    """Run multiple episodes using vectorized environments on one GPU
    
    Uses SubprocVecEnv with only 2 envs to stay within GPU memory limits.
    Each TORAX environment uses ~5-8GB GPU memory, so 2 envs = ~16GB < 40GB A100.
    """
    import os
    import time
    import psutil
    
    # Set JAX flags
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16'
    
    import gymnasium as gym
    import gymtorax
    import numpy as np
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    start_time = time.time()
    process = psutil.Process()
    
    print(f"Starting {n_envs} vectorized environments (SubprocVecEnv) - CPUs: {psutil.cpu_count()}, Memory: {psutil.virtual_memory().total / 1e9:.1f}GB")
    
    # Create vectorized environment
    def make_env(env_id: int):
        def _init():
            # Import inside subprocess to register gymtorax
            import gymnasium as gym
            import gymtorax
            
            env = gym.make('gymtorax/IterHybrid-v0')
            if seed is not None:
                env.reset(seed=seed + env_id)
            return env
        return _init
    
    # Use SubprocVecEnv for true parallelism
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method='forkserver')
    
    all_episode_rewards = []
    all_episode_lengths = []
    
    # Run episodes
    for episode_batch in range(episodes_per_env):
        obs = vec_env.reset()
        episode_rewards = np.zeros(n_envs)
        episode_lengths = np.zeros(n_envs, dtype=int)
        dones = np.zeros(n_envs, dtype=bool)
        
        while not dones.all():
            # Random actions for all environments
            actions = np.array([vec_env.action_space.sample() for _ in range(n_envs)])
            obs, rewards, new_dones, infos = vec_env.step(actions)
            
            # Update only non-done environments
            episode_rewards += rewards * (~dones)
            episode_lengths += (~dones).astype(int)
            dones = new_dones
        
        all_episode_rewards.extend(episode_rewards.tolist())
        all_episode_lengths.extend(episode_lengths.tolist())
        
        print(f"Batch {episode_batch + 1}/{episodes_per_env}: Mean reward={np.mean(episode_rewards):.4f}, Mean length={np.mean(episode_lengths):.1f}")
    
    vec_env.close()
    
    elapsed = time.time() - start_time
    mem_used = process.memory_info().rss / 1e9
    total_episodes = len(all_episode_rewards)
    
    print(f"\nCompleted {total_episodes} episodes in {elapsed:.2f}s ({elapsed/total_episodes:.2f}s per episode)")
    print(f"Memory used: {mem_used:.2f}GB")
    
    return {
        "episode_rewards": all_episode_rewards,
        "episode_lengths": all_episode_lengths,
        "total_time": elapsed,
        "time_per_episode": elapsed / total_episodes,
        "n_envs": n_envs,
        "episodes_per_env": episodes_per_env
    }


@app.function(gpu="A100", cpu=16.0, memory=16384, timeout=3600)
def run_single_episode(episode_id: int, seed: int = None):
    """Run a single episode on one GPU (non-vectorized baseline)"""
    import os
    import time
    import psutil
    
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16'
    
    import gymnasium as gym
    import gymtorax
    import numpy as np
    
    start_time = time.time()
    process = psutil.Process()
    
    print(f"Episode {episode_id} starting - Available CPUs: {psutil.cpu_count()}, Memory: {psutil.virtual_memory().total / 1e9:.1f}GB")
    
    env = gym.make('gymtorax/IterHybrid-v0')
    ep_seed = (seed + episode_id) if seed is not None else None
    observation, info = env.reset(seed=ep_seed)
    
    episode_reward = 0
    episode_length = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
    
    elapsed = time.time() - start_time
    mem_used = process.memory_info().rss / 1e9
    cpu_percent = process.cpu_percent()
    print(f"Episode {episode_id}: Reward={episode_reward:.4f}, Length={episode_length}, Time={elapsed:.2f}s, Memory={mem_used:.2f}GB, CPU={cpu_percent:.1f}%")
    
    return {
        "episode_id": episode_id,
        "reward": float(episode_reward),
        "length": int(episode_length),
        "time": elapsed
    }


@app.function(cpu=4.0, memory=4096, timeout=7200)
def run_random_search_vectorized(n_episodes: int = 10, n_containers: int = 2, seed: int = None):
    """Run random search using vectorized environments across multiple containers"""
    import numpy as np
    
    print(f"\n=== Starting Vectorized Random Search ===")
    print(f"Total episodes: {n_episodes}")
    print(f"Containers: {n_containers}")
    
    # Distribute episodes across containers
    episodes_per_container = n_episodes // n_containers
    n_envs_per_container = 2  # Run 2 parallel envs per container (GPU memory safe)
    episodes_per_env = max(1, episodes_per_container // n_envs_per_container)
    
    print(f"Envs per container: {n_envs_per_container}")
    print(f"Episodes per env: {episodes_per_env}")
    
    # Launch containers in parallel
    results = list(run_vectorized_episodes.map(
        [n_envs_per_container] * n_containers,
        [episodes_per_env] * n_containers,
        [seed + i * 1000 if seed else None for i in range(n_containers)]
    ))
    
    # Aggregate results from all containers
    episode_rewards = []
    episode_lengths = []
    total_times = []
    
    for r in results:
        episode_rewards.extend(r["episode_rewards"])
        episode_lengths.extend(r["episode_lengths"])
        total_times.append(r["total_time"])
    
    episode_rewards = episode_rewards[:n_episodes]  # Trim to exact count
    episode_lengths = episode_lengths[:n_episodes]
    episode_times = [t / len(results[0]["episode_rewards"]) for t in total_times]
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    print(f"\n=== Random Search Results ===")
    print(f"Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"Mean length: {mean_length:.1f} ± {std_length:.1f}")
    print(f"Mean episode time: {np.mean(episode_times):.2f}s ± {np.std(episode_times):.2f}s")
    print(f"Total wall time: {max(episode_times):.2f}s (parallel)")
    print(f"Episode rewards: {[f'{r:.4f}' for r in episode_rewards]}")
    print(f"Episode lengths: {episode_lengths}")
    
    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_length": float(mean_length),
        "std_length": float(std_length),
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_lengths": [int(l) for l in episode_lengths],
    }


# ========== SAC TRAINING ==========

from gymnasium.spaces import Dict as SpaceDict, Tuple as SpaceTuple, Box
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
import gymnasium as gym


class FlattenObsWrapper(gym.ObservationWrapper):
    """Flattens ANY observation space (nested Dict/Tuple/etc.) into a single Box"""
    def __init__(self, env):
        super().__init__(env)
        self._orig_obs_space = env.observation_space
        self.observation_space = flatten_space(self._orig_obs_space)

    def observation(self, obs):
        flattened = flatten(self._orig_obs_space, obs)
        # Check for NaN/Inf in observations
        if np.any(~np.isfinite(flattened)):
            print(f"WARNING: Non-finite values in observation, replacing with safe values")
            flattened = np.nan_to_num(flattened, nan=0.0, posinf=1e6, neginf=-1e6)
        return flattened


class FlattenActionWrapper(gym.ActionWrapper):
    """If the env has a Dict (or Tuple) action space, expose it as a flat Box for SB3"""
    def __init__(self, env):
        super().__init__(env)
        self._orig_action_space = env.action_space
        if isinstance(self._orig_action_space, (SpaceDict, SpaceTuple)):
            self.action_space = flatten_space(self._orig_action_space)
            self._needs_unflatten = True
        elif isinstance(self._orig_action_space, Box):
            self.action_space = self._orig_action_space
            self._needs_unflatten = False
        else:
            raise TypeError(f"Unsupported action space type: {type(self._orig_action_space)}")

    def action(self, action: np.ndarray):
        if self._needs_unflatten:
            return unflatten(self._orig_action_space, action)
        return action


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations using running statistics"""
    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.obs_var = np.ones(env.observation_space.shape, dtype=np.float32)
        self.obs_count = epsilon

    def observation(self, obs):
        # Update running statistics
        batch_mean = obs
        batch_var = np.zeros_like(obs)
        batch_count = 1

        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count

        self.obs_mean += delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.obs_count * batch_count / total_count
        self.obs_var = M2 / total_count
        self.obs_count = total_count

        # Normalize
        normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
        return np.clip(normalized, -10, 10)


def make_sac_env(normalize=True):
    """Create environment with wrappers for SAC training"""
    import gymnasium as gym
    import gymtorax
    
    env = gym.make("gymtorax/IterHybrid-v0")

    # Flatten observations first
    if isinstance(env.observation_space, (SpaceDict, SpaceTuple)) or not isinstance(env.observation_space, Box):
        env = FlattenObsWrapper(env)

    # Add normalization
    if normalize:
        env = NormalizeObservation(env)

    # Flatten actions
    if isinstance(env.action_space, (SpaceDict, SpaceTuple)) or not isinstance(env.action_space, Box):
        env = FlattenActionWrapper(env)

    assert isinstance(env.observation_space, Box), "Obs must be Box after wrapping"
    assert isinstance(env.action_space, Box), "Action must be Box after wrapping"
    return env


@app.function(gpu="A100:4", cpu=16.0, memory=32768, timeout=7200)
def train_sac(total_timesteps: int = 10000, learning_rate: float = 3e-4, buffer_size: int = 100000):
    """Train SAC agent on TORAX environment with extensive logging"""
    import os
    import time
    import psutil
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    import torch.nn as nn
    
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16'
    
    print("="*80)
    print("STARTING SAC TRAINING ON MODAL GPU")
    print("="*80)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Buffer size: {buffer_size}")
    print(f"Available CPUs: {psutil.cpu_count()}")
    print(f"Available Memory: {psutil.virtual_memory().total / 1e9:.1f}GB")
    print("="*80)
    
    # Create environment
    print("\n[1/5] Creating environment...")
    env = make_sac_env(normalize=True)
    print(f"✓ Observation space: {env.observation_space}")
    print(f"✓ Action space: {env.action_space}")
    
    # Test environment
    print("\n[2/5] Testing environment...")
    obs, info = env.reset()
    print(f"✓ Initial observation shape: {obs.shape}")
    print(f"✓ Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"✓ Contains NaN: {np.any(np.isnan(obs))}")
    print(f"✓ Contains Inf: {np.any(np.isinf(obs))}")
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ After step - Observation valid: {not np.any(np.isnan(obs))}")
    print(f"✓ After step - Reward: {reward:.4f}")
    
    # Custom callback for detailed logging
    class DetailedLoggingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_rewards = []
            self.episode_lengths = []
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.last_log_time = time.time()
            self.start_time = time.time()
            
        def _on_step(self) -> bool:
            self.current_episode_reward += self.locals['rewards'][0]
            self.current_episode_length += 1
            
            # Log episode completion
            if self.locals['dones'][0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                
                elapsed = time.time() - self.start_time
                print(f"\n{'='*60}")
                print(f"Episode {len(self.episode_rewards)} completed at timestep {self.num_timesteps}")
                print(f"  Reward: {self.current_episode_reward:.4f}")
                print(f"  Length: {self.current_episode_length}")
                print(f"  Elapsed time: {elapsed:.1f}s")
                print(f"  Timesteps/sec: {self.num_timesteps / elapsed:.1f}")
                
                if len(self.episode_rewards) >= 5:
                    recent_rewards = self.episode_rewards[-5:]
                    print(f"  Last 5 episodes avg reward: {np.mean(recent_rewards):.4f}")
                
                # Memory usage
                mem = psutil.virtual_memory()
                print(f"  Memory used: {mem.percent:.1f}% ({mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB)")
                print(f"{'='*60}")
                
                self.current_episode_reward = 0
                self.current_episode_length = 0
            
            # Periodic logging every 100 steps
            if self.num_timesteps % 100 == 0:
                current_time = time.time()
                if current_time - self.last_log_time >= 10:  # Log every 10 seconds
                    elapsed = current_time - self.start_time
                    print(f"[Timestep {self.num_timesteps}/{total_timesteps}] "
                          f"Elapsed: {elapsed:.1f}s | "
                          f"Episodes: {len(self.episode_rewards)} | "
                          f"Steps/sec: {self.num_timesteps / elapsed:.1f}")
                    self.last_log_time = current_time
            
            return True
    
    # Create SAC model
    print("\n[3/5] Creating SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        policy_kwargs=dict(
            net_arch=[512, 512],
            activation_fn=nn.Tanh
        )
    )
    print("✓ SAC model created")
    print(f"  Policy network: {model.policy}")
    
    # Train
    print("\n[4/5] Starting training...")
    print("="*80)
    
    callback = DetailedLoggingCallback()
    training_start = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            callback=callback
        )
        training_time = time.time() - training_start
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Total training time: {training_time:.1f}s")
        print(f"Total episodes: {len(callback.episode_rewards)}")
        print(f"Average reward: {np.mean(callback.episode_rewards):.4f} ± {np.std(callback.episode_rewards):.4f}")
        print(f"Average episode length: {np.mean(callback.episode_lengths):.1f}")
        print(f"Timesteps per second: {total_timesteps / training_time:.1f}")
        print("="*80)
        
        # Evaluate
        print("\n[5/5] Evaluating trained model...")
        from stable_baselines3.common.evaluation import evaluate_policy
        
        eval_env = make_sac_env(normalize=True)
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
        
        print(f"✓ Evaluation complete")
        print(f"  Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
        
        return {
            "success": True,
            "training_time": training_time,
            "total_episodes": len(callback.episode_rewards),
            "episode_rewards": [float(r) for r in callback.episode_rewards],
            "episode_lengths": [int(l) for l in callback.episode_lengths],
            "mean_reward": float(np.mean(callback.episode_rewards)),
            "std_reward": float(np.std(callback.episode_rewards)),
            "eval_mean_reward": float(mean_reward),
            "eval_std_reward": float(std_reward),
        }
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"TRAINING FAILED!")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


@app.local_entrypoint()
def test():
    # Compare vectorized vs non-vectorized approaches
    import time
    
    print("\n" + "="*60)
    print("VECTORIZED APPROACH (2 envs per container, 4 containers)")
    print("="*60)
    start = time.time()
    result_vec = run_random_search_vectorized.remote(n_episodes=16, n_containers=4, seed=42)
    vec_time = time.time() - start
    
    print(f"\n" + "="*60)
    print(f"Vectorized Results:")
    print(f"Mean reward: {result_vec['mean_reward']:.4f} ± {result_vec['std_reward']:.4f}")
    print(f"Wall time: {vec_time:.2f}s")
    print(f"Time per episode: {vec_time / 16:.2f}s")
    print(f"Speedup vs sequential: {(16 * 150) / vec_time:.1f}x")
    print(f"Parallelism: 4 containers × 2 envs = 8 parallel episodes")
    print("="*60)


@app.function(gpu="A100", cpu=16.0, memory=16384, timeout=1800)
def diagnose_gpu_usage():
    """Diagnose why GPU utilization is only 1% during training"""
    import os
    import time
    import subprocess
    
    # Configure JAX
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=16'
    os.environ['JAX_PLATFORMS'] = 'cuda'
    
    import jax
    import jax.numpy as jnp
    import gymnasium as gym
    import gymtorax
    import numpy as np
    
    print("="*80)
    print("GPU UTILIZATION DIAGNOSTIC")
    print("="*80)
    
    # 1. Check JAX configuration
    print("\n[1] JAX Configuration:")
    print(f"  JAX version: {jax.__version__}")
    print(f"  Devices: {jax.devices()}")
    print(f"  Default backend: {jax.default_backend()}")
    print(f"  Device count: {jax.device_count()}")
    
    # 2. Test pure JAX GPU computation
    print("\n[2] Testing pure JAX GPU computation...")
    start = time.time()
    for i in range(100):
        x = jnp.ones((1000, 1000))
        y = jnp.dot(x, x)
        y.block_until_ready()  # Force synchronization
    jax_time = time.time() - start
    print(f"  100 matrix multiplications: {jax_time:.3f}s")
    print(f"  Device: {y.device}")
    
    # 3. Create gymtorax environment and profile
    print("\n[3] Creating gymtorax environment...")
    env = gym.make('gymtorax/IterHybrid-v0')
    print(f"  Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # 4. Profile environment reset
    print("\n[4] Profiling environment reset...")
    start = time.time()
    obs, info = env.reset()
    reset_time = time.time() - start
    print(f"  Reset time: {reset_time:.3f}s")
    print(f"  Observation type: {type(obs)}")
    
    # 5. Profile environment steps
    print("\n[5] Profiling 100 environment steps...")
    step_times = []
    for i in range(100):
        action = env.action_space.sample()
        start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - start
        step_times.append(step_time)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print(f"  Mean step time: {np.mean(step_times)*1000:.2f}ms")
    print(f"  Median step time: {np.median(step_times)*1000:.2f}ms")
    print(f"  Min step time: {np.min(step_times)*1000:.2f}ms")
    print(f"  Max step time: {np.max(step_times)*1000:.2f}ms")
    print(f"  Steps per second: {1.0/np.mean(step_times):.1f}")
    
    # 6. Check if TORAX is using CPU or GPU
    print("\n[6] Checking TORAX computation backend...")
    print("  NOTE: gymtorax wraps TORAX, which is a JAX-based plasma simulator")
    print("  If GPU utilization is low, possible causes:")
    print("  a) TORAX may be doing CPU-bound preprocessing")
    print("  b) Small batch size - GPU underutilized for single episodes")
    print("  c) TORAX may not be JIT-compiled properly")
    print("  d) Data transfer overhead between CPU/GPU")
    
    # 7. Monitor GPU during computation
    print("\n[7] Running nvidia-smi during computation...")
    print("  Starting 10 second environment rollout...")
    
    # Start background nvidia-smi monitoring
    import threading
    
    def monitor_gpu():
        for _ in range(10):
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            print(f"    GPU: {result.stdout.strip()}")
            time.sleep(1)
    
    monitor_thread = threading.Thread(target=monitor_gpu)
    monitor_thread.start()
    
    # Run environment for 10 seconds
    start = time.time()
    steps = 0
    while time.time() - start < 10:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            obs, info = env.reset()
    
    monitor_thread.join()
    print(f"  Completed {steps} steps in 10 seconds ({steps/10:.1f} steps/sec)")
    
    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nLIKELY ISSUE: gymtorax/TORAX is CPU-bound or has small compute kernels")
    print("\nPOSSIBLE SOLUTIONS:")
    print("1. Use vectorized environments to batch GPU operations")
    print("2. Increase batch size in SAC training (more parallel episodes)")
    print("3. Check if TORAX has GPU-specific configuration options")
    print("4. Profile TORAX internals to find CPU bottlenecks")
    print("5. Consider if the physics simulation is inherently sequential")
    print("="*80)
    
    return {
        "jax_time": jax_time,
        "reset_time": reset_time,
        "mean_step_time": float(np.mean(step_times)),
        "steps_per_second": float(1.0/np.mean(step_times))
    }


@app.local_entrypoint()
def train():
    """Run SAC training on Modal"""
    print("Starting SAC training on Modal GPU...")
    result = train_sac.remote(total_timesteps=10000, learning_rate=3e-4, buffer_size=100000)
    
    if result["success"]:
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"Training time: {result['training_time']:.1f}s")
        print(f"Episodes completed: {result['total_episodes']}")
        print(f"Training mean reward: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
        print(f"Evaluation mean reward: {result['eval_mean_reward']:.4f} ± {result['eval_std_reward']:.4f}")
        print("="*80)
    else:
        print(f"\nTraining failed: {result['error']}")


@app.local_entrypoint()
def diagnose():
    """Run GPU utilization diagnostic"""
    print("Running GPU diagnostic...")
    result = diagnose_gpu_usage.remote()
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"JAX computation time (100 matmuls): {result['jax_time']:.3f}s")
    print(f"Environment reset time: {result['reset_time']:.3f}s")
    print(f"Mean step time: {result['mean_step_time']*1000:.2f}ms")
    print(f"Steps per second: {result['steps_per_second']:.1f}")
    print("="*80)
