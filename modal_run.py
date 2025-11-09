import modal

# 1) Define a Modal Image that includes NumPy
image = modal.Image.debian_slim().pip_install("jax[cuda12]", "gymnasium", "gymtorax")

# 2) Attach the image
app = modal.App("example-custom-container", image=image)



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


@app.function(gpu="A100:4")
def square(x=2):
    check_cuda_version()
    import gymnasium as gym
    import gymtorax

    # Create environment
    env = gym.make('gymtorax/IterHybrid-v0')

    # Reset and run with a random agent
    observation, info = env.reset()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward)
    
   
