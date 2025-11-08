import modal

app = modal.App("nuclear-fusion-rl")

N_GPUS = 1
# Create image with required dependencies and add local files
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]",
        "gymtorax>=0.1.1",
        "torax==1.0.3",
        "matplotlib>=3.8.0",
        "numpy>=1.26.0",
        "tqdm",
    )
    .add_local_file("run.py", remote_path="/root/run.py")
    .add_local_file("agent.py", remote_path="/root/agent.py")
    .add_local_file("main.py", remote_path="/root/main.py")
    .add_local_file("optimization-for-constraints/shape_guard.py", remote_path="/root/optimization-for-constraints/shape_guard.py")
)

@app.function(image=image, gpu="A100")
def train():
    import gymnasium as gym
    from run import run
    from agent import RandomAgent
    
    # Create environment to get action space
    env = gym.make("gymtorax/IterHybrid-v0")
    agent = RandomAgent(action_space=env.action_space)

    
    run(agent=agent, num_episodes=10, env=env)
    
def main():
    train.remote()
