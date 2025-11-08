import gymnasium as gym
import gymtorax

# Create environment
env = gym.make("gymtorax/IterHybrid-v0")

# Reset and run with a random agent
observation, info = env.reset()
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
