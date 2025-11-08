import gymnasium as gym
import numpy as np
import gymtorax

if __name__ == "__main__":


    envs = [gym.make("gymtorax/IterHybrid-v0") for _ in range(8)]
    obs = [env.reset()[0] for env in envs]

    actions = np.array([env.action_space.sample() for env in envs])

    results = [env.step(a) for env, a in zip(envs, actions)]
    print(results)
