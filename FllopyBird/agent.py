import flappy_bird_gymnasium  # pip install flappy-bird-gymnasium
import gymnasium as gym
from dqn import DQN
import torch
import numpy as np


device = "cuda"
class Agent:
    def run(self, is_training = True,render = False):
        env = gym.make("CartPole", render_mode = "human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states,num_actions).to(device)

        obs,_ = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, _,info = env.step()


            if terminated:
                break
            
        env.close()


