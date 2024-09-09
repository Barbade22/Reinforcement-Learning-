import gym
import numpy as np
import os

env = gym.make("CartPole-v1", render_mode="human")

# File paths to load Q-table and epsilon
q_table_file = "q_table_cartpole.npy"
epsilon_file = "epsilon_cartpole.npy"

# Check if a saved Q-table exists
if os.path.isfile(q_table_file):
    q_table = np.load(q_table_file)
    print("Loaded Q-table from file.")
else:
    raise FileNotFoundError("No saved Q-table found.")

# Check if a saved epsilon value exists
if os.path.isfile(epsilon_file):
    epsilon = np.load(epsilon_file)
    print("Loaded epsilon from file.")
else:
    raise FileNotFoundError("No saved epsilon value found.")

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

# Play the game using the trained Q-table
for episode in range(10):  # Play 10 episodes for example
    observation = env.reset()
    if isinstance(observation, tuple):
        state = observation[0]
    else:
        state = observation

    discrete_state = get_discrete_state(state)
    done = False
    episode_reward = 0

    while not done:
        # Get action from Q-table
        action = np.argmax(q_table[discrete_state])

        new_state, reward, terminated, truncated, info = env.step(action)
        if isinstance(new_state, tuple):
            state = new_state[0]
        else:
            state = new_state

        new_discrete_state = get_discrete_state(state)
        
        env.render()

        episode_reward += reward
        discrete_state = new_discrete_state

        # End the episode if terminated or truncated
        if terminated or truncated:
            done = True

    print(f"Episode: {episode}, Reward: {episode_reward}")

env.close()
