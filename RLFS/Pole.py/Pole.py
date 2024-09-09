import gym
import numpy as np
import os

env = gym.make("CartPole-v1", render_mode="human")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 15000
SHOW_EVERY = 1500

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # Exploration rate
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# File paths to save/load Q-table and epsilon
q_table_file = "q_table_cartpole.npy"
epsilon_file = "epsilon_cartpole.npy"

# Check if a saved Q-table exists
if os.path.isfile(q_table_file):
    q_table = np.load(q_table_file)
    print("Loaded Q-table from file.")
else:
    # Initialize Q-table with random values
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Check if a saved epsilon value exists
if os.path.isfile(epsilon_file):
    epsilon = np.load(epsilon_file)
    print("Loaded epsilon from file.")

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
    observation = env.reset()
    if isinstance(observation, tuple):
        state = observation[0]
    else:
        state = observation

    discrete_state = get_discrete_state(state)
    done = False
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        render = True
        print(f"Episode: {episode}")
    else:
        render = False

    while not done:
        if np.random.random() > epsilon:
            # Get action from Q-table
            action = np.argmax(q_table[discrete_state])
        else:
            # Take random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, info = env.step(action)
        if isinstance(new_state, tuple):
            state = new_state[0]
        else:
            state = new_state

        new_discrete_state = get_discrete_state(state)
        
        if render:
            env.render()

        episode_reward += reward

        if not terminated and not truncated:
            # Maximum possible Q value in the next step
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q value (for the current state and performed action)
            current_q = q_table[discrete_state + (action,)]
            # New Q value for the current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Update Q-table with new Q value
            q_table[discrete_state + (action,)] = new_q
        else:
            # Simulation ended - no specific goal to achieve in CartPole, just update Q value directly
            q_table[discrete_state + (action,)] = reward

        discrete_state = new_discrete_state

        # End the episode if terminated or truncated
        if terminated or truncated:
            done = True

    print(f"Episode: {episode}, Reward: {episode_reward}")

    # Decay epsilon
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

# Save Q-table and epsilon value at the end of training
np.save(q_table_file, q_table)
np.save(epsilon_file, epsilon)

env.close()
