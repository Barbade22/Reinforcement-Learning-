import gym
import numpy as np
import os

env = gym.make("MountainCar-v0",render_mode = "human")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 1000

DISCRETE_OS_SIZE = [30] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


epsilon = 1 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table_file = "q_table30.npy"
epsilon_file = "epsilon30.npy"

if os.path.isfile(q_table_file):
    q_table = np.load(q_table_file)
    print("Loaded Q-table from file.")
else:
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

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
        # print(f"Episode: {episode}")
    else:
        render = False

    while not done:
        if np.random.random() > epsilon:

            action = np.argmax(q_table[discrete_state])
        else:
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

        if not terminated:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state

        if terminated:
            done = True

    if state[0] >= env.goal_position:
        episode_reward += 150 

    if episode % SHOW_EVERY ==0:

        print(f"Episode: {episode}, Reward: {episode_reward}")

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

np.save(q_table_file, q_table)
np.save(epsilon_file, epsilon)

env.close()
