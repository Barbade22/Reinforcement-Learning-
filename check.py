import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Create the CartPole environment
env = gym.make("CartPole-v1")
nb_actions = env.action_space.n

# Build a simple neural network model
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

# Get the number of actions and states
states = env.observation_space.shape[0]
actions = env.action_space.n

# Create the model
model = build_model(states, actions)
print(model.summary())

# Configure and compile the agent
policy = BoltzmannQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)

# Test the agent
dqn.test(env, nb_episodes=10, visualize=True)
