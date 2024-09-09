import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential,Model

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


env = gym.make("MountainCarContinuous-v0", render_mode="human")

states = env.observation_space.shape[0]
actions = env.action_space.n
print(states,actions)

# episodes = 10
# l = []
# for episode in range(episodes):
#     state, _ = env.reset()
#     terminated, truncated = False, False
#     score = 0

#     while not (terminated or truncated):
#         action = random.choice([0, 1])
#         state, reward, terminated, truncated, info = env.step(action)
#         score += reward
#         env.render()
#     l.append(score)

#     print(f"Episode {episode} Score: {score}")
# print(sorted(l))

# env.close()

model = Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(24,activation="relu"))
model.add(Dense(24,activation="relu"))
model.add(Dense(actions,activation = "Linear"))

agent = DQNAgent(
    model=model,
    memory = SequentialMemory(limit=50000,window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions = actions,
    nb_steps_warmup=10,
    target_model_update=0.01

)

agent.compile(Adam(lr=0.001),metrics=["mae"])
agent.fit(env, nb_steps=100000,visualize=True,verbose=1)

results = agent.test(env,nb_episodes=10,visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()