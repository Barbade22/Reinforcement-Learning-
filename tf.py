import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def select_action(state, policy_net, epsilon, n_actions):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).argmax().item()
    else:
        return random.choice(range(n_actions))

def optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma):
    if len(memory) < batch_size:
        return
    
    transitions = random.sample(memory, batch_size)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0]).to(device)
    action_batch = torch.tensor(batch[1]).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch[2]).to(device)
    next_state_batch = torch.cat(batch[3]).to(device)
    done_batch = torch.tensor(batch[4], dtype=torch.float32).to(device)  # Convert boolean to float

    q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()

    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))
    
    loss = nn.functional.mse_loss(q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

env = gym.make("MountainCarContinuous-v0", render_mode="human")

states = env.observation_space.shape[0]
action_space = np.linspace(env.action_space.low, env.action_space.high, 10).reshape(-1, env.action_space.shape[0])
actions = len(action_space)
print(states, actions)

policy_net = DQN(states, actions).to(device)
target_net = DQN(states, actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = deque(maxlen=50000)

epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 500
gamma = 0.99
batch_size = 64
target_update = 10

steps_done = 0

num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor([np.array(state)], dtype=torch.float32).to(device)
    terminated, truncated = False, False
    score = 0

    while not (terminated or truncated):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * steps_done / epsilon_decay)
        action_idx = select_action(state, policy_net, epsilon, actions)
        action = action_space[action_idx]
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = torch.tensor([np.array(next_state)], dtype=torch.float32).to(device)
        done = terminated or truncated

        memory.append((state, action_idx, reward, next_state, done))
        state = next_state
        score += reward
        steps_done += 1

        optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma)
    
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Episode {episode} Score: {score}")

env.close()

# # Test the agent
# num_test_episodes = 10
# test_scores = []
# for episode in range(num_test_episodes):
#     state, _ = env.reset()
#     state = torch.tensor([np.array(state)], dtype=torch.float32).to(device)
#     terminated, truncated = False, False
#     score = 0

#     while not (terminated or truncated):
#         action_idx = select_action(state, policy_net, 0, actions)  # No exploration during testing
#         action = action_space[action_idx]
#         next_state, reward, terminated, truncated, info = env.step(action)
#         next_state = torch.tensor([np.array(next_state)], dtype=torch.float32).to(device)
#         state = next_state
#         score += reward
#         env.render()
    
#     test_scores.append(score)
#     print(f"Test Episode {episode} Score: {score}")

# print(f"Average test score: {np.mean(test_scores)}")

# env.close()
