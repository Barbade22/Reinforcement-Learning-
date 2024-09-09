import torch
import torch.nn as nn
import torch.nn.functional as F
device ="cuda"
class DQN(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim = 256):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__=='__main__':
    state_dim = 12
    action_dim = 2
    state = torch.randn(10,state_dim).to(device)
    net = DQN(state_dim,action_dim).to(device)
    action = net(state)
    print(action)
