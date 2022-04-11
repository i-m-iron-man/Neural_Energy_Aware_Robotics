import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

EPS = 0.003

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=64, hidden2=1024, hidden3=512, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        #self.fc4 = nn.Linear(hidden3, hidden4)
        self.fc5 = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        #self.fc4.weight.data = fanin_init(self.fc4.weight.data.size())
        self.fc5.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        out = self.fc1(state)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out,action],1))
        out = self.relu(out)
        out = self.fc3(out)
        #out = self.relu(out)
        #out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, action_lim, hidden1=1024, hidden2=512, hidden3=64, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.action_lim = action_lim
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.tanh(out)
        out = out * self.action_lim
        return out

