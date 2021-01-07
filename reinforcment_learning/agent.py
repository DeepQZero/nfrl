import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Agent:
    def __init__(self):
        self.net = Net()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

    def act(self, state, epsilon=0.2):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(200)
        else:
            return np.argmax([self.net(torch.from_numpy(np.concatenate((state, [i]))).float())
                              for i in range(200)])

    def train(self, state, action, score):
        self.optimizer.zero_grad()
        new_choice = torch.from_numpy(np.concatenate((state, [action]))).float()
        actual = self.net(new_choice)
        loss = self.criterion(actual, torch.Tensor([score]))
        loss.backward()
        self.optimizer.step()
