import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium_2048.envs

# Model by https://github.com/bowass/2048-DQN-Agent/tree/main

class DQN_CNN(nn.Module):
    def __init__(self, input_channels=16, output_dim=4):
        super().__init__()
        # input channels, output channels, kernel_size
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=(1,2))
        self.conv2 = nn.Conv2d(input_channels, 128, kernel_size=(2,1))

        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,2))
        self.conv12 = nn.Conv2d(128, 128, kernel_size=(2,1))
        self.conv21 = nn.Conv2d(128, 128, kernel_size=(1,2))
        self.conv22 = nn.Conv2d(128, 128, kernel_size=(2,1))

        self.fc1 = nn.Linear(128*3*4*2+128*3*3*2+128*4*2*2, 256)
        self.out = nn.Linear(256, output_dim)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))

        x11 = F.relu(self.conv11(x1))
        x12 = F.relu(self.conv12(x1))
        x21 = F.relu(self.conv21(x2))
        x22 = F.relu(self.conv22(x2))

        x1_flat = x1.flatten(start_dim=1)
        x2_flat = x2.flatten(start_dim=1)
        x11_flat = x11.flatten(start_dim=1)
        x12_flat = x12.flatten(start_dim=1)
        x21_flat = x21.flatten(start_dim=1)
        x22_flat = x22.flatten(start_dim=1)

        x = torch.cat([x1_flat, x2_flat, x11_flat, x12_flat, x21_flat, x22_flat], dim=1)

        x = F.relu(self.fc1(x))
        return self.out(x)

# Utility functions

def find_high_tile(state):
    state = np.argmax(state, axis=2)
    state = 2 ** state * (state > 0)
    return np.max(state)

def preprocess(state): # 4, 4, 16
    state = torch.tensor(state)
    transposed = state.permute(2, 0, 1)   # (16, 4, 4)
    return transposed

def reward_shaping(state):
    state = np.argmax(state, axis=2)
    grid = 2 ** state * (state > 0)
    empty = np.count_nonzero(grid == 0)
    return empty
