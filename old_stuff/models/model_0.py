import numpy as np  # for matrix and numerical operations
import random  # for epsilon-greedy random action selection
import torch  # PyTorch core library
import torch.nn as nn  # for building neural networks
import torch.optim as optim  # for optimizer (e.g. Adam)
import torch.nn.functional as F  # for activation functions like ReLU
from collections import deque  # not used here, but useful for experience buffers
import gymnasium as gym  # for interacting with RL environments
import gymnasium_2048.envs  # registers the 2048 environment
import time  # for timing (optional)


class DQN(nn.Module):
    def __init__(self, input_dim=16, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # first hidden layer: 16 -> 128
        self.fc2 = nn.Linear(128, 128)  # second hidden layer: 128 -> 128
        self.out = nn.Linear(128, output_dim)  # output layer: 128 -> 4 (actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # apply ReLU to first hidden layer
        x = F.relu(self.fc2(x))  # apply ReLU to second hidden layer
        return self.out(x)  # raw Q-values (no activation, used directly)



class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity  # max buffer size
        self.buffer = []  # stores experiences (s, a, r, s', done)
        self.pos = 0  # index to insert the next experience
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # priority scores
        self.alpha = alpha  # how much prioritization to use (0 = uniform)

    def add(self, transition, td_error):
        max_prio = max(self.priorities.max(), td_error + 1e-6)  # avoid zero prob
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)  # append new experience
        else:
            self.buffer[self.pos] = transition  # overwrite oldest
        self.priorities[self.pos] = max_prio  # assign priority
        self.pos = (self.pos + 1) % self.capacity  # circular buffer

    def sample(self, batch_size, beta=0.4):
        prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[:self.pos]
        probs = prios ** self.alpha  # turn priorities into sampling probs
        probs /= probs.sum()  # normalize to sum to 1

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)  # sample
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)  # importance-sampling weights
        weights /= weights.max()  # normalize weights to 1 max
        weights = torch.tensor(weights, dtype=torch.float32)

        return samples, weights, indices  # return data and metadata

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err.item()) + 1e-6  # update priorities post-training



def preprocess(state):
    state = np.argmax(state, axis=2)  # extract log2 tile values from one-hot
    state = 2 ** state * (state > 0)  # convert log2 back to actual tile values
    result = np.log2(state + 1).flatten() / 16.0  # normalize and flatten
    return result

def find_high_tile(state):
    state = np.argmax(state, axis=2)  # extract actual tiles
    state = 2 ** state * (state > 0)
    return np.max(state)  # return highest tile seen

def is_strict_decreasing_powers(arr):
    arr = [x for x in arr if x != 0]
    return all(x >= y for x, y in zip(arr, arr[1:]))

def compute_monotonicity(grid):     # reward monotonic rows/cols, weighted by tile value in row/col
    grid = np.argmax(grid, axis=2)
    grid = 2 ** grid * (grid > 0)
    #     GRID
    # [[0 0 0 4]
    #  [0 0 4 0]
    #  [0 0 0 0]
    #  [0 2 8 2]]

    bonus = 0.0

    # Check rows
    for row in grid:
        max_val = max(row)
        log_scale = np.log2(max_val) if max_val > 0 else 0
        if is_strict_decreasing_powers(row) or is_strict_decreasing_powers(row[::-1]):
            bonus += 0.1 * log_scale

    # Check columns
    for col in zip(*grid):
        max_val = max(col)
        log_scale = np.log2(max_val) if max_val > 0 else 0
        if is_strict_decreasing_powers(col) or is_strict_decreasing_powers(col[::-1]):
            bonus += 0.1 * log_scale

    return bonus

def reward_max_tile_in_corner(grid, bonus=1.0):
    grid = np.argmax(grid, axis=2)  # shape (4, 4), values are 0–15
    powers = 2 ** grid * (grid > 0)  # recover tile values
    max_val = np.max(powers)

    corners = [powers[0, 0], powers[0, -1], powers[-1, 0], powers[-1, -1]]
    if max_val in corners:
        return bonus
    return 0.0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")  # initialize environment

    main_net = DQN().to(device)  # main Q-network
    target_net = DQN().to(device)  # target Q-network (used for stability)
    target_net.load_state_dict(main_net.state_dict())  # sync weights
    optimizer = optim.Adam(main_net.parameters(), lr=1e-4)
    buffer = PrioritizedReplayBuffer(10000)  # initialize replay buffer

    batch_size = 64
    gamma = 0.99  # discount factor
    epsilon = 1.0  # exploration rate
    epsilon_decay = 0.995
    epsilon_min = 0.1
    target_update_freq = 10  # how often to sync target_net
    num_episodes = 500
    total_steps = 0

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        high_tile = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)  # explore
            else:
                state_tensor = torch.tensor([preprocess(state)], dtype=torch.float32).to(device)
                with torch.no_grad():
                    action = main_net(state_tensor).argmax().item()  # exploit

            next_state, reward, terminated, truncated, _ = env.step(action)

            if np.array_equal(state, next_state):
                shaped_reward = reward - 2.0  # base reward minus penalty
            else:
                bonus = compute_monotonicity(next_state) + reward_max_tile_in_corner(next_state)
                shaped_reward = reward + bonus  # reward + layout bonus
            done = terminated or truncated

            with torch.no_grad(): # estimate priority (temporal-difference)
                s_tensor = torch.tensor([preprocess(state)], dtype=torch.float32).to(device)
                ns_tensor = torch.tensor([preprocess(next_state)], dtype=torch.float32).to(device)

                q_sa = main_net(s_tensor)[0, action]
                next_action = main_net(ns_tensor).argmax().item()
                q_next = target_net(ns_tensor)[0, next_action]

                target = shaped_reward + gamma * q_next * (0.0 if done else 1.0)
                td_error = abs(q_sa.item() - target.item())

            buffer.add((state, action, shaped_reward, next_state, done), td_error)

            state = next_state
            total_reward += reward
            high_tile = max(high_tile, np.max(state))  # track max tile

            if len(buffer.buffer) >= batch_size:
                samples, weights, indices = buffer.sample(batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*samples)

                # Batch prep
                batch_states = torch.tensor([preprocess(s) for s in batch_states], dtype=torch.float32).to(device)
                batch_actions = torch.tensor(batch_actions).to(device)
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                batch_next_states = torch.tensor([preprocess(s) for s in batch_next_states], dtype=torch.float32).to(device)
                batch_dones = torch.tensor(batch_dones, dtype=torch.float32).to(device)
                weights = weights.to(device)

                # Q(s,a)
                q_values = main_net(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                # Double DQN: action from main_net, value from target_net
                next_actions = main_net(batch_next_states).argmax(1, keepdim=True)
                next_q_values = target_net(batch_next_states).gather(1, next_actions).squeeze(1)

                # Bellman target
                target_q_values = batch_rewards + gamma * next_q_values * (1 - batch_dones)

                # Loss + backward
                loss = ((q_values - target_q_values.detach()) ** 2 * weights).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                td_errors = torch.abs(q_values - target_q_values.detach())
                buffer.update_priorities(indices, td_errors)  # refresh priorities

            total_steps += 1
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(main_net.state_dict())  # sync weights

        epsilon = max(epsilon * epsilon_decay, epsilon_min)  # decay epsilon

        if episode % 50 == 0:
            print(f"Episode {episode:03d} | Score: {total_reward:.0f} | High tile: {find_high_tile(state)} | Epsilon: {epsilon:.3f}")

    env.close()  # clean up
    torch.save(main_net.state_dict(), "dqn_2048.pth")
    print("Model saved!")
