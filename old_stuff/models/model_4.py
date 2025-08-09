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
import csv


class DQN_CNN(nn.Module):
    def __init__(self, input_channels=16, output_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(128 * 2 * 2, 512)
        self.out = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [batch, 128, 3, 3]
        x = F.relu(self.conv2(x))  # [batch, 128, 2, 2]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)


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
    result = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1)
    return result

def find_high_tile(state):
    state = np.argmax(state, axis=2)  # extract actual tiles
    state = 2 ** state * (state > 0)
    return np.max(state)  # return highest tile seen



from scipy.spatial.distance import pdist

def global_monotonicity(grid):
    """
    Checks if rows/columns follow decreasing pattern.
    """
    def line_score(line):
        nonzero = [x for x in line if x != 0]
        return sum(x >= y for x, y in zip(nonzero, nonzero[1:]))

    score = 0
    for row in grid:
        score += line_score(row)
    for col in grid.T:
        score += line_score(col)
    return score

def corner_priority(grid):
    max_tile = np.max(grid)
    if grid[0, 0] == max_tile:
        return np.log2(max_tile)
    return 0

def merge_potential(grid):
    """
    Reward possible merges. Scaled by tile value.
    """
    reward = 0.0
    for i in range(4):
        for j in range(4):
            val = grid[i, j]
            if val == 0:
                continue
            for dx, dy in [(1, 0), (0, 1)]:
                ni, nj = i + dx, j + dy
                if ni < 4 and nj < 4 and grid[ni, nj] == val:
                    reward += np.log2(val) * 2
    return reward

def board_emptiness_bonus(grid):
    """
    Encourage open space — especially early game.
    Personally I play with a lot more empty space
    """
    empty = np.count_nonzero(grid == 0)
    max_val = np.max(grid)
    return (empty / 16.0) ** 2 * np.log2(max_val) # stronger early on

def consolidation_score(grid):
    """
    Penalize large tiles scattered apart (≥16).
    """
    positions = np.argwhere(grid >= 16)
    if len(positions) <= 1:
        return 0.0
    dists = pdist(positions, 'cityblock')
    return -np.mean(dists) / 3.0

def is_game_over(grid):
    """
    Stronger penalty if game ends prematurely.
    """
    if np.any(grid == 0):
        return 0.0
    for i in range(4):
        for j in range(3):
            if grid[i, j] == grid[i, j + 1] or grid[j, i] == grid[j + 1, i]:
                return 0.0
    max_tile = np.max(grid)
    return -100.0 + np.log2(max_tile)

def max_tile_adjacent_bonus(grid):
    """
    Reward if two max tiles are adjacent (encourages consolidation of largest tiles).
    """
    max_val = np.max(grid)
    positions = np.argwhere(grid == max_val)
    for i, (x1, y1) in enumerate(positions):
        for x2, y2 in positions[i + 1:]:
            if abs(x1 - x2) + abs(y1 - y2) == 1:  # adjacent (manhattan distance 1)
                return max_val
    return 0.0

def reward_shaping(grid, step_count, reward):
    """
    Combined reward function. Assumes one-hot encoded Gym board.
    """
    grid = np.argmax(grid, axis=2)
    grid = 2 ** grid * (grid > 0)
    max_val = np.max(grid)
    return max_val + max_tile_adjacent_bonus(grid) + reward
    max_bonus = np.log2(max_val)**2

    # Modular components
    a = corner_priority(grid)
    b = global_monotonicity(grid)
    # d = merge_potential(grid)
    e = board_emptiness_bonus(grid)
    # f = consolidation_score(grid)
    g = is_game_over(grid)
    h = max_tile_adjacent_bonus(grid)
    t = -0.001 * step_count

    # print(a, b, e, g, t, reward, max_bonus, h)
    total_reward = (a + b + e + t + reward + max_bonus + h)
    total_reward = np.clip(total_reward, -100.0, 100.0)
    return total_reward


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")  # initialize environment

    main_net = DQN_CNN().to(device)  # main Q-network
    target_net = DQN_CNN().to(device)  # target Q-network (used for stability)
    target_net.load_state_dict(main_net.state_dict())  # sync weights
    optimizer = optim.Adam(main_net.parameters(), lr=1e-4)
    buffer = PrioritizedReplayBuffer(10000)  # initialize replay buffer

    batch_size = 64
    gamma = 0.99  # discount factor
    epsilon = 1.0  # exploration rate
    epsilon_decay = 0.998
    epsilon_min = 0.01
    target_update_freq = 100  # how often to sync target_net
    num_episodes = 2000
    total_steps = 0

    with open('run.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "total", "high_tile"])

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0.0
        high_tile = 0
        done = False
        steps = 0

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)  # explore
                next_state, reward, terminated, truncated, _ = env.step(action)
            else:
                state_tensor = torch.tensor(np.array([preprocess(state)]), dtype=torch.float32).to(device)
                with torch.no_grad():
                    # action = main_net(state_tensor).argmax().item()  # exploit
                    q_vals = main_net(state_tensor).squeeze(0)  # shape: [4]

                    actions = torch.argsort(q_vals, descending=True).tolist()
                    for action in actions:
                        next_state, reward, terminated, truncated, _ = env.step(action) # 1 is right, 0 is up, 3 is left, 4 is down
                        if not np.array_equal(state, next_state):
                            break

            steps += 1
            if np.array_equal(state, next_state):
                shaped_reward = reward_shaping(next_state, steps, reward) - 2.0  # base reward minus penalty
            else:
                shaped_reward = reward_shaping(next_state, steps, reward) # merge reward
            done = terminated or truncated

            with torch.no_grad(): # estimate priority (temporal-difference)
                s_tensor = torch.tensor(np.array([preprocess(state)]), dtype=torch.float32).to(device)
                ns_tensor = torch.tensor(np.array([preprocess(next_state)]), dtype=torch.float32).to(device)

                q_sa = main_net(s_tensor)[0, action]
                next_action = main_net(ns_tensor).argmax().item()
                q_next = target_net(ns_tensor)[0, next_action]

                target = shaped_reward + gamma * q_next * (0.0 if done else 1.0)
                td_error = abs(q_sa.item() - target.item())

            buffer.add((state, action, shaped_reward, next_state, done), td_error)

            state = next_state
            total_reward += float(reward)
            high_tile = max(high_tile, np.max(state))  # track max tile

            if len(buffer.buffer) >= batch_size:
                samples, weights, indices = buffer.sample(batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*samples)

                # Batch prep
                batch_states = torch.tensor(np.array([preprocess(s) for s in batch_states]), dtype=torch.float32).to(device)
                batch_actions = torch.tensor(batch_actions).to(device)
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                batch_next_states = torch.tensor(np.array([preprocess(s) for s in batch_next_states]), dtype=torch.float32).to(device)
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

        with open('run.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, shaped_reward, find_high_tile(state)])

        if episode % 50 == 0:
            print(f"Episode {episode:03d} | Shaped: {shaped_reward:.0f} | High tile: {find_high_tile(state)} | Epsilon: {epsilon:.3f}")

    env.close()  # clean up
    torch.save(main_net.state_dict(), "dqn_2048.pth")
    print("Model saved!")
