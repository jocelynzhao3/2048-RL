import pygame
import numpy as np
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import csv

class DQN(nn.Module):
    def __init__(self, input_dim=256, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def human_preprocess(state):
    return np.array(state, dtype=np.float32)

def find_high_tile(state):
    state = np.argmax(state, axis=2)  # extract actual tiles
    state = 2 ** state * (state > 0)
    return np.max(state)  # return highest tile seen

def train_behavior_cloning(model, demo_file="human_demos.pkl", epochs=100):
    with open(demo_file, "rb") as f:
        demos = pickle.load(f)

    X = torch.tensor([human_preprocess(s) for s, a in demos], dtype=torch.float32)
    y = torch.tensor([a for s, a in demos], dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "pretrained_bc.pth")






import gymnasium as gym  # for interacting with RL environments
import gymnasium_2048.envs  # registers the 2048 environment
import torch.optim as optim  # for optimizer (e.g. Adam)
import random


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
    # Convert from one-hot or tile values to flat float vector of length 256
    if isinstance(state, np.ndarray):
        if state.ndim == 3 and state.shape[2] == 16:  # one-hot [4, 4, 16]
            state = np.argmax(state, axis=2)
        state = 2 ** state * (state > 0)
        flat = state.flatten()
        vec = np.zeros(256, dtype=np.float32)
        for val in flat:
            if val != 0:
                vec[int(np.log2(val))] += 1
        return vec
    else:
        raise ValueError("Expected state to be a NumPy array.")



def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available
    env = gym.make("gymnasium_2048/TwentyFortyEight-v0")  # initialize environment

    main_net = DQN().to(device)  # main Q-network
    main_net.load_state_dict(torch.load("pretrained_bc.pth", map_location=device))
    target_net = DQN().to(device)  # target Q-network (used for stability)
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
            done = terminated or truncated

            with torch.no_grad(): # estimate priority (temporal-difference)
                s_tensor = torch.tensor(np.array([preprocess(state)]), dtype=torch.float32).to(device)
                ns_tensor = torch.tensor(np.array([preprocess(next_state)]), dtype=torch.float32).to(device)

                q_sa = main_net(s_tensor)[0, action]
                next_action = main_net(ns_tensor).argmax().item()
                q_next = target_net(ns_tensor)[0, next_action]

                target = reward + gamma * q_next * (0.0 if done else 1.0)
                td_error = abs(q_sa.item() - target.item())

            buffer.add((state, action, reward, next_state, done), td_error)

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
            writer.writerow([episode, reward, find_high_tile(state)])

        if episode % 50 == 0:
            print(f"Episode {episode:03d} | Reward: {reward:.0f} | High tile: {find_high_tile(state)} | Epsilon: {epsilon:.3f}")

    env.close()  # clean up
    torch.save(main_net.state_dict(), "dqn_2048.pth")
    print("Model saved!")


if __name__ == "__main__":
    # model = DQN()
    # train_behavior_cloning(model)

    train_model()
