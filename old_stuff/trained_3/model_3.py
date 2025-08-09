# HRL-2048 Implementation with Meta-controller and Controller
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
import gymnasium_2048.envs
import csv

# --- Constants ---
GOALS = [64, 128, 256, 512, 1024]
gamma = 0.99
batch_size = 100
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
target_update_freq = 100
num_episodes = 1000

# Curriculum state
curriculum_threshold = 0.75  # Require 75% success rate
curriculum_window = 200
curriculum_progress = {g: deque(maxlen=curriculum_window) for g in GOALS}
unlocked_goals = [GOALS[0]]
last_unlocked_episode = 0
unlock_delay = 100

def update_curriculum(epsilon, episode):
    global unlocked_goals, last_unlocked_episode
    for i, g in enumerate(GOALS[:-1]):
        if g in unlocked_goals and GOALS[i+1] not in unlocked_goals:
            rate = sum(curriculum_progress[g]) / len(curriculum_progress[g]) if curriculum_progress[g] else 0.0
            print(f'Rate: {rate}')
            if rate > curriculum_threshold and len(curriculum_progress[g]) >= curriculum_window // 2 and episode - last_unlocked_episode > unlock_delay:
                print(f"🔓 Unlocking goal {GOALS[i+1]} (achieved {g} in {rate:.2f} of recent episodes)")
                unlocked_goals.append(GOALS[i+1])
                last_unlocked_episode = episode
                return max(epsilon, 0.6)
    return epsilon

# --- Utilities ---
def preprocess(state):
    state = np.argmax(state, axis=2)
    state = 2 ** state * (state > 0)
    result = np.log2(state + 1).flatten() / 16.0
    return result

def find_high_tile(state):
    state = np.argmax(state, axis=2)
    state = 2 ** state * (state > 0)
    return np.max(state)

def max_tile_in_corner(state):
    state = np.argmax(state, axis=2)
    grid = 2 ** state * (state > 0)
    max_val = np.max(grid)
    if max_val in [grid[0][0], grid[0][-1], grid[-1][-1], grid[-1][0]]:
        return 0.5
    return 0.0

def is_goal_achieved(state, goal):
    return find_high_tile(state) >= goal

def intrinsic_reward(state, goal):
    return 1.0 if is_goal_achieved(state, goal) else 0.0

# --- Models ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# --- Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.5):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def add(self, transition, td_error):
        max_prio = max(self.priorities.max(), abs(td_error) + 1e-6)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, episode, beta=0.4):
        beta += episode/num_episodes/5
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return samples, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err.item()) + 1e-6

    def __len__(self):
        return len(self.buffer)



# --- Main HRL Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("gymnasium_2048/TwentyFortyEight-v0")

meta_net = DQN(input_dim=16, output_dim=len(GOALS)).to(device)
meta_target = DQN(input_dim=16, output_dim=len(GOALS)).to(device)
meta_target.load_state_dict(meta_net.state_dict())
meta_opt = optim.Adam(meta_net.parameters(), lr=1e-4)
meta_buffer = PrioritizedReplayBuffer(10000)

controller_net = DQN(input_dim=16 + len(GOALS), output_dim=4).to(device)
controller_target = DQN(input_dim=16 + len(GOALS), output_dim=4).to(device)
controller_target.load_state_dict(controller_net.state_dict())
controller_opt = optim.Adam(controller_net.parameters(), lr=1e-4)
controller_buffer = PrioritizedReplayBuffer(10000)

total_steps = 0

if __name__ == "__main__":

    with open('run.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "total", "high_tile", "goal"])

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        goal = None

        while not done:
            # --- Meta-controller selects subgoal ---
            state_vec = torch.tensor(preprocess(state), dtype=torch.float32).unsqueeze(0).to(device)

            unlocked_idxs = [i for i, g in enumerate(GOALS) if g in unlocked_goals]
            goal_idx = random.choice(unlocked_idxs)
            # if random.random() < epsilon:
            #     unlocked_idxs = [i for i, g in enumerate(GOALS) if g in unlocked_goals]
            #     goal_idx = random.choice(unlocked_idxs)
            # else:
            #     with torch.no_grad():
            #         goal_idx = meta_net(state_vec).argmax().item()
            goal = GOALS[goal_idx]

            goal_vec = torch.zeros(len(GOALS))
            goal_vec[goal_idx] = 1.0

            subgoal_achieved = False
            steps = 0

            while not subgoal_achieved and not done and steps < 50:
                # --- Controller selects action ---
                input_vec = torch.cat([state_vec.squeeze(0), goal_vec]).unsqueeze(0)
                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_vals = controller_net(input_vec.to(device))
                        action = q_vals.argmax().item()

                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                reward = intrinsic_reward(next_state, goal) + max_tile_in_corner(next_state)

                state = next_state
                state_vec = torch.tensor(preprocess(state), dtype=torch.float32).unsqueeze(0).to(device)
                subgoal_achieved = reward > 0.0
                total_reward += reward
                steps += 1

            meta_reward = np.log2(find_high_tile(state) + 1) / 11.0

            if len(controller_buffer) >= batch_size:
                samples, weights, indices = controller_buffer.sample(batch_size, episode)
                s, g, a, r, ns, d = zip(*samples)

                s = torch.tensor(np.array(s), dtype=torch.float32).to(device)
                g = torch.tensor(np.array(g), dtype=torch.float32).to(device)
                a = torch.tensor(a).to(device)
                r = torch.tensor(r, dtype=torch.float32).to(device)
                ns = torch.tensor(np.array(ns), dtype=torch.float32).to(device)
                d = torch.tensor(d, dtype=torch.float32).to(device)
                weights = weights.to(device)

                input_s = torch.cat([s, g], dim=1)
                input_ns = torch.cat([ns, g], dim=1)

                q_sa = controller_net(input_s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = controller_target(input_ns).max(1)[0]
                target = r + gamma * q_next * (1 - d)

                td_errors = target - q_sa
                loss = (weights * td_errors.pow(2)).mean()

                controller_opt.zero_grad()
                loss.backward()
                controller_opt.step()

                controller_buffer.update_priorities(indices, td_errors.detach())

            # --- Meta-controller Learning ---
            if len(meta_buffer) >= batch_size:
                samples, weights, indices = meta_buffer.sample(batch_size, episode)
                s, a, r, ns, d = zip(*samples)

                s = torch.tensor(np.array(s), dtype=torch.float32).to(device)
                a = torch.tensor(a).to(device)
                r = torch.tensor(r, dtype=torch.float32).to(device)
                ns = torch.tensor(np.array(ns), dtype=torch.float32).to(device)
                d = torch.tensor(d, dtype=torch.float32).to(device)
                weights = weights.to(device)

                q_sa = meta_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = meta_target(ns).max(1)[0]
                target = r + gamma * q_next * (1 - d)

                td_errors = target - q_sa
                loss = (weights * td_errors.pow(2)).mean()

                meta_opt.zero_grad()
                loss.backward()
                meta_opt.step()

                meta_buffer.update_priorities(indices, td_errors.detach())

            # --- Add transitions with TD error ---
            # Controller TD error calculation
            controller_state_input = torch.cat([state_vec.squeeze(0), goal_vec]).unsqueeze(0).to(device)
            with torch.no_grad():
                current_q = controller_net(controller_state_input).squeeze(0)[action]
                next_input = torch.cat([torch.tensor(preprocess(next_state), dtype=torch.float32), goal_vec]).unsqueeze(0).to(device)
                max_next_q = controller_target(next_input).max(1)[0].item()
                td_error = reward + gamma * max_next_q * (1 - int(done)) - current_q.item()

            controller_buffer.add(
                (preprocess(next_state), goal_vec.numpy(), action, reward, preprocess(next_state), done),
                abs(td_error)
            )

            # Meta-controller TD error calculation
            state_vec_cpu = torch.tensor(preprocess(state), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_val = meta_net(state_vec_cpu.to(device))[0][goal_idx].item()
                next_q = meta_target(state_vec_cpu.to(device)).max(1)[0].item()
                meta_td_error = meta_reward + gamma * next_q * (1 - int(done)) - q_val

            meta_buffer.add(
                (preprocess(next_state), goal_idx, meta_reward, preprocess(state), done),
                abs(meta_td_error)
            )


            total_steps += 1
            if total_steps % target_update_freq == 0:
                meta_target.load_state_dict(meta_net.state_dict())
                controller_target.load_state_dict(controller_net.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        curriculum_progress[goal].append(1.0 if subgoal_achieved else 0.0)
        epsilon = update_curriculum(epsilon, episode)

        with open('run.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, total_reward, find_high_tile(state), goal])


        if episode % 100 == 0:
            print(f"Ep {episode} | Reward: {total_reward:.2f} | Eps: {epsilon:.3f} | High tile: {find_high_tile(state)}")


    env.close()
    torch.save(meta_net.state_dict(), "meta_controller.pth")
    torch.save(controller_net.state_dict(), "controller.pth")
    print("Models saved!")
