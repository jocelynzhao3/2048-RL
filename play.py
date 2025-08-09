"""
GUI for 2048 gameplay with trained model
"""

import torch
import numpy as np
import gymnasium as gym
import gymnasium_2048.envs
from model import DQN_CNN, preprocess
import time


if __name__ == "__main__":

    env = gym.make("gymnasium_2048/TwentyFortyEight-v0", render_mode="human")
    state, _ = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN_CNN().to(device)
    model.load_state_dict(torch.load("trained_models/dqn_2048_3000.pth", map_location=device))
    model.eval()

    done = False
    total_reward = 0.0
    total_moves = 0

    while not done:
        state_tensor = torch.tensor(np.array([preprocess(state)]), dtype=torch.float32).to(device)
        with torch.no_grad():
            q_vals = model(state_tensor).squeeze(0)

            actions = torch.argsort(q_vals, descending=True).tolist()
            for action in actions:
                if action == 2:
                    continue
                next_state, reward, terminated, truncated, _ = env.step(action) # 1 is right, 0 is up, 3 is left, 2 is down
                if not np.array_equal(state, next_state):
                    break
            else:
                next_state, reward, terminated, truncated, _ = env.step(2)

        done = terminated or truncated
        total_reward += reward
        total_moves += 1

        state = next_state
        env.render()
        time.sleep(0.03)

    print(f"Game Over! Score: {total_reward}")
    grid = np.argmax(next_state, axis=2)
    grid = 2 ** grid * (grid > 0)
    print(f"Ending board state: \n {grid}")
    env.close()
