import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import gymnasium_2048.envs
import torch.nn as nn  # for building neural networks
from models.model_2 import DQN, preprocess, find_high_tile  # import your model class and preprocess function
import time

import csv

env = gym.make("gymnasium_2048/TwentyFortyEight-v0", render_mode="human")

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN().to(device)
model.load_state_dict(torch.load("trained_models/dqn_2048_hits_1024.pth", map_location=device))
model.eval()

import random

human_feedback = []

def ask_human(state):
    print("Current state:")
    state = np.argmax(state, axis=2)  # extract log2 tile values from one-hot
    state = 2 ** state * (state > 0)
    print(state)  # Visualize
    action = input("Choose action (1 is right, 0 is up, 3 is left, 2 is down): ")
    while action not in "0123" or len(action) == 0:
        action = input("Choose action (1 is right, 0 is up, 3 is left, 2 is down): ")
    return int(action)

def get_action_with_feedback(state, model, feedback_prob=1):
    # Flatten the board
    grid = np.argmax(state, axis=2)  # extract log2 tile values from one-hot
    grid = 2 ** grid * (grid > 0)
    max_value = np.max(grid)
    state_tensor = torch.tensor(np.array([preprocess(state)]), dtype=torch.float32).to(device)
    q_values = model(state_tensor).squeeze(0)  # shape: [4]

    if (random.random() < feedback_prob and max_value >=256):
        human_action = ask_human(state)
        human_feedback.append((state_tensor, human_action))
        next_state, reward, done, truncated, info = env.step(human_action)
        return next_state, reward, done
    else:
        actions = torch.argsort(q_values, descending=True).tolist()
        for action in actions:
            next_state, reward, terminated, truncated, _ = env.step(action) # 1 is right, 0 is up, 3 is left, 2 is down
            if not np.array_equal(state, next_state):
                break
        return next_state, reward, terminated

def fine_tune_model(model, feedback, lr=1e-4, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Fix here: use torch.stack to batch the input tensors
    X = torch.stack([s.squeeze(0).cpu() for s, a in feedback])  # shape: [N, 16]
    y = torch.LongTensor([a for s, a in feedback])

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)  # shape: [N, 4]
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


NUM_EPISODES = 1  # Change as you want

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Ask model/human for action
        next_state, reward, done = get_action_with_feedback(state, model, feedback_prob=0.1)
        total_reward += reward
        state = next_state

        time.sleep(0.01)  # Slow down rendering to observe

    print(f"Episode {episode+1} finished with reward {total_reward}")


if len(human_feedback) > 0:
    fine_tune_model(model, human_feedback, lr=1e-4, epochs=100)
    torch.save(model.state_dict(), "dqn_2048_human_tuned.pth")
    print("Model updated and saved as dqn_2048_human_tuned.pth")
else:
    print("No human feedback collected — skipping fine-tuning.")
