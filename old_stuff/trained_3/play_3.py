import torch
import numpy as np
import gymnasium as gym
import time
from trained_3.model_3 import DQN, preprocess, find_high_tile

# --- Load Environment ---
env = gym.make("gymnasium_2048/TwentyFortyEight-v0", render_mode="human")
state, info = env.reset()
done = False

# --- Load Models ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GOALS = [128, 256, 512, 1024, 2048]
meta_model = DQN(input_dim=16, output_dim=len(GOALS)).to(device)
meta_model.load_state_dict(torch.load("meta_controller.pth", map_location=device))
meta_model.eval()

controller_model = DQN(input_dim=16 + len(GOALS), output_dim=4).to(device)
controller_model.load_state_dict(torch.load("controller.pth", map_location=device))
controller_model.eval()

# --- Define goals ---
GOAL_TILES = [64, 128, 256, 512, 1024]

def select_goal(meta_model, state_tensor):
    with torch.no_grad():
        goal_qs = meta_model(state_tensor)
        goal_idx = torch.argmax(goal_qs).item()
    return GOAL_TILES[goal_idx]

def goal_achieved(state, goal):
    return find_high_tile(state) >= goal

def one_hot_goal(goal, goal_space):
    vec = np.zeros(len(goal_space), dtype=np.float32)
    if goal in goal_space:
        vec[goal_space.index(goal)] = 1.0
    return vec

# --- Run Episode ---
total_reward = 0
total_moves = 0

# --- Select goal and convert to one-hot vector ---
goal = select_goal(meta_model, torch.tensor([preprocess(state)], dtype=torch.float32).to(device))
goal_vec = one_hot_goal(goal, GOALS)  # use 5-dim one-hot

while not done:
    # Format state and goal for controller
    state_tensor = torch.tensor([preprocess(state)], dtype=torch.float32).to(device)  # shape: [1, 16]
    goal_tensor = torch.tensor([goal_vec], dtype=torch.float32).to(device)            # shape: [1, 5]
    state_goal_tensor = torch.cat([state_tensor, goal_tensor], dim=1)                 # shape: [1, 21]

    # Choose best action
    with torch.no_grad():
        q_vals = controller_model(state_goal_tensor).squeeze(0)
        actions = torch.argsort(q_vals, descending=True).tolist()

    # Try valid actions until state changes
    for action in actions:
        next_state, reward, terminated, truncated, _ = env.step(action)
        if not np.array_equal(state, next_state):
            break

    done = terminated or truncated
    total_reward += reward
    total_moves += 1
    state = next_state

    # Pick new goal if achieved
    if goal_achieved(state, goal):
        goal = select_goal(meta_model, torch.tensor([preprocess(state)], dtype=torch.float32).to(device))
        goal_vec = one_hot_goal(goal, GOALS)

    env.render()
    time.sleep(0.03)

print(f"Game Over! Score: {total_reward}")
print(f"High tile: {find_high_tile(state)} in {total_moves} moves")
env.close()
