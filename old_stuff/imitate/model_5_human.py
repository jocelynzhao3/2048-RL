import gymnasium as gym  # for interacting with RL environments
import gymnasium_2048.envs  # registers the 2048 environment
import pygame
import pickle
import numpy as np
from gymnasium.wrappers import FlattenObservation

# Load environment
env = gym.make("gymnasium_2048/TwentyFortyEight-v0", render_mode=None)
env = FlattenObservation(env)
DEMO_DATA = []

# Helper function to draw the 2048 board
def draw_board(screen, state):
    tile_size = 100
    padding = 10
    font = pygame.font.SysFont(None, 36)
    board = 2 ** np.argmax(state.reshape(4, 4, 16), axis=2)
    for i in range(4):
        for j in range(4):
            value = board[i][j]
            rect = pygame.Rect(j * tile_size, i * tile_size, tile_size - padding, tile_size - padding)
            pygame.draw.rect(screen, (200, 200, 200), rect)
            if value > 0:
                if value == 1:
                    text = font.render("", True, (0, 0, 0))
                else:
                    text = font.render(str(value), True, (0, 0, 0))
                screen.blit(text, (j * tile_size + 25, i * tile_size + 30))

def play_and_collect(env):
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("2048 Human Play")
    clock = pygame.time.Clock()
    running = True
    state, _ = env.reset()

    while running:
        screen.fill((255, 255, 255))
        draw_board(screen, state)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3

                if action is not None:
                    DEMO_DATA.append((state.copy(), action))
                    state, reward, done, truncated, _ = env.step(action)
                    if done or truncated:
                        state, _ = env.reset()

        clock.tick(10)

    pygame.quit()
    with open("human_demos.pkl", "wb") as f:
        pickle.dump(DEMO_DATA, f)
    print(f"Saved {len(DEMO_DATA)} state-action pairs to human_demos.pkl")

# Run it
if __name__ == "__main__":
    play_and_collect(env)
