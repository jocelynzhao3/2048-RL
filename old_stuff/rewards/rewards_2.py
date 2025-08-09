import numpy as np


def count_merges(grid):
    total = 0
    for row in grid:
        for i in range(3):
            if row[i] != 0 and row[i] == row[i+1]:
                total += np.log2(row[i] * 2)
    for col in grid.T:
        for i in range(3):
            if col[i] != 0 and col[i] == col[i+1]:
                total += np.log2(col[i] * 2)
    return total

def smoothness(grid):
    diff = 0
    for i in range(4):
        for j in range(3):
            if grid[i, j] and grid[i, j+1]:
                diff += abs(np.log2(grid[i, j]) - np.log2(grid[i, j+1]))
            if grid[j, i] and grid[j+1, i]:
                diff += abs(np.log2(grid[j, i]) - np.log2(grid[j+1, i]))
    return diff

def monotonicity(grid):
    score = 0
    for row in grid:
        inc = sum(row[i] >= row[i+1] for i in range(3))
        dec = sum(row[i] <= row[i+1] for i in range(3))
        score += max(inc, dec)
    for col in grid.T:
        inc = sum(col[i] >= col[i+1] for i in range(3))
        dec = sum(col[i] <= col[i+1] for i in range(3))
        score += max(inc, dec)
    return score

def is_game_over(grid):
    if np.any(grid == 0):
        return False
    for i in range(4):
        for j in range(3):
            if grid[i,j] == grid[i,j+1] or grid[j,i] == grid[j+1,i]:
                return False
    return True


def shaped_reward_function(grid):
    grid = np.argmax(grid, axis=2)
    grid = 2 ** grid * (grid > 0)

    reward = 0.0

    # === Merge Reward ===
    merges = count_merges(grid)                  # log2(new tile from merge)
    reward += merges

    # === Max Tile ===
    max_tile = np.max(grid)
    reward += np.log2(max_tile)

    # === Smoothness ===
    reward += -0.1 * smoothness(grid)

    # === Monotonicity ===
    reward += 0.1 * monotonicity(grid)

    # === Empty Tiles ===
    reward += 0.3 * np.count_nonzero(grid == 0)

    # === Max in Corner Bonus ===
    if max_tile in [grid[0,0], grid[0,3], grid[3,0], grid[3,3]]:
        reward += 1.5

    # === Optional: Game Over Penalty ===
    if is_game_over(grid):
        reward -= 100

    return reward
