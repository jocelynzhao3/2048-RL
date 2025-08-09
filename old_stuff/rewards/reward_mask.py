import numpy as np

goal_mask = np.array([
    [16, 15, 14, 13],
    [9, 10, 11, 12],
    [8, 7, 6, 5],
    [4, 3, 2, 1]
])

def board_log2(grid):
    return np.where(grid > 0, np.log2(grid), 0)

def snake_score(grid, base_mask=goal_mask):
    log_grid = board_log2(grid)
    max_log = np.max(log_grid) if np.max(log_grid) > 0 else 1
    max_val = np.max(grid)

    best_score = -np.inf
    for k in range(4):  # try 0°, 90°, 180°, 270°
        rotated_mask = np.rot90(base_mask, k=k)
        norm_mask = rotated_mask / np.max(rotated_mask)
        tile_weights = log_grid / max_log
        tile_weights = tile_weights ** 1.5

        score = np.sum(tile_weights * norm_mask)
        best_score = max(best_score, score)

    # Boost if 2nd and 3rd top tiles are near the top
    flat = grid.flatten()
    top_tiles = np.sort(flat[flat > 0])[-3:]
    if len(top_tiles) == 3 and top_tiles[2] / top_tiles[1] <= 2 and top_tiles[1] / top_tiles[0] <= 2:
        best_score += 1.5

    return best_score

def max_tile_in_corner(grid):
    max_val = np.max(grid)

    corners = [grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]]
    return float(max_val in corners)

def boxed_values(grid):
    rows, cols = grid.shape

    for i in range(rows):
        for j in range(cols):
            val = grid[i, j]
            if val == 0:
                continue

            up    = grid[i - 1, j] if i > 0 else float('inf')
            down  = grid[i + 1, j] if i < rows - 1 else float('inf')
            left  = grid[i, j - 1] if j > 0 else float('inf')
            right = grid[i, j + 1] if j < cols - 1 else float('inf')

            if all(neigh > val for neigh in [up, down, left, right]):
                return -1.0
    return 0.0


def pinned_max_side(grid):
    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)

    if max(col_sums) > max(row_sums):
        max_col_index = np.argmax(col_sums)
        if 0 not in grid[:, max_col_index]:
            return 0.5
    else:
        max_row_index = np.argmax(row_sums)
        if 0 not in grid[max_row_index, :]:
            return 0.5

    return 0.0

def board_emptiness(grid):
    return np.count_nonzero(grid == 0)


def is_game_over(grid):
    if np.any(grid == 0):
        return False
    for i in range(4):
        for j in range(3):
            if grid[i,j] == grid[i,j+1] or grid[j,i] == grid[j+1,i]:
                return 0.0
    max_tile = np.max(grid)
    return -10 if max_tile < 128 else -5 if max_tile < 256 else -1


def reward_shaping(grid):
    """
    Main reward function
    The goal is to score a board based on how "good" it is

    grid: one-hot encoded state representation of the 2048 board
    returns bonus shaped reward value
    """
    grid = np.argmax(grid, axis=2)
    grid = 2 ** grid * (grid > 0)

    reward =(
    1.5 * max_tile_in_corner(grid) +
    2.0 * snake_score(grid) +
    0.5 * pinned_max_side(grid) +
    0.2 * board_emptiness(grid) +
    boxed_values(grid) +
    is_game_over(grid))

    return reward
