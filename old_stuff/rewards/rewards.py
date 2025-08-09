import numpy as np


def max_tile_in_corner(grid):
    max_val = np.max(grid)

    corners = [grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]]
    if max_val in corners:
        return 1.5
    return 0.0


# def is_strict_by_power_of_2(arr):
#     arr = arr[arr > 0]
#     if len(arr) < 2:
#         return False

#     logs = np.log2(arr)
#     diffs = np.diff(logs)

#     if np.all(diffs == 1):
#         return 1.0
#     if np.all(diffs == -1):
#         return -1.0
#     return 0.0

# def has_alternating_pattern(arr):
#     arr = arr[arr != 0]
#     if len(arr) < 2:
#         return False
#     return np.all(arr[:-1] * arr[1:] == -1)


# def snake_decreasing_monotonicity(grid):  # TODO: improve to ensure snake down by powers of 2?
#     if not max_tile_in_corner(grid):
#         return 0

#     """
#     Along AXIS 0: columns
#     Along AXIS 1: rows
#     """

#     result_col = np.apply_along_axis(is_strict_by_power_of_2, 0, grid)
#     result_row = np.apply_along_axis(is_strict_by_power_of_2, 1, grid)

#     row_sums = grid.sum(axis=1)
#     col_sums = grid.sum(axis=0)


#     if max(col_sums) > max(row_sums):
#         bonus=np.sum(abs(result_col)) / 10
#         if has_alternating_pattern(result_col):
#             bonus += 2
#     else:
#         bonus=np.sum(abs(result_row)) / 10
#         if has_alternating_pattern(result_col):
#             bonus += 2
#     max_tile = np.max(grid)
#     return (np.log2(max_tile) / 10.0) * bonus



def is_strict_decreasing_powers(arr):
    arr = [x for x in arr if x != 0]
    return all(x >= y for x, y in zip(arr, arr[1:]))    # x = 2*y

def compute_monotonicity(grid):
    bonus = 0.0
    for line in list(grid) + list(grid.T):
        arr = [x for x in line if x > 0]
        if len(arr) < 2:
            continue
        logs = np.log2(arr)
        if all(x >= y for x, y in zip(logs, logs[1:])) or all(x <= y for x, y in zip(logs, logs[1:])):
            bonus += np.sum(logs) / 40  # reward more for high monotonic tiles
    return bonus



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
                return -min(np.log2([up, down, left, right]))
    return 0

def flat_against_high_side(grid):  # could simplify this to punish against bad action?

    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)

    if max(col_sums) > max(row_sums):
        zero_cols = np.where(np.all(grid == 0, axis=0))[0]
        for col in zero_cols:
            # check left neighbor
            if col > 0 and np.all(grid[:, col - 1] != 0):
                return -1.0
            # check right neighbor
            if col < grid.shape[1] - 1 and np.all(grid[:, col + 1] != 0):
                return -1.0
    else:
        zero_rows = np.where(np.all(grid == 0, axis=1))[0]
        for row in zero_rows:
            # check upper neighbor
            if row > 0 and np.all(grid[row - 1] != 0):
                return -1.0
            # check lower neighbor
            if row < grid.shape[0] - 1 and np.all(grid[row + 1] != 0):
                return -1.0

    return 0.5


def board_emptiness(grid):
    empty = np.count_nonzero(grid == 0)
    return (empty ** 1.5) / 8.0  # scale more aggressively

def one_of_high_tile(grid):
    mask = (grid > 32)
    filtered_arr = grid[mask]
    unique_elements = np.unique(filtered_arr, return_counts=False)
    return float(len(unique_elements) == len(filtered_arr))

def pinned_max_side(grid):
    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)

    if max(col_sums) > max(row_sums):
        max_col_index = np.argmax(col_sums)
        if 0 not in grid[:, max_col_index]:
            return 1.0
    else:
        max_row_index = np.argmax(row_sums)
        if 0 not in grid[max_row_index, :]:
            return 1.0

    return 0.0

def high_tile_flow(grid):
    max_val = np.max(grid)
    pos = np.argwhere(grid == max_val)[0]
    r, c = pos
    score = 0
    if r == 0 or r == grid.shape[0] - 1:
        if all(grid[r, i] >= grid[r, i+1] for i in range(grid.shape[1]-1)):
            score += 0.5
    if c == 0 or c == grid.shape[1] - 1:
        if all(grid[i, c] >= grid[i+1, c] for i in range(grid.shape[0]-1)):
            score += 0.5
    return score

def merge_potential(grid):
    merges = 0
    for row in grid:
        merges += sum(row[i] == row[i+1] for i in range(3) if row[i] != 0)
    for col in grid.T:
        merges += sum(col[i] == col[i+1] for i in range(3) if col[i] != 0)
    return merges * 0.1

def tile_variance_penalty(grid):
    nonzero = grid[grid > 0]
    if len(nonzero) <= 1:
        return 0
    return -np.var(np.log2(nonzero))


def reward_shaping(grid):
    """
    Main reward function
    The goal is to score a board based on how "good" it is

    grid: one-hot encoded state representation of the 2048 board
    returns bonus shaped reward value
    """
    grid = np.argmax(grid, axis=2)
    grid = 2 ** grid * (grid > 0)

    a = max_tile_in_corner(grid)
    b = compute_monotonicity(grid)
    c = boxed_values(grid)
    d = flat_against_high_side(grid)
    e = board_emptiness(grid)
    f = pinned_max_side(grid)
    g = high_tile_flow(grid)
    h = merge_potential(grid)
    i = tile_variance_penalty(grid)

    raw = a + b + c + d + e + f + g + h + i
    return np.tanh(raw / 5.0)  # scaling factor adjusts sharpness
