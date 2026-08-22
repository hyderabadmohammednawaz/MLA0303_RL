"""
Experiment 17: Hierarchical Reinforcement Learning (HRL) for an
autonomous household robot performing multiple tasks (fetch item,
then dispose trash) efficiently, using a MAXQ-style task decomposition
(root task -> {Fetch subtask, Dispose subtask} -> primitive navigate
actions). HAM (Hierarchies of Abstract Machines) corresponds to
encoding each subtask as a small finite-state machine that only
exposes a restricted action set to the level above it, which is what
the Fetch/Dispose subtask policies below do.
"""
 
import random
from collections import defaultdict
 
random.seed(6)
 
GRID = 5
ITEM = (0, 4)
TRASH_BIN = (4, 0)
HOME = (4, 4)
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
ALPHA, GAMMA, EPISODES = 0.15, 0.9, 800
 
 
def move(pos, action):
    dr, dc = DELTA[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
        return pos
    return nxt
 
 
def train_subtask_policy(target):
    """Low-level MAXQ subtask: learn to navigate the robot to `target`."""
    Q = defaultdict(lambda: [0.0] * len(ACTIONS))
    for _ in range(EPISODES):
        pos = (random.randint(0, GRID - 1), random.randint(0, GRID - 1))
        epsilon = 0.3
        for _ in range(30):
            if pos == target:
                break
            if random.random() < epsilon:
                a = random.randrange(len(ACTIONS))
            else:
                a = max(range(len(ACTIONS)), key=lambda i: Q[pos][i])
            nxt = move(pos, ACTIONS[a])
            reward = 10 if nxt == target else -1
            best_next = max(Q[nxt])
            Q[pos][a] += ALPHA * (reward + GAMMA * best_next - Q[pos][a])
            pos = nxt
    return Q
 
 
def run_subtask(Q, pos, target, max_steps=15):
    path = [pos]
    for _ in range(max_steps):
        if pos == target:
            break
        a = max(range(len(ACTIONS)), key=lambda i: Q[pos][i])
        pos = move(pos, ACTIONS[a])
        path.append(pos)
    return path
 
 
if __name__ == "__main__":
    # Level 1 (MAXQ subtasks): learn navigation policies for each subtask
    fetch_Q = train_subtask_policy(ITEM)
    dispose_Q = train_subtask_policy(TRASH_BIN)
    home_Q = train_subtask_policy(HOME)
 
    # Level 0 (Root task): sequence the subtasks --
    # Fetch item -> Dispose trash -> Return home
    pos = (0, 0)
    full_path = [pos]
 
    path1 = run_subtask(fetch_Q, pos, ITEM)
    full_path += path1[1:]
    pos = ITEM
    print("Root task step 1 -- Fetch subtask path:", path1)
 
    path2 = run_subtask(dispose_Q, pos, TRASH_BIN)
    full_path += path2[1:]
    pos = TRASH_BIN
    print("Root task step 2 -- Dispose subtask path:", path2)
 
    path3 = run_subtask(home_Q, pos, HOME)
    full_path += path3[1:]
    print("Root task step 3 -- Return-home subtask path:", path3)
 
    print("\nFull hierarchical task execution path:")
    print(" -> ".join(str(p) for p in full_path))
    print(f"Total primitive actions executed: {len(full_path) - 1}")
 
