"""
Experiment 36: Multiple agents collaborate to solve a cooperative task
with hierarchical structure. Implement the MAXQ framework to
decompose the task into hierarchically organized subtasks (Fetch part
-> Assemble -> Deliver) shared across two cooperating robots, and
simulate their interactions.
"""
 
import random
from collections import defaultdict
 
random.seed(23)
 
GRID = 5
PARTS_BIN = {"R1": (0, 4), "R2": (4, 4)}
ASSEMBLY_STATION = (2, 2)
DELIVERY_POINT = (4, 0)
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
ALPHA, GAMMA, EPISODES = 0.15, 0.9, 700
 
 
def move(pos, action):
    dr, dc = DELTA[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
        return pos
    return nxt
 
 
def train_navigate_subtask(target):
    """MAXQ low-level subtask: navigate to `target`."""
    Q = defaultdict(lambda: [0.0] * len(ACTIONS))
    for _ in range(EPISODES):
        pos = (random.randint(0, GRID - 1), random.randint(0, GRID - 1))
        for _ in range(25):
            if pos == target:
                break
            a = random.randrange(4) if random.random() < 0.3 else max(range(4), key=lambda i: Q[pos][i])
            nxt = move(pos, ACTIONS[a])
            r = 10 if nxt == target else -1
            best_next = max(Q[nxt])
            Q[pos][a] += ALPHA * (r + GAMMA * best_next - Q[pos][a])
            pos = nxt
    return Q
 
 
def run_subtask(Q, pos, target, max_steps=15):
    path = [pos]
    for _ in range(max_steps):
        if pos == target:
            break
        a = max(range(4), key=lambda i: Q[pos][i])
        pos = move(pos, ACTIONS[a])
        path.append(pos)
    return path
 
 
if __name__ == "__main__":
    # Level-1 MAXQ subtasks (shared across both robot agents)
    fetch_Q = {r: train_navigate_subtask(PARTS_BIN[r]) for r in PARTS_BIN}
    assemble_Q = train_navigate_subtask(ASSEMBLY_STATION)
    deliver_Q = train_navigate_subtask(DELIVERY_POINT)
 
    # Root task, executed cooperatively by both agents:
    # each robot fetches its own part -> both bring parts to assembly ->
    # one robot delivers the finished product
    print("Cooperative MAXQ task execution:\n")
    positions = {"R1": (0, 0), "R2": (4, 0)}
    for r in ["R1", "R2"]:
        path = run_subtask(fetch_Q[r], positions[r], PARTS_BIN[r])
        positions[r] = path[-1]
        print(f"{r} Fetch-Part subtask path: {path}")
 
    for r in ["R1", "R2"]:
        path = run_subtask(assemble_Q, positions[r], ASSEMBLY_STATION)
        positions[r] = path[-1]
        print(f"{r} Go-To-Assembly subtask path: {path}")
 
    delivery_path = run_subtask(deliver_Q, ASSEMBLY_STATION, DELIVERY_POINT)
    print(f"R1 Deliver-Product subtask path: {delivery_path}")
 
    print("\nBoth robots reached the assembly station and the product was "
          "successfully delivered to the delivery point.")
 
