"""
Experiment 28: A robot navigates a grid to perform tasks. Use
Bellman's optimality equation to compute the optimal state-value
function for the robot's navigation tasks, and demonstrate the
optimal path.
"""
 
GRID = 5
GOAL = (4, 0)
OBSTACLES = {(1, 0), (1, 1), (3, 3)}
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
GAMMA = 0.9
STEP_COST = -1
GOAL_REWARD = 15
THETA = 1e-4
 
 
def next_state(s, a):
    dr, dc = DELTA[a]
    nxt = (s[0] + dr, s[1] + dc)
    if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID) or nxt in OBSTACLES:
        return s
    return nxt
 
 
def reward(s, s2):
    return GOAL_REWARD if s2 == GOAL else STEP_COST
 
 
def value_iteration():
    V = {(r, c): 0.0 for r in range(GRID) for c in range(GRID) if (r, c) not in OBSTACLES}
    sweeps = 0
    while True:
        delta = 0
        for s in list(V.keys()):
            if s == GOAL:
                continue
            best = max(reward(s, next_state(s, a)) + GAMMA * V[next_state(s, a)] for a in ACTIONS)
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        sweeps += 1
        if delta < THETA:
            break
    return V, sweeps
 
 
if __name__ == "__main__":
    V, sweeps = value_iteration()
    print(f"Bellman optimality equation converged in {sweeps} sweeps.\n")
 
    print("Optimal state-value function V*(s) (row by row):")
    for r in range(GRID):
        row_vals = [f"{V[(r, c)]:6.2f}" if (r, c) not in OBSTACLES else "  OBS " for c in range(GRID)]
        print(" ".join(row_vals))
 
    state, path = (0, 4), [(0, 4)]
    for _ in range(15):
        if state == GOAL:
            break
        best_a = max(ACTIONS, key=lambda a: reward(state, next_state(state, a)) + GAMMA * V[next_state(state, a)])
        state = next_state(state, best_a)
        path.append(state)
    print(f"\nOptimal path from (0,4) to goal {GOAL}:")
    print(" -> ".join(str(p) for p in path))
 
