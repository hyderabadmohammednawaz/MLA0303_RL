"""
Experiment 27: Simulate an autonomous car navigating a simple road
network with intersections, following traffic rules, using policy
iteration to reach the destination safely.
State = (row, col, stopped_flag) on a grid road network with a few
intersections that require the car to STOP before proceeding.
"""
 
GRID = 5
DESTINATION = (4, 4)
INTERSECTIONS = {(1, 1), (2, 3), (3, 1)}
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STOP_AND_GO"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
GAMMA = 0.9
 
 
def next_pos(pos, action):
    if action == "STOP_AND_GO":
        return pos
    dr, dc = DELTA[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
        return pos
    return nxt
 
 
def transition(pos, action, stopped_here):
    nxt = next_pos(pos, action)
    if pos in INTERSECTIONS and not stopped_here and action != "STOP_AND_GO":
        return nxt, False, -10            # traffic-rule violation: ran the intersection
    if action == "STOP_AND_GO":
        return pos, True, -0.5             # small time cost for obeying the stop
    if nxt == DESTINATION:
        return nxt, False, 20
    return nxt, False, -1
 
 
STATES = [(r, c, s) for r in range(GRID) for c in range(GRID) for s in (False, True)]
 
 
def policy_iteration():
    V = {s: 0.0 for s in STATES}
    policy = {s: "RIGHT" for s in STATES if s[:2] != DESTINATION}
    for _ in range(50):
        for _ in range(100):                       # policy evaluation
            delta = 0
            for s in STATES:
                if s[:2] == DESTINATION:
                    continue
                pos, stopped = s[:2], s[2]
                a = policy[s]
                nxt, new_stopped, r = transition(pos, a, stopped)
                new_v = r + GAMMA * V[(nxt[0], nxt[1], new_stopped)]
                delta = max(delta, abs(new_v - V[s]))
                V[s] = new_v
            if delta < 1e-4:
                break
        stable = True                                # policy improvement
        for s in STATES:
            if s[:2] == DESTINATION:
                continue
            pos, stopped = s[:2], s[2]
            best_a, best_v = None, float("-inf")
            for a in ACTIONS:
                nxt, new_stopped, r = transition(pos, a, stopped)
                v = r + GAMMA * V[(nxt[0], nxt[1], new_stopped)]
                if v > best_v:
                    best_v, best_a = v, a
            if best_a != policy[s]:
                stable = False
            policy[s] = best_a
        if stable:
            break
    return policy, V
 
 
if __name__ == "__main__":
    policy, V = policy_iteration()
    state = (0, 0, False)
    path = [state[:2]]
    actions_log = []
    violations = 0
    for _ in range(30):
        if state[:2] == DESTINATION:
            break
        a = policy[state]
        pos, stopped = state[:2], state[2]
        if pos in INTERSECTIONS and not stopped and a != "STOP_AND_GO":
            violations += 1
        nxt, new_stopped, r = transition(pos, a, stopped)
        state = (nxt[0], nxt[1], new_stopped)
        actions_log.append(a)
        path.append(state[:2])
 
    print("Optimal traffic-rule-respecting route:")
    print(" -> ".join(str(p) for p in path))
    print("Actions taken:", actions_log)
    print(f"Traffic-rule violations along the route: {violations}")
 
