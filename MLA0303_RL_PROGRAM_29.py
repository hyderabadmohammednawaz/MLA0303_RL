"""
Experiment 29: Design a dynamic programming solution to optimize
traffic light timings at an intersection to minimize vehicle wait
times. Define states, actions, rewards and transitions, and
implement the policy iteration algorithm.
State = (NS_queue, EW_queue, current_green) each queue in 0..4.
Actions = KEEP_PHASE, SWITCH_PHASE.
"""
 
import itertools
 
MAX_Q = 4
ACTIONS = ["KEEP", "SWITCH"]
GAMMA = 0.9
 
STATES = list(itertools.product(range(MAX_Q + 1), range(MAX_Q + 1), [0, 1]))
 
 
def transition(state, action):
    ns, ew, green = state
    if action == "SWITCH":
        green = 1 - green
    if green == 0:
        ns = max(0, ns - 2)     # NS is green, clears traffic
    else:
        ew = max(0, ew - 2)
    ns = min(MAX_Q, ns + 1) if ns < MAX_Q else ns    # arrivals (simplified, deterministic)
    ew = min(MAX_Q, ew + 1) if ew < MAX_Q else ew
    reward = -(ns + ew) - (1 if action == "SWITCH" else 0)   # small switching overhead
    return (ns, ew, green), reward
 
 
def policy_evaluation(policy, V, theta=1e-3):
    while True:
        delta = 0
        for s in STATES:
            a = policy[s]
            s2, r = transition(s, a)
            new_v = r + GAMMA * V[s2]
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v
        if delta < theta:
            break
    return V
 
 
def policy_improvement(V):
    policy = {}
    for s in STATES:
        best_a, best_v = None, float("-inf")
        for a in ACTIONS:
            s2, r = transition(s, a)
            v = r + GAMMA * V[s2]
            if v > best_v:
                best_v, best_a = v, a
        policy[s] = best_a
    return policy
 
 
def policy_iteration():
    V = {s: 0.0 for s in STATES}
    policy = {s: "KEEP" for s in STATES}
    for i in range(30):
        V = policy_evaluation(policy, V)
        new_policy = policy_improvement(V)
        if new_policy == policy:
            return policy, V, i + 1
        policy = new_policy
    return policy, V, 30
 
 
if __name__ == "__main__":
    policy, V, iterations = policy_iteration()
    print(f"Policy Iteration converged in {iterations} iterations.\n")
 
    print("Learned policy for sample congested states:")
    for s in [(4, 4, 0), (4, 0, 0), (0, 4, 0), (3, 3, 1)]:
        print(f"  state (NS={s[0]}, EW={s[1]}, green={'NS' if s[2]==0 else 'EW'}) -> action: {policy[s]}")
 
    state = (4, 4, 0)
    total_wait = 0
    print("\nSimulated 8-cycle run from a fully congested state:")
    for t in range(8):
        a = policy[state]
        state, r = transition(state, a)
        total_wait += -r
        print(f"  Cycle {t+1}: action={a:<7} -> NS={state[0]}, EW={state[1]}, green={'NS' if state[2]==0 else 'EW'}")
    print(f"Total accumulated waiting cost over 8 cycles: {total_wait}")
 
