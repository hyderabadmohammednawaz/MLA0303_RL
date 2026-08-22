"""
Experiment 32: Use Dueling DQN to train an agent for optimal
navigation in a complex gridworld and compare its performance with
standard DQN. The Dueling architecture decomposes Q(s,a) = V(s) +
A(s,a) - mean_a A(s,a); here V and A are learned as separate tabular
streams (equivalent to a neural network with two output heads).
"""
 
import random
from collections import defaultdict
 
random.seed(19)
 
GRID = 6
GOAL = (5, 5)
OBSTACLES = {(1, 1), (1, 2), (1, 3), (3, 1), (3, 3), (3, 4), (4, 1)}
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
ALPHA, GAMMA, EPISODES = 0.15, 0.9, 1200
 
 
def step(state, action):
    dr, dc = DELTA[action]
    nxt = (state[0] + dr, state[1] + dc)
    if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID) or nxt in OBSTACLES:
        return state, -5, False
    if nxt == GOAL:
        return nxt, 25, True
    return nxt, -1, False
 
 
def train_dqn(episodes=EPISODES):
    Q = defaultdict(lambda: [0.0] * len(ACTIONS))
    epsilon = 1.0
    rewards_hist = []
    for ep in range(episodes):
        s, total = (0, 0), 0
        for _ in range(80):
            a = random.randrange(4) if random.random() < epsilon else max(range(4), key=lambda i: Q[s][i])
            s2, r, done = step(s, ACTIONS[a])
            target = r + (0 if done else GAMMA * max(Q[s2]))
            Q[s][a] += ALPHA * (target - Q[s][a])
            s = s2
            total += r
            if done:
                break
        epsilon = max(0.05, epsilon * 0.995)
        rewards_hist.append(total)
    return Q, rewards_hist
 
 
def train_dueling_dqn(episodes=EPISODES):
    V = defaultdict(float)
    A = defaultdict(lambda: [0.0] * len(ACTIONS))
    epsilon = 1.0
    rewards_hist = []
    for ep in range(episodes):
        s, total = (0, 0), 0
        for _ in range(80):
            q_s = [V[s] + A[s][i] - sum(A[s]) / 4 for i in range(4)]
            a = random.randrange(4) if random.random() < epsilon else max(range(4), key=lambda i: q_s[i])
            s2, r, done = step(s, ACTIONS[a])
            q_s2 = [V[s2] + A[s2][i] - sum(A[s2]) / 4 for i in range(4)]
            target = r + (0 if done else GAMMA * max(q_s2))
            td_error = target - q_s[a]
            V[s] += ALPHA * td_error
            A[s][a] += ALPHA * td_error
            s = s2
            total += r
            if done:
                break
        epsilon = max(0.05, epsilon * 0.995)
        rewards_hist.append(total)
    return (V, A), rewards_hist
 
 
if __name__ == "__main__":
    Q, rewards_dqn = train_dqn()
    (V, A), rewards_dueling = train_dueling_dqn()
 
    print(f"{'Algorithm':<14}{'Avg reward (first 100 ep)':<28}{'Avg reward (last 100 ep)'}")
    print(f"{'DQN':<14}{sum(rewards_dqn[:100]) / 100:<28.2f}{sum(rewards_dqn[-100:]) / 100:.2f}")
    print(f"{'DuelingDQN':<14}{sum(rewards_dueling[:100]) / 100:<28.2f}{sum(rewards_dueling[-100:]) / 100:.2f}")
 
