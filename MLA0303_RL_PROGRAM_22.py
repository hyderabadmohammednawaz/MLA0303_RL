"""
Experiment 22: Implement Q-learning to develop an AI agent that plays
a simple grid-based game (a basic version of Pac-Man). The agent
learns to collect rewards (food) and avoid penalties (a moving ghost).
"""
 
import random
from collections import defaultdict
 
random.seed(12)
 
GRID = 5
FOOD = {(0, 4), (4, 0), (2, 2)}
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
ALPHA, GAMMA, EPISODES = 0.15, 0.9, 2000
 
 
def move(pos, action):
    dr, dc = DELTA[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
        return pos
    return nxt
 
 
def ghost_move(ghost):
    return move(ghost, random.choice(ACTIONS))
 
 
class PacManEnv:
    def reset(self):
        self.pacman = (0, 0)
        self.ghost = (4, 4)
        self.food = set(FOOD)
        return self._state()
 
    def _state(self):
        return (self.pacman, self.ghost, frozenset(self.food))
 
    def step(self, action):
        self.pacman = move(self.pacman, action)
        reward = -1
        if self.pacman in self.food:
            self.food.remove(self.pacman)
            reward = 10
        self.ghost = ghost_move(self.ghost)
        done = False
        if self.pacman == self.ghost:
            reward = -20
            done = True
        elif len(self.food) == 0:
            reward += 20
            done = True
        return self._state(), reward, done
 
 
def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: Q[state][a])
 
 
def train():
    env = PacManEnv()
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    epsilon = 1.0
    rewards_hist = []
    for ep in range(EPISODES):
        state = env.reset()
        total = 0
        for _ in range(60):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            best_next = max(Q[next_state].values())
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
            state = next_state
            total += reward
            if done:
                break
        epsilon = max(0.05, epsilon * 0.998)
        rewards_hist.append(total)
    return Q, env, rewards_hist
 
 
if __name__ == "__main__":
    Q, env, rewards = train()
    print(f"Average score (first 100 episodes): {sum(rewards[:100]) / 100:.2f}")
    print(f"Average score (last 100 episodes) : {sum(rewards[-100:]) / 100:.2f}")
 
    # Play one evaluation game and print outcome
    state = env.reset()
    for t in range(40):
        action = max(ACTIONS, key=lambda a: Q[state][a])
        state, reward, done = env.step(action)
        if done:
            outcome = "Pac-Man WON (all food collected)!" if reward > 0 else "Pac-Man was caught by the ghost."
            print(f"\nEvaluation game finished in {t + 1} moves. {outcome}")
            break
 
