"""
Experiment 19: Multi-Agent Reinforcement Learning (MARL) for a
multi-robot warehouse system to optimize cooperative task allocation
and navigation. Two independent Q-learning robots share a grid and
must each pick up a different item and deliver it to the packing
station without colliding.
"""
 
import random
from collections import defaultdict
 
random.seed(8)
 
GRID = 5
ITEMS = {"R1": (0, 4), "R2": (4, 0)}
STATION = (2, 2)
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
ALPHA, GAMMA, EPISODES = 0.15, 0.9, 1500
 
 
class WarehouseMARL:
    def reset(self):
        self.pos = {"R1": (0, 0), "R2": (4, 4)}
        self.carrying = {"R1": False, "R2": False}
        return self._states()
 
    def _states(self):
        return {r: (self.pos[r], self.carrying[r]) for r in self.pos}
 
    def step(self, actions):
        rewards = {}
        proposed = {}
        for r, a in actions.items():
            dr, dc = DELTA[a]
            nxt = (self.pos[r][0] + dr, self.pos[r][1] + dc)
            if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
                nxt = self.pos[r]
            proposed[r] = nxt
 
        # Collision avoidance: if both robots propose the same cell, both stay
        if proposed["R1"] == proposed["R2"]:
            proposed["R1"], proposed["R2"] = self.pos["R1"], self.pos["R2"]
 
        done = {}
        for r in self.pos:
            self.pos[r] = proposed[r]
            reward = -1
            if not self.carrying[r] and self.pos[r] == ITEMS[r]:
                self.carrying[r] = True
                reward = 5
            elif self.carrying[r] and self.pos[r] == STATION:
                reward = 15
                done[r] = True
            rewards[r] = reward
            done.setdefault(r, False)
        return self._states(), rewards, done
 
 
def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: Q[state][a])
 
 
def train():
    env = WarehouseMARL()
    Q = {r: defaultdict(lambda: {a: 0.0 for a in ACTIONS}) for r in ITEMS}
    epsilon = 1.0
    episode_rewards = []
    for ep in range(EPISODES):
        states = env.reset()
        total = {r: 0 for r in ITEMS}
        for _ in range(40):
            actions = {r: epsilon_greedy(Q[r], states[r], epsilon) for r in ITEMS}
            next_states, rewards, done = env.step(actions)
            for r in ITEMS:
                best_next = max(Q[r][next_states[r]].values())
                Q[r][states[r]][actions[r]] += ALPHA * (
                    rewards[r] + GAMMA * best_next - Q[r][states[r]][actions[r]]
                )
                total[r] += rewards[r]
            states = next_states
            if all(done.values()):
                break
        epsilon = max(0.05, epsilon * 0.997)
        episode_rewards.append(sum(total.values()))
    return Q, env, episode_rewards
 
 
if __name__ == "__main__":
    Q, env, rewards = train()
    print(f"Average combined reward (first 50 episodes): {sum(rewards[:50]) / 50:.2f}")
    print(f"Average combined reward (last 50 episodes) : {sum(rewards[-50:]) / 50:.2f}")
 
    states = env.reset()
    paths = {r: [env.pos[r]] for r in ITEMS}
    for _ in range(20):
        actions = {r: max(ACTIONS, key=lambda a: Q[r][states[r]][a]) for r in ITEMS}
        states, rewards, done = env.step(actions)
        for r in ITEMS:
            paths[r].append(env.pos[r])
        if all(done.values()):
            break
    for r in ITEMS:
        print(f"{r} path: {paths[r]}")
 
