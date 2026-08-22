"""
Experiment 23: Autonomous vehicle that changes lanes on a highway to
avoid slower traffic and reach its destination faster, using PPO
(clipped surrogate objective) to optimize lane-changing decisions.
State = (lane, obstacle_pattern_index). Actions = STAY, LEFT, RIGHT.
"""
 
import numpy as np
 
np.random.seed(13)
 
N_LANES = 3
ACTIONS = ["STAY", "LEFT", "RIGHT"]
GAMMA = 0.95
EPISODES = 1500
ROAD_LENGTH = 10
 
 
def random_traffic_row():
    """1 = slow traffic blocking that lane, 0 = clear lane."""
    row = [0, 0, 0]
    row[np.random.randint(3)] = 1
    return row
 
 
class HighwayEnv:
    def reset(self):
        self.lane = 1
        self.step_idx = 0
        self.traffic = [random_traffic_row() for _ in range(ROAD_LENGTH)]
        return self._state()
 
    def _state(self):
        row = tuple(self.traffic[self.step_idx]) if self.step_idx < ROAD_LENGTH else (0, 0, 0)
        return (self.lane, row)
 
    def step(self, action):
        if action == 1:
            self.lane = max(0, self.lane - 1)
        elif action == 2:
            self.lane = min(N_LANES - 1, self.lane + 1)
        blocked = self.traffic[self.step_idx][self.lane] == 1
        reward = -5 if blocked else 2
        self.step_idx += 1
        done = self.step_idx >= ROAD_LENGTH
        if done:
            reward += 10
        return self._state(), reward, done
 
 
def state_key(s):
    return (s[0], s[1])
 
 
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
 
 
def collect_episode(theta, env):
    state = env.reset()
    traj = []
    for _ in range(ROAD_LENGTH):
        key = state_key(state)
        probs = softmax(theta[key]) if key in theta else softmax(np.zeros(3))
        if key not in theta:
            theta[key] = np.zeros(3)
        action = np.random.choice(3, p=probs)
        next_state, reward, done = env.step(action)
        traj.append((key, action, reward, probs.copy()))
        state = next_state
        if done:
            break
    return traj
 
 
def discounted_returns(traj):
    G, returns = 0, [0] * len(traj)
    for t in reversed(range(len(traj))):
        G = traj[t][2] + GAMMA * G
        returns[t] = G
    return returns
 
 
def train_ppo(episodes=EPISODES, clip_eps=0.2, lr=0.05):
    theta = {}
    env = HighwayEnv()
    baseline = 0.0
    rewards_hist = []
    for ep in range(episodes):
        traj = collect_episode(theta, env)
        returns = discounted_returns(traj)
        ep_return = sum(r for _, _, r, _ in traj)
        baseline = 0.95 * baseline + 0.05 * ep_return
        for (key, a, r, old_probs), G_t in zip(traj, returns):
            advantage = G_t - baseline
            new_probs = softmax(theta[key])
            ratio = new_probs[a] / (old_probs[a] + 1e-8)
            clipped_ratio = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
            surrogate = min(ratio * advantage, clipped_ratio * advantage)
            grad = -new_probs
            grad[a] += 1
            theta[key] += lr * surrogate * grad
        rewards_hist.append(ep_return)
    return theta, env, rewards_hist
 
 
if __name__ == "__main__":
    theta, env, rewards = train_ppo()
    print(f"Average return (first 100 episodes): {np.mean(rewards[:100]):.2f}")
    print(f"Average return (last 100 episodes) : {np.mean(rewards[-100:]):.2f}")
 
    state = env.reset()
    lanes_taken = [env.lane]
    total_reward = 0
    for _ in range(ROAD_LENGTH):
        key = state_key(state)
        action = int(np.argmax(theta[key])) if key in theta else 0
        state, reward, done = env.step(action)
        lanes_taken.append(env.lane)
        total_reward += reward
        if done:
            break
    print("Evaluation lane sequence:", lanes_taken)
    print("Evaluation total reward:", total_reward)
 
