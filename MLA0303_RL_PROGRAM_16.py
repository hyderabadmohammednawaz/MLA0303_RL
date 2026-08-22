"""
Experiment 16: Compare policy gradient algorithms (REINFORCE vs
Actor-Critic) for an autonomous lane-keeping system to improve
driving performance and stability.
State = lateral offset from lane center, discretized into 7 bins.
"""
 
import numpy as np
 
np.random.seed(4)
 
N_BINS = 7
CENTER = 3
ACTIONS = ["STEER_LEFT", "STEER_RIGHT", "HOLD"]
GAMMA = 0.95
EPISODES = 1200
 
 
class LaneEnv:
    def reset(self):
        self.offset = np.random.choice([1, 2, 4, 5])
        self.steps = 0
        return self.offset
 
    def step(self, action):
        if action == 0:
            self.offset = max(0, self.offset - 1)
        elif action == 1:
            self.offset = min(N_BINS - 1, self.offset + 1)
        self.steps += 1
        if self.offset in (0, N_BINS - 1):
            return self.offset, -15, True          # ran off the lane
        reward = 3 if self.offset == CENTER else -abs(self.offset - CENTER)
        done = self.steps >= 20
        return self.offset, reward, done
 
 
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
 
 
def reinforce(episodes=EPISODES, lr=0.05):
    theta = np.zeros((N_BINS, len(ACTIONS)))
    env = LaneEnv()
    baseline, rewards_hist = 0.0, []
    for ep in range(episodes):
        state = env.reset()
        traj = []
        for _ in range(25):
            probs = softmax(theta[state])
            action = np.random.choice(len(ACTIONS), p=probs)
            next_state, reward, done = env.step(action)
            traj.append((state, action, reward, probs))
            state = next_state
            if done:
                break
        G, returns = 0, [0] * len(traj)
        for t in reversed(range(len(traj))):
            G = traj[t][2] + GAMMA * G
            returns[t] = G
        ep_return = sum(r for _, _, r, _ in traj)
        baseline = 0.95 * baseline + 0.05 * ep_return
        for (s, a, r, probs), G_t in zip(traj, returns):
            adv = G_t - baseline
            grad = -probs
            grad[a] += 1
            theta[s] += lr * adv * grad
        rewards_hist.append(ep_return)
    return theta, rewards_hist
 
 
def actor_critic(episodes=EPISODES, lr_actor=0.05, lr_critic=0.1):
    theta = np.zeros((N_BINS, len(ACTIONS)))
    V = np.zeros(N_BINS)
    env = LaneEnv()
    rewards_hist = []
    for ep in range(episodes):
        state = env.reset()
        total = 0
        for _ in range(25):
            probs = softmax(theta[state])
            action = np.random.choice(len(ACTIONS), p=probs)
            next_state, reward, done = env.step(action)
            td_target = reward + (0 if done else GAMMA * V[next_state])
            td_error = td_target - V[state]
            V[state] += lr_critic * td_error
            grad = -probs
            grad[action] += 1
            theta[state] += lr_actor * td_error * grad
            state = next_state
            total += reward
            if done:
                break
        rewards_hist.append(total)
    return theta, rewards_hist
 
 
if __name__ == "__main__":
    theta_r, rewards_r = reinforce()
    theta_ac, rewards_ac = actor_critic()
 
    print(f"{'Algorithm':<14}{'Avg reward (first 100 ep)':<28}{'Avg reward (last 100 ep)'}")
    print(f"{'REINFORCE':<14}{np.mean(rewards_r[:100]):<28.2f}{np.mean(rewards_r[-100:]):.2f}")
    print(f"{'ActorCritic':<14}{np.mean(rewards_ac[:100]):<28.2f}{np.mean(rewards_ac[-100:]):.2f}")
 
