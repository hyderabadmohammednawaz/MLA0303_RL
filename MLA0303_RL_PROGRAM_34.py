"""
Experiment 34: Model a smart home system that adjusts heating and
cooling to maintain comfort while minimizing energy usage, using the
REINFORCE algorithm to optimize temperature settings.
State = discretized current room temperature (bins 0-9, representing
16C-25C); Actions = HEAT, COOL, OFF.
"""
 
import numpy as np
 
np.random.seed(21)
 
N_BINS = 10
COMFORT_BIN = 5           # corresponds to ~21C
ACTIONS = ["HEAT", "COOL", "OFF"]
GAMMA = 0.95
EPISODES = 2000
 
 
class ThermostatEnv:
    def reset(self):
        self.temp_bin = np.random.choice([2, 3, 7, 8])
        self.t = 0
        return self.temp_bin
 
    def step(self, action):
        energy_cost = 0
        if action == 0:                   # HEAT
            self.temp_bin = min(N_BINS - 1, self.temp_bin + 1)
            energy_cost = 2
        elif action == 1:                 # COOL
            self.temp_bin = max(0, self.temp_bin - 1)
            energy_cost = 2
        else:                             # OFF: drifts toward outside temp (bin 8, "hot outside")
            self.temp_bin += 1 if self.temp_bin < 8 else 0
            energy_cost = 0
 
        comfort_penalty = abs(self.temp_bin - COMFORT_BIN)
        reward = -comfort_penalty - energy_cost * 0.5
        if self.temp_bin == COMFORT_BIN:
            reward += 3
        self.t += 1
        done = self.t >= 20
        return self.temp_bin, reward, done
 
 
theta = np.zeros((N_BINS, len(ACTIONS)))
 
 
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
 
 
def train():
    env = ThermostatEnv()
    baseline = 0.0
    rewards_hist = []
    for ep in range(EPISODES):
        state = env.reset()
        traj = []
        for _ in range(20):
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
            theta[s] += 0.05 * adv * grad
        rewards_hist.append(ep_return)
    return env, rewards_hist
 
 
if __name__ == "__main__":
    env, rewards = train()
    print(f"Average return (first 100 episodes): {np.mean(rewards[:100]):.2f}")
    print(f"Average return (last 100 episodes) : {np.mean(rewards[-100:]):.2f}")
 
    for start_bin in [2, 8]:
        env.temp_bin, env.t = start_bin, 0
        state = env.temp_bin
        path = [state]
        for _ in range(10):
            action = int(np.argmax(theta[state]))
            state, _, done = env.step(action)
            path.append(state)
            if done:
                break
        print(f"Starting temp bin {start_bin} -> trajectory: {path}")
 
