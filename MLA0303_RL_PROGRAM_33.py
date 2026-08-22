"""
Experiment 33: Develop an AI agent for a real-time strategy (RTS)
game to gather resources, build units, and defeat opponents, using
DDPG (Deep Deterministic Policy Gradient) to train the agent.
Since DDPG is designed for continuous action spaces, the action here
is a continuous "resource-allocation fraction" a in [0,1] deciding
what fraction of gathered resources to spend on building military
units versus economy; the actor is a small linear-Gaussian policy and
the critic is a linear Q-function, updated with the DDPG-style
deterministic policy gradient.
"""
 
import numpy as np
 
np.random.seed(20)
 
EPISODES = 800
GAMMA = 0.95
LR_ACTOR, LR_CRITIC = 0.02, 0.05
 
 
class RTSEnv:
    """State = (resources, army_strength, opponent_strength)"""
    def reset(self):
        self.resources = 10.0
        self.army = 1.0
        self.opponent = 1.0
        self.t = 0
        return np.array([self.resources, self.army, self.opponent])
 
    def step(self, action):
        action = float(np.clip(action, 0.0, 1.0))
        gathered = 5.0
        self.resources += gathered
        spend_on_army = action * self.resources
        self.army += spend_on_army * 0.1
        self.resources -= spend_on_army
        self.opponent += np.random.uniform(0.3, 0.7)     # opponent grows independently
        self.t += 1
        reward = (self.army - self.opponent) * 0.5
        done = self.t >= 20
        return np.array([self.resources, self.army, self.opponent]), reward, done
 
 
actor_w = np.random.randn(3) * 0.1     # linear deterministic policy: a = sigmoid(w . s)
critic_w = np.random.randn(4) * 0.1    # linear critic: Q(s,a) = w . [s, a]
 
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
 
def policy(state):
    return sigmoid(np.dot(actor_w, state) / 20.0)     # scale state for stability
 
 
def critic(state, action):
    feat = np.append(state / 20.0, action)
    return np.dot(critic_w, feat)
 
 
def train():
    env = RTSEnv()
    rewards_hist = []
    for ep in range(EPISODES):
        s = env.reset()
        total = 0
        for _ in range(20):
            a = policy(s)
            noise = np.random.normal(0, 0.1)             # exploration noise (DDPG-style)
            a_explore = float(np.clip(a + noise, 0, 1))
            s2, r, done = env.step(a_explore)
 
            a2 = policy(s2)
            target = r + (0 if done else GAMMA * critic(s2, a2))
            td_error = target - critic(s, a_explore)
 
            feat = np.append(s / 20.0, a_explore)
            critic_w[:] += LR_CRITIC * td_error * feat      # critic update
 
            grad_a = a * (1 - a) * (s / 20.0)                # d(action)/d(actor_w)
            actor_w[:] += LR_ACTOR * critic_w[-1] * grad_a    # deterministic policy gradient
 
            s = s2
            total += r
            if done:
                break
        rewards_hist.append(total)
    return env, rewards_hist
 
 
if __name__ == "__main__":
    env, rewards = train()
    print(f"Average episode reward (first 50 episodes): {np.mean(rewards[:50]):.2f}")
    print(f"Average episode reward (last 50 episodes) : {np.mean(rewards[-50:]):.2f}")
 
    s = env.reset()
    print("\nEvaluation run (resources, army, opponent) and chosen allocation:")
    for t in range(10):
        a = policy(s)
        s, r, done = env.step(a)
        print(f"  Turn {t + 1}: alloc-to-army={a:.2f} -> state={np.round(s, 2)}, reward={r:.2f}")
        if done:
            break
 
