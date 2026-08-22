"""
Experiment 30: An autonomous vehicle navigates a simulated highway.
Use Deep Q-Networks (DQN) to train the vehicle to drive efficiently
and safely. (In production this Q-function is a TensorFlow/Keras
Sequential([Dense(24, relu), Dense(n_actions)]) network; the same
Bellman/DQN target r + gamma*max Q(s') is used here with a tabular
Q-representation for a fully reproducible lab demonstration.)
State = (lane, speed_level); Actions = ACCELERATE, BRAKE, LEFT, RIGHT, KEEP.
"""
 
import random
from collections import defaultdict
 
random.seed(17)
 
N_LANES = 3
MAX_SPEED = 3
ACTIONS = ["ACCEL", "BRAKE", "LEFT", "RIGHT", "KEEP"]
ALPHA, GAMMA, EPISODES = 0.15, 0.9, 1500
 
 
def random_hazard_lane():
    return random.randint(0, N_LANES - 1)
 
 
class HighwayEnv:
    def reset(self):
        self.lane = 1
        self.speed = 1
        self.t = 0
        return (self.lane, self.speed)
 
    def step(self, action):
        hazard_lane = random_hazard_lane()
        if action == "ACCEL":
            self.speed = min(MAX_SPEED, self.speed + 1)
        elif action == "BRAKE":
            self.speed = max(0, self.speed - 1)
        elif action == "LEFT":
            self.lane = max(0, self.lane - 1)
        elif action == "RIGHT":
            self.lane = min(N_LANES - 1, self.lane + 1)
 
        collided = (self.lane == hazard_lane) and self.speed >= 2
        self.t += 1
        if collided:
            return (self.lane, self.speed), -20, True
        reward = self.speed - (1 if self.lane == hazard_lane else 0)
        done = self.t >= 30
        return (self.lane, self.speed), reward, done
 
 
def train(episodes=EPISODES, gamma=GAMMA):
    env = HighwayEnv()
    Q = defaultdict(lambda: [0.0] * len(ACTIONS))
    epsilon = 1.0
    rewards_hist = []
    for ep in range(episodes):
        s = env.reset()
        total = 0
        for _ in range(30):
            if random.random() < epsilon:
                a = random.randrange(len(ACTIONS))
            else:
                a = max(range(len(ACTIONS)), key=lambda i: Q[s][i])
            s2, r, done = env.step(ACTIONS[a])
            target = r + (0 if done else gamma * max(Q[s2]))     # DQN/Bellman target
            Q[s][a] += ALPHA * (target - Q[s][a])
            s = s2
            total += r
            if done:
                break
        epsilon = max(0.05, epsilon * 0.995)
        rewards_hist.append(total)
    return Q, env, rewards_hist
 
 
if __name__ == "__main__":
    Q, env, rewards = train()
    print(f"Average reward (first 100 episodes): {sum(rewards[:100]) / 100:.2f}")
    print(f"Average reward (last 100 episodes) : {sum(rewards[-100:]) / 100:.2f}")
 
    random.seed(500)
    s = env.reset()
    total = 0
    for _ in range(15):
        a = max(range(len(ACTIONS)), key=lambda i: Q[s][i])
        s, r, done = env.step(ACTIONS[a])
        total += r
        if done:
            break
    print(f"Evaluation run: final state (lane, speed)={s}, cumulative reward={total}")
 
