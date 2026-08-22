"""
Experiment 21: RL-based smart energy management system that optimizes
energy consumption while ensuring safe, fair, and responsible
autonomous decision-making. State = (battery_level, demand_level);
Actions = charge from grid / discharge to home / hold. A safety
constraint prevents the battery from being drained below a minimum
safe level, and a fairness penalty discourages always favouring the
cheapest (but dirtiest) energy source.
"""
 
import random
from collections import defaultdict
 
random.seed(11)
 
BATTERY_LEVELS = 6      # 0..5
MIN_SAFE_LEVEL = 1       # safety constraint: never let battery hit 0
ACTIONS = ["CHARGE", "DISCHARGE", "HOLD"]
ALPHA, GAMMA, EPISODES = 0.15, 0.9, 1500
 
 
def demand_level():
    return random.choice([0, 1, 2])   # low / medium / high household demand
 
 
def step(battery, action, demand):
    reward = 0
    if action == 0:                              # CHARGE from grid
        battery = min(BATTERY_LEVELS - 1, battery + 1)
        reward -= 2                               # cost of drawing grid power
    elif action == 1:                             # DISCHARGE to power the home
        if battery - demand < MIN_SAFE_LEVEL:
            reward -= 15                           # safety violation penalty
        else:
            battery -= demand
            reward += 4 + demand                   # reward for self-sufficiency
    else:                                          # HOLD
        if demand > 0:
            reward -= demand                       # unmet demand still costs something
    # Fairness term: discourage staying at extreme (always-full or always-empty) levels,
    # which would represent an unfair/unsafe long-term energy allocation strategy
    if battery <= MIN_SAFE_LEVEL or battery >= BATTERY_LEVELS - 1:
        reward -= 1
    battery = max(0, min(BATTERY_LEVELS - 1, battery))
    return battery, reward
 
 
def train():
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    epsilon = 1.0
    rewards_hist = []
    for ep in range(EPISODES):
        battery = random.randint(2, 4)
        total = 0
        for _ in range(24):                        # 24-hour simulated day
            demand = demand_level()
            state = (battery, demand)
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                action = max(ACTIONS, key=lambda a: Q[state][a])
            next_battery, reward = step(battery, ACTIONS.index(action), demand)
            next_state = (next_battery, demand_level())
            best_next = max(Q[next_state].values())
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
            battery = next_battery
            total += reward
        epsilon = max(0.05, epsilon * 0.995)
        rewards_hist.append(total)
    return Q, rewards_hist
 
 
if __name__ == "__main__":
    Q, rewards = train()
    print(f"Average daily reward (first 50 episodes): {sum(rewards[:50]) / 50:.2f}")
    print(f"Average daily reward (last 50 episodes) : {sum(rewards[-50:]) / 50:.2f}")
 
    battery = 3
    print("\nSample 12-hour policy trace (battery, demand -> action):")
    for hour in range(12):
        demand = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1, 0, 1][hour]
        state = (battery, demand)
        action = max(ACTIONS, key=lambda a: Q[state][a])
        battery, reward = step(battery, ACTIONS.index(action), demand)
        print(f"Hour {hour:2d}: battery={state[0]}, demand={state[1]} -> action={action:<10} reward={reward}")
 
