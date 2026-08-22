"""
Experiment 39: Apply RL to optimize healthcare management (patient
scheduling and resource allocation). A hospital ward has a limited
number of beds; the agent decides whether to ADMIT or DEFER each
arriving patient to maximize patient outcomes while minimizing
resource overuse and unmet urgent demand.
State = (free_beds, incoming_patient_urgency); Actions = ADMIT, DEFER.
"""
 
import random
from collections import defaultdict
 
random.seed(25)
 
MAX_BEDS = 5
ACTIONS = ["ADMIT", "DEFER"]
ALPHA, GAMMA, EPISODES = 0.15, 0.9, 2000
 
 
def patient_urgency():
    return random.choice([0, 1, 2])   # 0=low, 1=medium, 2=high (emergency)
 
 
def discharge_event():
    return random.random() < 0.3       # probability a bed frees up each step
 
 
def step(free_beds, action, urgency):
    reward = 0
    if action == 0:                    # ADMIT
        if free_beds > 0:
            free_beds -= 1
            reward = 5 + urgency * 3     # treating higher-urgency patients is more valuable
        else:
            reward = -10 - urgency * 5   # turning away a patient with no capacity: costly, worse if urgent
    else:                               # DEFER
        reward = -urgency * 4            # deferring urgent patients is risky
    if discharge_event() and free_beds < MAX_BEDS:
        free_beds += 1
    return free_beds, reward
 
 
def train():
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    epsilon = 1.0
    rewards_hist = []
    for ep in range(EPISODES):
        free_beds = MAX_BEDS
        total = 0
        for _ in range(30):
            urgency = patient_urgency()
            state = (free_beds, urgency)
            action = random.choice(ACTIONS) if random.random() < epsilon else max(ACTIONS, key=lambda a: Q[state][a])
            next_free, reward = step(free_beds, ACTIONS.index(action), urgency)
            next_state = (next_free, patient_urgency())
            best_next = max(Q[next_state].values())
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
            free_beds = next_free
            total += reward
        epsilon = max(0.05, epsilon * 0.995)
        rewards_hist.append(total)
    return Q, rewards_hist
 
 
if __name__ == "__main__":
    Q, rewards = train()
    print(f"Average daily reward (first 100 episodes): {sum(rewards[:100]) / 100:.2f}")
    print(f"Average daily reward (last 100 episodes) : {sum(rewards[-100:]) / 100:.2f}")
 
    print("\nLearned admission policy:")
    for free_beds in range(MAX_BEDS + 1):
        for urgency in range(3):
            state = (free_beds, urgency)
            action = max(ACTIONS, key=lambda a: Q[state][a])
            print(f"  free_beds={free_beds}, urgency={urgency} -> {action}")
 
