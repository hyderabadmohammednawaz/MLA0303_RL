"""
Experiment 18: Meta-Reinforcement Learning for an adaptive industrial
robot that must quickly learn new manufacturing tasks (different
target assembly-slots on a work bench). A Reptile-style meta-learning
update is used: the robot practices Q-learning on many randomly
sampled tasks, and after each task nudges a shared "meta" Q-table
towards the task-adapted Q-table, so that when a brand-new task
arrives only a handful of fine-tuning steps are needed to reach it,
compared to learning entirely from scratch (zero-initialized Q-table).
"""
 
import random
from collections import defaultdict
 
N_SLOTS = 6                      # possible target assembly slots (states)
ACTIONS = ["LEFT", "RIGHT"]
ALPHA, GAMMA = 0.3, 0.9
META_LR = 0.4
META_ITERATIONS = 400
INNER_EPISODES = 20
 
 
def step(pos, action, target):
    if action == 0:
        pos = max(0, pos - 1)
    else:
        pos = min(N_SLOTS - 1, pos + 1)
    if pos == target:
        return pos, 10, True
    return pos, -1, False
 
 
def q_dict_from(source_Q):
    return defaultdict(lambda: [0.0] * len(ACTIONS), {s: list(v) for s, v in source_Q.items()})
 
 
def train_task(init_Q, target, episodes, epsilon=0.2):
    """Q-learning on one task (one target slot), starting from init_Q."""
    Q = q_dict_from(init_Q)
    for _ in range(episodes):
        pos = random.randint(0, N_SLOTS - 1)
        for _ in range(N_SLOTS + 2):
            if random.random() < epsilon:
                a = random.randrange(len(ACTIONS))
            else:
                a = max(range(len(ACTIONS)), key=lambda i: Q[pos][i])
            nxt, r, done = step(pos, a, target)
            best_next = 0 if done else max(Q[nxt])
            Q[pos][a] += ALPHA * (r + GAMMA * best_next - Q[pos][a])
            pos = nxt
            if done:
                break
    return Q
 
 
def meta_train():
    meta_Q = defaultdict(lambda: [0.0] * len(ACTIONS))
    for it in range(META_ITERATIONS):
        target = random.randint(0, N_SLOTS - 1)
        adapted_Q = train_task(meta_Q, target, episodes=INNER_EPISODES)
        for s in range(N_SLOTS):
            for a in range(len(ACTIONS)):
                meta_Q[s][a] += META_LR * (adapted_Q[s][a] - meta_Q[s][a])   # Reptile update
    return meta_Q
 
 
def evaluate(Q, target, max_steps=N_SLOTS + 2):
    pos = 0
    path = [pos]
    for _ in range(max_steps):
        a = max(range(len(ACTIONS)), key=lambda i: Q[pos][i])
        pos, r, done = step(pos, a, target)
        path.append(pos)
        if done:
            return path, r
    return path, None
 
 
if __name__ == "__main__":
    random.seed(7)
    meta_Q = meta_train()
 
    print("Comparing adaptation speed on a brand-new task (target slot 5):\n")
    for label, init_Q, fine_tune_episodes in [
        ("From scratch (zero-init)", defaultdict(lambda: [0.0] * len(ACTIONS)), 2),
        ("Meta-initialized (Reptile)", meta_Q, 2),
    ]:
        random.seed(42)
        adapted_Q = train_task(init_Q, target=5, episodes=fine_tune_episodes, epsilon=0.1)
        path, reward = evaluate(adapted_Q, target=5)
        print(f"{label:<28} | after {fine_tune_episodes} fine-tune episodes -> "
              f"path={path}, reward={reward}")
 
    print("\nComparing on a second new task (target slot 2):\n")
    for label, init_Q, fine_tune_episodes in [
        ("From scratch (zero-init)", defaultdict(lambda: [0.0] * len(ACTIONS)), 2),
        ("Meta-initialized (Reptile)", meta_Q, 2),
    ]:
        random.seed(43)
        adapted_Q = train_task(init_Q, target=2, episodes=fine_tune_episodes, epsilon=0.1)
        path, reward = evaluate(adapted_Q, target=2)
        print(f"{label:<28} | after {fine_tune_episodes} fine-tune episodes -> "
              f"path={path}, reward={reward}")
 
