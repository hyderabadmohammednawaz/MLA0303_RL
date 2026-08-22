"""
Experiment 40: Apply RL to personalize educational experiences for
students in an online learning platform. The agent chooses the
difficulty level of the next question/lesson to maximize student
learning gain while avoiding frustration (too hard) or boredom (too
easy). State = (student_skill_level, last_result); Actions = EASY,
MEDIUM, HARD content.
"""
 
import random
from collections import defaultdict
 
random.seed(26)
 
SKILL_LEVELS = 5          # 0 (novice) .. 4 (expert)
ACTIONS = ["EASY", "MEDIUM", "HARD"]
ALPHA, GAMMA, EPISODES = 0.15, 0.9, 2000
 
 
def simulate_answer(skill, difficulty):
    """Probability of a correct answer depends on how well the content
    difficulty matches the student's current skill level."""
    gap = difficulty - skill
    prob_correct = max(0.05, min(0.95, 0.8 - 0.3 * gap))
    return random.random() < prob_correct
 
 
def step(skill, action_idx, last_result):
    correct = simulate_answer(skill, action_idx)
    if correct and action_idx >= skill:
        skill = min(SKILL_LEVELS - 1, skill + 1)      # learning gain from an appropriate challenge
    reward = 0
    if correct:
        reward = 2 + action_idx                        # reward for succeeding, more for harder content
    else:
        reward = -3 if action_idx > skill + 1 else -1   # bigger penalty for frustration (too hard)
    return skill, int(correct), reward
 
 
def train():
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    epsilon = 1.0
    rewards_hist = []
    for ep in range(EPISODES):
        skill = random.randint(0, 2)
        last_result = 1
        total = 0
        for _ in range(20):
            state = (skill, last_result)
            action = random.choice(ACTIONS) if random.random() < epsilon else max(ACTIONS, key=lambda a: Q[state][a])
            action_idx = ACTIONS.index(action)
            next_skill, correct, reward = step(skill, action_idx, last_result)
            next_state = (next_skill, correct)
            best_next = max(Q[next_state].values())
            Q[state][action] += ALPHA * (reward + GAMMA * best_next - Q[state][action])
            skill, last_result = next_skill, correct
            total += reward
        epsilon = max(0.05, epsilon * 0.995)
        rewards_hist.append(total)
    return Q, rewards_hist
 
 
if __name__ == "__main__":
    Q, rewards = train()
    print(f"Average session reward (first 100 episodes): {sum(rewards[:100]) / 100:.2f}")
    print(f"Average session reward (last 100 episodes) : {sum(rewards[-100:]) / 100:.2f}")
 
    print("\nLearned content-selection policy (skill, last_result -> action):")
    for skill in range(SKILL_LEVELS):
        for last_result in (0, 1):
            state = (skill, last_result)
            action = max(ACTIONS, key=lambda a: Q[state][a])
            print(f"  skill={skill}, last_correct={last_result} -> {action}")
 
