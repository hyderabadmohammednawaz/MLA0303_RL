"""
Experiment 38: An autonomous robot navigates through a partially
observable environment with limited sensor information. Implement a
POMDP framework (belief-state tracking with a particle filter) so the
robot can localize itself and plan navigation actions robustly under
partial observability.
"""
 
import random
 
random.seed(24)
 
GRID = 6
TRUE_START = (2, 2)
GOAL = (5, 5)
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
N_PARTICLES = 300
SENSOR_NOISE = 0.3     # probability the sensor reading is off by one cell
 
 
def move(pos, action):
    dr, dc = DELTA[action]
    nxt = (pos[0] + dr, pos[1] + dc)
    if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
        return pos
    return nxt
 
 
def noisy_sensor_reading(true_pos):
    if random.random() > SENSOR_NOISE:
        return true_pos
    dr, dc = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    noisy = (true_pos[0] + dr, true_pos[1] + dc)
    if 0 <= noisy[0] < GRID and 0 <= noisy[1] < GRID:
        return noisy
    return true_pos
 
 
def init_particles():
    return [(random.randint(0, GRID - 1), random.randint(0, GRID - 1)) for _ in range(N_PARTICLES)]
 
 
def predict_particles(particles, action):
    return [move(p, action) for p in particles]
 
 
def update_particles(particles, observation):
    """Weight particles by closeness to the sensor reading, then resample
    (this is the particle-filter belief update for the POMDP)."""
    weights = []
    for p in particles:
        dist = abs(p[0] - observation[0]) + abs(p[1] - observation[1])
        weights.append(max(0.01, 1.0 - 0.3 * dist))
    total = sum(weights)
    weights = [w / total for w in weights]
    resampled = random.choices(particles, weights=weights, k=len(particles))
    return resampled
 
 
def belief_estimate(particles):
    from collections import Counter
    counts = Counter(particles)
    return counts.most_common(1)[0][0]
 
 
def choose_action_towards(estimated_pos, goal):
    if estimated_pos[0] < goal[0]:
        return "DOWN"
    if estimated_pos[0] > goal[0]:
        return "UP"
    if estimated_pos[1] < goal[1]:
        return "RIGHT"
    if estimated_pos[1] > goal[1]:
        return "LEFT"
    return "UP"
 
 
if __name__ == "__main__":
    true_pos = TRUE_START
    particles = init_particles()
 
    print("Step | True Position | Sensor Reading | Belief Estimate")
    for step in range(10):
        obs = noisy_sensor_reading(true_pos)
        particles = update_particles(particles, obs)
        estimate = belief_estimate(particles)
        print(f"{step:4d} | {str(true_pos):<14}| {str(obs):<15}| {estimate}")
 
        if true_pos == GOAL:
            print("\nRobot has reached the goal.")
            break
        action = choose_action_towards(estimate, GOAL)
        true_pos = move(true_pos, action)
        particles = predict_particles(particles, action)
 
    print(f"\nFinal true position: {true_pos}, Goal: {GOAL}, "
          f"Reached: {true_pos == GOAL}")
 
