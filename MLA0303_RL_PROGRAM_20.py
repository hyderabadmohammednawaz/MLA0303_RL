"""
Experiment 20: Model a search-and-rescue robot using a Partially
Observable Markov Decision Process (POMDP). The robot cannot directly
observe which of several rooms the survivor is in; it maintains a
belief (probability distribution) over the survivor's location and
updates that belief using noisy sensor readings (Bayesian filtering),
then chooses the room to search that has the highest belief.
"""
 
import random
import numpy as np
 
random.seed(9)
np.random.seed(9)
 
ROOMS = ["Room_A", "Room_B", "Room_C", "Room_D"]
TRUE_SURVIVOR_ROOM = "Room_C"
SENSOR_ACCURACY = 0.7          # probability the sensor correctly detects the room
 
 
def get_noisy_observation(true_room):
    """The robot's sensor correctly reports the room with probability
    SENSOR_ACCURACY, otherwise reports a uniformly random wrong room."""
    if random.random() < SENSOR_ACCURACY:
        return true_room
    return random.choice([r for r in ROOMS if r != true_room])
 
 
def bayesian_belief_update(belief, observation):
    """P(room | obs) proportional to P(obs | room) * P(room)"""
    new_belief = {}
    for room in ROOMS:
        likelihood = SENSOR_ACCURACY if room == observation else (1 - SENSOR_ACCURACY) / (len(ROOMS) - 1)
        new_belief[room] = likelihood * belief[room]
    total = sum(new_belief.values())
    for room in ROOMS:
        new_belief[room] /= total
    return new_belief
 
 
def search_and_rescue(n_sensor_readings=6):
    belief = {room: 1 / len(ROOMS) for room in ROOMS}     # uniform prior
    print(f"Initial belief: { {r: round(v, 3) for r, v in belief.items()} }")
 
    for t in range(1, n_sensor_readings + 1):
        obs = get_noisy_observation(TRUE_SURVIVOR_ROOM)
        belief = bayesian_belief_update(belief, obs)
        print(f"Step {t}: sensor observation = {obs:<8} "
              f"-> belief = { {r: round(v, 3) for r, v in belief.items()} }")
 
    best_room = max(belief, key=belief.get)
    return best_room, belief
 
 
if __name__ == "__main__":
    best_room, final_belief = search_and_rescue()
    print(f"\nRobot decides to search: {best_room} "
          f"(belief = {final_belief[best_room]:.3f})")
    print(f"Actual survivor location was: {TRUE_SURVIVOR_ROOM}")
    print(f"Search successful: {best_room == TRUE_SURVIVOR_ROOM}")
 
