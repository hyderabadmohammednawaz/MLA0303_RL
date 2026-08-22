"""
Experiment 25: An online platform uses bandit algorithms to decide
which advertisements to show. Implement epsilon-greedy, UCB, and
Thompson Sampling and determine which results in the highest
click-through rate over time.
"""
 
import random
import math
 
random.seed(15)
 
ADS = ["Ad_A", "Ad_B", "Ad_C", "Ad_D"]
TRUE_CTR = {"Ad_A": 0.05, "Ad_B": 0.12, "Ad_C": 0.09, "Ad_D": 0.15}
ROUNDS = 4000
 
 
def simulate_click(ad):
    return 1 if random.random() < TRUE_CTR[ad] else 0
 
 
def epsilon_greedy(epsilon=0.1, rounds=ROUNDS):
    Q = {ad: 0.0 for ad in ADS}
    N = {ad: 0 for ad in ADS}
    total_clicks = 0
    for t in range(1, rounds + 1):
        ad = random.choice(ADS) if random.random() < epsilon else max(Q, key=Q.get)
        r = simulate_click(ad)
        N[ad] += 1
        Q[ad] += (r - Q[ad]) / N[ad]
        total_clicks += r
    return total_clicks, Q, N
 
 
def ucb(c=2.0, rounds=ROUNDS):
    Q = {ad: 0.0 for ad in ADS}
    N = {ad: 0 for ad in ADS}
    total_clicks = 0
    for t in range(1, rounds + 1):
        if t <= len(ADS):
            ad = ADS[t - 1]                       # play each arm once first
        else:
            ad = max(ADS, key=lambda a: Q[a] + c * math.sqrt(math.log(t) / N[a]))
        r = simulate_click(ad)
        N[ad] += 1
        Q[ad] += (r - Q[ad]) / N[ad]
        total_clicks += r
    return total_clicks, Q, N
 
 
def thompson_sampling(rounds=ROUNDS):
    alpha = {ad: 1 for ad in ADS}    # Beta distribution success count
    beta = {ad: 1 for ad in ADS}     # Beta distribution failure count
    total_clicks = 0
    for t in range(1, rounds + 1):
        samples = {ad: random.betavariate(alpha[ad], beta[ad]) for ad in ADS}
        ad = max(samples, key=samples.get)
        r = simulate_click(ad)
        if r == 1:
            alpha[ad] += 1
        else:
            beta[ad] += 1
        total_clicks += r
    est_ctr = {ad: alpha[ad] / (alpha[ad] + beta[ad]) for ad in ADS}
    return total_clicks, est_ctr
 
 
if __name__ == "__main__":
    random.seed(15)
    eg_clicks, eg_Q, eg_N = epsilon_greedy()
    random.seed(15)
    ucb_clicks, ucb_Q, ucb_N = ucb()
    random.seed(15)
    ts_clicks, ts_ctr = thompson_sampling()
 
    print(f"{'Algorithm':<18}{'Total Clicks':<16}{'Achieved CTR'}")
    print(f"{'Epsilon-Greedy':<18}{eg_clicks:<16}{eg_clicks / ROUNDS:.4f}")
    print(f"{'UCB':<18}{ucb_clicks:<16}{ucb_clicks / ROUNDS:.4f}")
    print(f"{'Thompson Sampling':<18}{ts_clicks:<16}{ts_clicks / ROUNDS:.4f}")
 
    best = max(
        [("Epsilon-Greedy", eg_clicks), ("UCB", ucb_clicks), ("Thompson Sampling", ts_clicks)],
        key=lambda x: x[1],
    )
    print(f"\nBest performing algorithm: {best[0]} with {best[1]} total clicks")
 
