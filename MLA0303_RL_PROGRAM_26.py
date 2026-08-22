"""
Experiment 26: An online retailer uses a multi-armed bandit approach
to set prices dynamically. Simulate epsilon-greedy, UCB, and Thompson
Sampling pricing strategies and compare which maximizes revenue.
"""
 
import random
import math
 
random.seed(16)
 
PRICES = [9.99, 14.99, 19.99, 24.99]
# Higher price -> lower purchase probability, but higher revenue per sale
TRUE_BUY_PROB = {9.99: 0.40, 14.99: 0.30, 19.99: 0.22, 24.99: 0.12}
ROUNDS = 4000
 
 
def simulate_purchase(price):
    bought = random.random() < TRUE_BUY_PROB[price]
    revenue = price if bought else 0.0
    return revenue
 
 
def epsilon_greedy(epsilon=0.1, rounds=ROUNDS):
    Q = {p: 0.0 for p in PRICES}
    N = {p: 0 for p in PRICES}
    total_revenue = 0.0
    for t in range(1, rounds + 1):
        price = random.choice(PRICES) if random.random() < epsilon else max(Q, key=Q.get)
        rev = simulate_purchase(price)
        N[price] += 1
        Q[price] += (rev - Q[price]) / N[price]
        total_revenue += rev
    return total_revenue, Q
 
 
def ucb(c=2.0, rounds=ROUNDS):
    Q = {p: 0.0 for p in PRICES}
    N = {p: 0 for p in PRICES}
    total_revenue = 0.0
    for t in range(1, rounds + 1):
        if t <= len(PRICES):
            price = PRICES[t - 1]
        else:
            price = max(PRICES, key=lambda p: Q[p] + c * math.sqrt(math.log(t) / N[p]))
        rev = simulate_purchase(price)
        N[price] += 1
        Q[price] += (rev - Q[price]) / N[price]
        total_revenue += rev
    return total_revenue, Q
 
 
def thompson_sampling(rounds=ROUNDS):
    """Beta-Bernoulli Thompson Sampling on purchase probability; expected
    revenue is then probability * price."""
    alpha = {p: 1 for p in PRICES}
    beta = {p: 1 for p in PRICES}
    total_revenue = 0.0
    for t in range(1, rounds + 1):
        samples = {p: random.betavariate(alpha[p], beta[p]) * p for p in PRICES}
        price = max(samples, key=samples.get)
        rev = simulate_purchase(price)
        if rev > 0:
            alpha[price] += 1
        else:
            beta[price] += 1
        total_revenue += rev
    return total_revenue, {p: alpha[p] / (alpha[p] + beta[p]) for p in PRICES}
 
 
if __name__ == "__main__":
    random.seed(16)
    eg_rev, eg_Q = epsilon_greedy()
    random.seed(16)
    ucb_rev, ucb_Q = ucb()
    random.seed(16)
    ts_rev, ts_prob = thompson_sampling()
 
    print(f"{'Strategy':<18}{'Total Revenue'}")
    print(f"{'Epsilon-Greedy':<18}{eg_rev:.2f}")
    print(f"{'UCB':<18}{ucb_rev:.2f}")
    print(f"{'Thompson Sampling':<18}{ts_rev:.2f}")
 
    best = max([("Epsilon-Greedy", eg_rev), ("UCB", ucb_rev), ("Thompson Sampling", ts_rev)], key=lambda x: x[1])
    print(f"\nBest revenue-maximizing strategy: {best[0]} (${best[1]:.2f})")
 
