"""
Experiment 35: Implement a value-equivalence prediction model to
estimate the long-term performance of different investment
portfolios. TD(0) value-function learning is used to predict the
expected long-term (discounted) return of each portfolio allocation
strategy from historical-style simulated returns.
"""
 
import random
 
random.seed(22)
 
PORTFOLIOS = ["Conservative", "Balanced", "Aggressive"]
# Monthly return distribution parameters (mean %, std %) per portfolio
RETURN_PARAMS = {
    "Conservative": (0.3, 0.5),
    "Balanced": (0.6, 1.2),
    "Aggressive": (1.0, 2.5),
}
N_MONTHS = 24
ALPHA, GAMMA = 0.1, 0.95
EPISODES = 3000
 
 
def simulate_return(portfolio):
    mean, std = RETURN_PARAMS[portfolio]
    return random.gauss(mean, std)
 
 
def return_bin(monthly_return):
    if monthly_return < -1:
        return 0
    if monthly_return < 0:
        return 1
    if monthly_return < 1:
        return 2
    return 3
 
 
def td0_value_prediction(portfolio, episodes=EPISODES):
    """Predicts V(state) = expected discounted sum of future monthly
    returns, where state = last month's return bin."""
    V = {b: 0.0 for b in range(4)}
    for _ in range(episodes):
        state = return_bin(0.0)
        for _ in range(N_MONTHS):
            r = simulate_return(portfolio)
            next_state = return_bin(r)
            V[state] += ALPHA * (r + GAMMA * V[next_state] - V[state])
            state = next_state
    return V
 
 
if __name__ == "__main__":
    print(f"{'Portfolio':<14}{'Predicted long-term value V(state=neutral)'}")
    predicted_values = {}
    for p in PORTFOLIOS:
        V = td0_value_prediction(p)
        predicted_values[p] = V[return_bin(0.0)]
        print(f"{p:<14}{V[return_bin(0.0)]:.3f}")
 
    best = max(predicted_values, key=predicted_values.get)
    print(f"\nPortfolio with highest predicted long-term value: {best}")
 
    # Compare against the simple historical average monthly return per portfolio
    print("\nHistorical average monthly return (sanity check):")
    for p in PORTFOLIOS:
        avg = sum(simulate_return(p) for _ in range(5000)) / 5000
        print(f"  {p:<14}: {avg:.3f}%")
 
