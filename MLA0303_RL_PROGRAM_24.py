"""
Experiment 24: Develop an automated trading system using the
REINFORCE algorithm to learn trading strategies that maximize profit
and minimize risk in financial markets. State = discretized recent
price trend; Actions = BUY, SELL, HOLD.
"""
 
import numpy as np
 
np.random.seed(14)
 
N_DAYS = 40
N_TREND_BINS = 5          # 0=strong down .. 4=strong up
ACTIONS = ["BUY", "SELL", "HOLD"]
GAMMA = 0.95
EPISODES = 2000
 
 
def generate_price_series(n_days=N_DAYS):
    prices = [100.0]
    for _ in range(n_days - 1):
        change = np.random.normal(0, 1.5)
        prices.append(max(1.0, prices[-1] + change))
    return prices
 
 
def trend_bin(prices, t):
    if t == 0:
        return 2
    change = prices[t] - prices[t - 1]
    if change < -2:
        return 0
    if change < 0:
        return 1
    if change == 0:
        return 2
    if change < 2:
        return 3
    return 4
 
 
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
 
 
theta = np.zeros((N_TREND_BINS, 2, len(ACTIONS)))    # state = (trend, holding_stock)
 
 
def run_episode(prices):
    cash, shares = 1000.0, 0
    traj = []
    for t in range(len(prices)):
        trend = trend_bin(prices, t)
        holding = 1 if shares > 0 else 0
        state = (trend, holding)
        probs = softmax(theta[state])
        action = np.random.choice(len(ACTIONS), p=probs)
 
        reward = -0.2 if ACTIONS[action] == "HOLD" else 0   # small opportunity cost for holding
        if ACTIONS[action] == "BUY" and shares == 0:
            shares = cash / prices[t]
            cash = 0
            reward += 0.5 if trend >= 3 else -0.5            # small shaping signal
        elif ACTIONS[action] == "SELL" and shares > 0:
            cash = shares * prices[t]
            reward = cash - 1000.0        # profit/loss vs starting capital
            shares = 0
        traj.append((state, action, reward, probs))
    # Liquidate any remaining position at the final price
    portfolio_value = cash + shares * prices[-1]
    if shares > 0:
        traj[-1] = (traj[-1][0], traj[-1][1], portfolio_value - 1000.0, traj[-1][3])
    return traj, portfolio_value
 
 
def train():
    baseline = 0.0
    portfolio_history = []
    for ep in range(EPISODES):
        prices = generate_price_series()
        traj, portfolio_value = run_episode(prices)
        G, returns = 0, [0] * len(traj)
        for t in reversed(range(len(traj))):
            G = traj[t][2] + GAMMA * G
            returns[t] = G
        episode_return = portfolio_value - 1000.0
        baseline = 0.95 * baseline + 0.05 * episode_return
        for (state, action, r, probs), G_t in zip(traj, returns):
            advantage = G_t - baseline
            grad = -probs
            grad[action] += 1
            theta[state] += 0.02 * advantage * grad
        portfolio_history.append(portfolio_value)
    return portfolio_history
 
 
if __name__ == "__main__":
    history = train()
    print(f"Average final portfolio value (first 100 episodes): {np.mean(history[:100]):.2f}")
    print(f"Average final portfolio value (last 100 episodes) : {np.mean(history[-100:]):.2f}")
 
    np.random.seed(999)
    test_prices = generate_price_series()
    traj, final_value = run_episode(test_prices)
    actions_taken = [ACTIONS[a] for _, a, _, _ in traj]
    print(f"\nEvaluation run starting capital: 1000.00")
    print(f"Evaluation run final portfolio value: {final_value:.2f}")
    print(f"Sample of actions taken: {actions_taken[:15]}")
 
