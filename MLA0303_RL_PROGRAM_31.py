"""
Experiment 31: Implement the SARSA algorithm to develop an AI agent
that learns to play Tic-Tac-Toe against a random opponent.
"""
 
import random
from collections import defaultdict
 
random.seed(18)
 
WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
             (2, 5, 8), (0, 4, 8), (2, 4, 6)]
 
 
def winner(board):
    for a, b, c in WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    if " " not in board:
        return "DRAW"
    return None
 
 
def available_actions(board):
    return [i for i, v in enumerate(board) if v == " "]
 
 
def random_opponent_move(board):
    return random.choice(available_actions(board))
 
 
ALPHA, GAMMA, EPISODES = 0.2, 0.95, 20000
Q = defaultdict(lambda: defaultdict(float))
 
 
def epsilon_greedy(board, epsilon):
    actions = available_actions(board)
    if random.random() < epsilon:
        return random.choice(actions)
    key = tuple(board)
    return max(actions, key=lambda a: Q[key][a])
 
 
def train():
    epsilon = 1.0
    results = []
    for ep in range(EPISODES):
        board = [" "] * 9
        state = tuple(board)
        action = epsilon_greedy(board, epsilon)
        while True:
            board[action] = "X"                     # SARSA agent plays X
            w = winner(board)
            if w is not None:
                reward = 1 if w == "X" else (0 if w == "DRAW" else -1)
                Q[state][action] += ALPHA * (reward - Q[state][action])
                results.append(reward)
                break
 
            opp_move = random_opponent_move(board)     # random opponent plays O
            board[opp_move] = "O"
            w = winner(board)
            if w is not None:
                reward = 1 if w == "X" else (0 if w == "DRAW" else -1)
                Q[state][action] += ALPHA * (reward - Q[state][action])
                results.append(reward)
                break
 
            next_state = tuple(board)
            next_action = epsilon_greedy(board, epsilon)
            reward = 0
            Q[state][action] += ALPHA * (reward + GAMMA * Q[next_state][next_action] - Q[state][action])
            state, action = next_state, next_action
        epsilon = max(0.05, epsilon * 0.9997)
    return results
 
 
def play_greedy_game():
    board = [" "] * 9
    while True:
        state = tuple(board)
        actions = available_actions(board)
        action = max(actions, key=lambda a: Q[state][a])
        board[action] = "X"
        w = winner(board)
        if w is not None:
            return board, w
        opp_move = random_opponent_move(board)
        board[opp_move] = "O"
        w = winner(board)
        if w is not None:
            return board, w
 
 
if __name__ == "__main__":
    results = train()
    print(f"Win/draw/loss rate (first 500 games): "
          f"W={results[:500].count(1)}, D={results[:500].count(0)}, L={results[:500].count(-1)}")
    print(f"Win/draw/loss rate (last 500 games) : "
          f"W={results[-500:].count(1)}, D={results[-500:].count(0)}, L={results[-500:].count(-1)}")
 
    board, outcome = play_greedy_game()
    print("\nSample evaluation game (agent = X):")
    for i in range(0, 9, 3):
        print(board[i:i + 3])
    print("Outcome:", "Agent won!" if outcome == "X" else ("Draw." if outcome == "DRAW" else "Agent lost."))
 
