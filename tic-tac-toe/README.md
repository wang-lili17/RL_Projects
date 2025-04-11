# Tic Tac Toe

This is a simple implementation of Tic Tac Toe using **Reinforcement Learning (RL)** in Python. The RL agents learn to play the game by training through self-play, and can eventually compete with each other or a human player.

## How It Works

- **Training (`train`)**: Two RL agents play against each other to learn optimal strategies.
- **Competition (`compete`)**: Evaluates the learned strategies of both agents by making them compete using a greedy policy.
- **Play (`play`)**: A human can play against the trained RL agent (which always plays second).

## Learning Mechanism

- Each `RLPlayer` uses **ε-greedy strategy**:
  - During training: `ε=0.01` (1% exploration)
  - During evaluation/play: `ε=0` (fully greedy)
- After each game, the players update their state-value estimates to reinforce good moves and penalize bad ones.


