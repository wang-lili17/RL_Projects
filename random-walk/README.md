# Random Walk 

## Overview
This project implements both **Monte Carlo** (MC) and **Temporal Difference** (TD) methods for solving a random walk problem, with an emphasis on updating state-values through these reinforcement learning techniques. It allows for batch updates and visualizes the performance using Root Mean Squared Error (RMSE) metrics.

The problem setup is based on the **Gambler's Problem**, where a random walk is performed over a series of states. The agent starts in the middle state, and its objective is to reach the right terminal state with the highest reward (1.0). The goal of this project is to estimate the true state values using both MC and TD methods and visualize the convergence over episodes.

## Key Features:
- **Monte Carlo Method**: Updates state-values based on episodes, without requiring a model of the environment.
- **Temporal Difference Method**: Updates state-values based on the difference between successive states and rewards.
- **Batch Updating**: Allows for batch updates to improve the stability of learning.
- **State-Value Estimation**: Both MC and TD methods are applied to estimate the value of each state in the random walk process.
- **Visualization**: RMSE values are plotted for both methods with varying step sizes to compare their performance.


### Core Components:
1. **Monte Carlo Method** (`monte_carlo`):
   - Estimates state-values based on the average of the rewards obtained from episodes.
   - Updates state values incrementally using a simple reward structure (either 0 or 1).

2. **Temporal Difference Learning** (`temporal_difference`):
   - Updates state-values using the TD update rule, considering the estimated value of subsequent states.
   - Utilizes the Bellman equation for updating the state values iteratively.

3. **Batch Updating** (`batch_updating`):
   - Uses both TD and MC methods in a batch learning setup to update values over multiple runs and episodes.
   - Computes RMSE over multiple trials to measure convergence and error.

4. **Visualization** (`main.py`):
   - Plots the evolution of state-value estimates for both MC and TD methods over different episodes.
   - Compares the RMSE values across various step-size parameters, showing how the learning rate influences the convergence.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/random_walk.git

You can install the dependencies using:

```bash
pip install numpy matplotlib tqdm
