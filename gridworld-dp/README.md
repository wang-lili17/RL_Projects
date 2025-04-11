# Policy Evaluation in Gridworld

This project demonstrates **Iterative Policy Evaluation** in a 4x4 Gridworld environment. The agent in the grid can take one of four actions (left, up, right, or down) with equal probability. The objective is to compute the state-value function `vπ(s)` for the agent following a uniform random policy using both **in-place** and **out-of-place** computation methods.

---

## Project Overview

The Gridworld environment is used to demonstrate **Policy Evaluation** in Reinforcement Learning (RL). The goal is to compute the state-value function `vπ(s)` iteratively for each state under a given policy π. In this case, the policy is random, meaning each action is chosen with equal probability. 

The **Bellman Equation** is used to update the value of each state until convergence. The project computes the state values using both **in-place** (updating the same array) and **out-of-place** (updating a copy of the array) methods.

### Key Concepts:
- **State-Value Function** `vπ(s)`: This represents the expected return (reward) from a state `s` when following a specific policy `π`.
- **Bellman Equation**: The equation used to compute the value of each state iteratively based on the rewards from the next state.
- **In-place vs Out-of-place Updates**: 
  - **In-place**: Updates the state values directly in the existing array.
  - **Out-of-place**: Uses a copy of the state values for computation, preserving the original array during updates.

---

## Gridworld Environment

The environment is a 4x4 grid where each state represents a specific location. The agent can take one of the following actions at each state:

- **Left**: Move left
- **Up**: Move up
- **Right**: Move right
- **Down**: Move down

Each action has an equal probability of `1/4`. The agent will receive a reward of `-1` for every action until it reaches one of the **terminal states**. The **terminal states** are:
- **Top-left corner (0, 0)**
- **Bottom-right corner (3, 3)**

When the agent reaches a terminal state, the reward is `0`.

### Grid Structure:
- The grid is a **4x4 matrix** where each position is a state.
- **State transitions**: The agent moves to a new state based on its current state and action taken. If the action would move the agent out of bounds, it remains in the same state.

---

## Running the Code

### Prerequisites
To run this code, ensure you have the following Python packages installed:

- **NumPy**: Used for matrix operations and array handling.
- **Matplotlib**: Used for plotting the state-value grid.
  
You can install the required dependencies using:

```bash
pip install numpy matplotlib
