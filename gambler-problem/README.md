# Gambler Problem

This project implements **Value Iteration** to solve a **capital accumulation problem** in a stochastic environment. The goal is to determine the optimal policy for accumulating capital by making stakes at each state. The agent uses the value iteration algorithm to estimate the state-value function and the optimal policy.

---

## Project Overview

The project demonstrates the **Value Iteration Algorithm**, a method used to find the optimal policy for Markov Decision Processes (MDPs). In this scenario, the problem represents a process where an agent's capital can fluctuate based on the outcome of a coin toss. The agent must decide how much capital to stake based on the current state to maximize its expected future capital.

### Key Concepts:
- **Value Iteration**: A dynamic programming algorithm used to compute the optimal policy by iterating on the state-value function.
- **Policy**: The action taken by the agent at each state, which in this case is represented by the amount of capital to stake.

---

## Problem Description

In this problem, the agent starts with some capital (denoted by `state`), and at each step, it can stake a certain amount, which may either increase or decrease the capital based on a coin toss. The goal is to maximize the expected future capital by choosing the optimal stake at each state.

- The agent faces a **stochastic environment** where the next state depends on the outcome of a coin toss.
  - With probability `0.4`, the coin comes up heads, and the agent's capital increases by the stake.
  - With probability `0.6`, the coin comes up tails, and the agent's capital decreases by the stake.
- The agent's task is to determine the optimal amount to stake in each state to maximize future rewards.

### State Space:
- States range from `0` to `100` (denoted as `𝒮^+`), where each state corresponds to a specific amount of capital.

### Terminal State:
- The terminal state is the state with `100` capital, and the value at this state is set to `1.0`.

---

## Functions and Algorithm

### Key Variables:
- **goal**: The maximum capital the agent can accumulate (set to 100).
- **states**: The set of all possible states, which is a range from 0 to `goal`.
- **head_probability**: The probability of the coin toss resulting in heads (set to `0.4`).
- **state_value**: An array that holds the value of each state.
- **policy**: The optimal policy (amount to stake) for each state.

### Algorithm: Value Iteration
1. **Initialization**:
   - The state-value function (`state_value`) is initialized to zero for all states, except for the terminal state (`state_value[goal]`), which is set to 1.0.
   
2. **Value Iteration Loop**:
   - The value iteration loop starts by copying the current state-value and storing it in `old_state_value` for comparison later.
   - For each state (except the terminal state), the algorithm computes the expected return for all possible actions (stakes).
   - The action that maximizes the expected return is chosen as the optimal policy for that state.
   - The state-value is updated to reflect the maximum expected return for the current state.
   
3. **Convergence Check**:
   - The algorithm terminates when the maximum change in the state-value function between sweeps is less than a small threshold (`1e-9`), indicating convergence.

---

## Running the Code

### Prerequisites
To run the code, you need to have Python and the following libraries installed:

- **NumPy**: For handling arrays and matrix operations.
- **Matplotlib**: For plotting the results.

You can install the required dependencies using:

```bash
pip install numpy matplotlib
