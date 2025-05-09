# Windy Grid World with SARSA

This project implements the **SARSA (State-Action-Reward-State-Action)** algorithm to solve the classic **Windy Grid World** environment from reinforcement learning. The goal is to learn an optimal policy for reaching a goal state in a stochastic environment affected by "wind."

---

## Project Overview

The Windy Grid World is a grid-based reinforcement learning task where an agent must navigate from a start state to a goal state. The environment includes stochastic wind effects that push the agent upward in certain columns, making learning an effective strategy more challenging.

### Key Concepts:
- **SARSA Algorithm**: An on-policy temporal-difference (TD) learning method that updates the Q-values based on the action actually taken.
- **ε-greedy Policy**: A strategy that balances exploration and exploitation during learning.
- **Wind Effect**: Certain columns in the grid world push the agent upward, adding complexity to the environment.

---

## Environment Description

- The grid is **7 rows by 10 columns**.
- The **start state** is at coordinate `[3, 0]`.
- The **goal state** is at coordinate `[3, 7]`.
- The agent can take one of four actions:
  - `0` = up (↑)
  - `1` = down (↓)
  - `2` = left (←)
  - `3` = right (→)
- Wind strength varies per column:
  - Columns 3-5: wind = 1
  - Columns 6-7: wind = 2
  - Column 8: wind = 1

---

## SARSA Algorithm

The learning follows the SARSA update rule:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
\]

Where:
- \( Q(s, a) \): action-value estimate
- \( \alpha \): step-size (learning rate)
- \( \gamma \): discount factor
- \( r \): immediate reward
- \( a' \): next action chosen using ε-greedy

---

## Parameters

| Parameter                | Value    |
|--------------------------|----------|
| Grid Size                | 7 x 10   |
| Wind                     | Varies by column |
| Start State              | `[3, 0]` |
| Goal State               | `[3, 7]` |
| Reward per Step          | -1       |
| Discount Factor (𝛾)      | 1.0      |
| Step Size (α)            | 0.5      |
| Exploration Rate (ε)     | 0.1      |
| Max Episodes             | 170      |

---

## Running the Code

### Prerequisites

Install the required libraries:

```bash
pip install numpy matplotlib
