# Importance Sampling in Off-Policy Evaluation

This project demonstrates **Off-Policy Evaluation** using **Ordinary Importance Sampling** in a simple stochastic environment. The goal is to estimate the expected return of a **target policy** while following a different **behavior policy** that generates trajectories.

---

## Project Overview

The project highlights a fundamental concept in **Reinforcement Learning**: estimating the performance of a target policy when data is collected under a different behavior policy. This simulation showcases how importance sampling adjusts the evaluation to account for this policy mismatch.

### Key Concepts:
- **Off-Policy Evaluation**: Estimating the expected reward of a target policy using samples from a different policy.
- **Importance Sampling**: A statistical method to reweight the observed returns based on how likely the actions were under the target policy versus the behavior policy.
- **Variance in Estimates**: Demonstrates how high variance can arise from long action trajectories or rare actions.

---

## Problem Description

The environment is structured such that an agent selects actions according to a **behavior policy**, but we want to evaluate a **target policy**. The stochastic transitions and outcomes are influenced by simple binary choices.

- The **target policy** always chooses the action `left`.
- The **behavior policy** selects `left` or `right` with equal probability (uniform random).
- The trajectory continues until the agent either transitions to a terminal state (with some probability) or selects a terminating action.

### Actions:
- `left`: Has a 90% chance to continue and a 10% chance to terminate with a reward of `1`.
- `right`: Always terminates with a reward of `0`.

---

## Functions and Algorithm

### Key Functions:
- **`target_policy()`**: Always returns the action `left`.
- **`behavior_policy()`**: Randomly returns `left` or `right` with equal probability.
- **`play()`**: Simulates a trajectory using the behavior policy and records the actions until termination, returning the final reward and action history.

### Algorithm: Ordinary Importance Sampling
1. **Run Simulations**:
   - Simulate a large number of episodes (e.g., 100,000) using the behavior policy.
   - For each episode, record the action trajectory and resulting reward.

2. **Compute Importance Sampling Ratio**:
   - If the last action was `right`, the target policy would never have selected that — set ratio to 0.
   - Otherwise, compute the ratio as \( \frac{1}{(0.5)^n} \) where `n` is the length of the trajectory (since each `left` has a 0.5 chance under the behavior policy).

3. **Calculate Weighted Reward**:
   - Multiply the reward by the importance sampling ratio.
   - Accumulate and average the rewards over episodes to estimate the value of the target policy.

---

## Running the Code

### Prerequisites

To run the code, ensure you have Python installed along with the following libraries:

- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting the estimates.

You can install the dependencies using:

```bash
pip install numpy matplotlib
