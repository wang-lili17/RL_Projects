# Cliff Walking

This project implements and compares temporal-difference reinforcement learning algorithms on the **Cliff Walking** environment. It focuses on:

- SARSA  
- Expected SARSA  
- Q-Learning  

These methods are evaluated based on their reward performance and sensitivity to step-size (α), using asymptotic and interim performance comparisons.

## Project Overview

Cliff Walking is a classic reinforcement learning scenario that highlights the trade-off between exploration and exploitation. The agent must navigate from a start to a goal while avoiding a **cliff region** that results in high negative rewards.

### Environment Details

- **Grid Size**: 4 (height) × 12 (width)  
- **Start State**: (3, 0)  
- **Goal State**: (3, 11)  
- **Actions**: Up (0), Down (1), Left (2), Right (3)  
- **Cliff Region**: Bottom row between start and goal  
- **Rewards**:  
  - `-1` per step  
  - `-100` for falling off the cliff (resets to start)

## Implemented Methods

### SARSA

On-policy TD control algorithm using the action actually taken in the next state.

### Expected SARSA

On-policy algorithm using the expected value over all possible actions in the next state, weighted by the policy.

### Q-Learning

Off-policy TD control algorithm using the maximum reward obtainable from the next state.

## ε-Greedy Policy

All methods use ε-greedy policies for exploration:
- Probability ε of choosing a random action  
- Probability 1−ε of choosing the best-known action  

## Experiments & Visualization

### Learning Curve

Plots the average sum of rewards per episode for:

- SARSA  
- Q-learning  

### Step-size Sensitivity

Compares different methods across step-sizes in the range `[0.1, 1.0]`:
- **Asymptotic Performance**: Averaged over all episodes  
- **Interim Performance**: Averaged over the first 100 episodes  

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`
- `tqdm`

Install dependencies:

```bash
pip install numpy matplotlib tqdm
