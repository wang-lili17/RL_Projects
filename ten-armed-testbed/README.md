# Ten Armed Testbed
This project demonstrates different multi-armed bandit strategies through simulation and visualization, based on examples from **Reinforcement Learning** (e.g., Sutton & Barto). Each figure represents the behavior of one or more algorithms across steps.

---

## 1. Reward Distribution

Visualizes the reward distribution for different actions using a violin plot.

---

## 2. Greedy vs ε-Greedy Action Selection

Compares the performance of greedy and ε-greedy methods (with ε = 0, 0.1, 0.01).  
Both average reward and % optimal action over time are shown.

---

## 3. Optimistic Initial Values vs Realistic Values

Compares two bandits:
- One with optimistic initial values (Q₁(a) = 5, ε = 0)
- One with realistic values (Q₁(a) = 0, ε = 0.1)  
Helps illustrate how initial estimates affect exploration.

---

## 4. Upper Confidence Bound (UCB) vs ε-Greedy

Compares:
- UCB method (c = 2)
- ε-Greedy (ε = 0.1)  
Evaluates how confidence bounds can improve exploration.

---

## 5. Gradient Bandit Algorithms (GBA)

Tests 4 GBA configurations with different:
- Learning rates (α = 0.1 and 0.4)
- Baseline usage (with and without)  
Analyzes their performance in terms of % optimal action.

---

##  Requirements

Make sure to have the following Python packages installed:

Numpy

Matplotlib