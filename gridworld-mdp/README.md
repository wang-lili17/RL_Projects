readme_content = """
# Grid World Value Function Visualization

This project implements the evaluation and visualization of **state-value functions** and **optimal policies** in a grid world environment using the Bellman equations from reinforcement learning theory.

## Figures and Explanations

### Figure 3.2 – Value Function under a Random Policy
- Shows the value function for a fixed (random) policy with a discount factor $\\gamma = 0.9$.
- Computed using the Bellman equation (system of linear equations).
- Notable observations:
  - Negative values near the lower edge indicate the high probability of hitting the grid boundary.
  - State **A** has high reward (10), but its value is lower due to transitions to **A′**.
  - State **B** is valued more than its immediate reward (5) because transitions to **B′** often result in reaching **A** or **B** again.

---

### Figure 3.5 – Optimal Value Function and Policies
- Solves the Bellman **optimality** equation for $v_*$.
- Two outputs are generated:
  - Optimal **value function** (center image).
  - Optimal **policy** showing arrows for best actions (right image).
- Multiple arrows in a cell indicate multiple optimal actions.

---

## Code Structure

### Main Scripts
- Computes value functions for:
  - A random policy (`v_π`)
  - The optimal policy (`v_*`)
- Convergence achieved by iterative updates using Bellman equations.

### `src/grid_world.py`
- Core module containing:
  - Environment step function `step(state, action)`
  - Visualizer `draw(grid, is_policy=False)`
  - Definitions for special states A, A′, B, B′
  - Grid layout and actions

## Dependencies
- `numpy`
- `matplotlib`

