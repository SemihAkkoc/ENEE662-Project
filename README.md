# ENEE662-Project: Model Predictive Control Implementations

Convex optimization course project for ENEE662 at UMD.

## Project Overview

This project implements and compares three Model Predictive Control (MPC) strategies for a 2D double integrator system:
- **Deterministic MPC (DMPC)** - Standard MPC without disturbance handling
- **Stochastic MPC (SMPC)** - Chance-constrained MPC with probabilistic constraint satisfaction
- **Robust MPC (RMPC)** - Tube-based MPC with hard constraint guarantees under bounded disturbances

## System Description

All implementations control a 2D double integrator system representing a point mass in the xy-plane:

**State**: `x = [px, vx, py, vy]ᵀ` (positions and velocities in x and y)  
**Input**: `u = [ux, uy]ᵀ` (accelerations in x and y)  
**Sampling time**: `dt = 0.05s`

### Dynamics
```
A = [1  dt  0   0 ]
    [0  1   0   0 ]
    [0  0   1   dt]
    [0  0   0   1 ]

B = [0.5*dt²  0     ]
    [dt       0     ]
    [0        0.5*dt²]
    [0        dt    ]
```

### Constraints
- **State**: `|px|, |py| ≤ 2.0 m`, `|vx|, |vy| ≤ 1.0 m/s`
- **Input**: `|ux|, |uy| ≤ 0.65-0.7 m/s²` (varies by implementation)

### Objective
Navigate from origin to reference position `p_ref = [1.5, 1.5]` with zero velocity while respecting constraints.

---

## 1. Deterministic MPC (DMPC.py)

### Description
Standard MPC implementation assuming perfect model and no disturbances. Solves a finite-horizon optimal control problem at each time step.

### Key Features
- **Language**: Python
- **Solver**: CVXPY with OSQP
- **Horizon**: N = 10 steps
- **Cost**: Quadratic stage cost `Q = diag([5,1,5,1])`, `R = 0.1*I`, terminal cost from LQR
- **Constraints**: Hard state and input bounds applied at each prediction step

### Requirements
```bash
pip install numpy cvxpy matplotlib scipy
```

### Usage
```bash
python DMPC.py
```

### Output
- **Figure 1**: 2D trajectory plot showing spatial path
- **Figure 2**: Time evolution of positions, velocities, and control inputs
- **Figure 3**: Solver performance metrics (objective value, computation time)

---

## 2. Stochastic MPC (SMPC.m)

### Description
Chance-constrained MPC that handles stochastic additive disturbances using constraint tightening. Ensures probabilistic constraint satisfaction (e.g., 99% probability).

### Key Features
- **Language**: MATLAB
- **Solver**: YALMIP with quadprog
- **Horizon**: Np = 50 steps
- **Disturbance Model**: Bounded uniform noise on positions `w ∈ [-0.02, 0.02]`
- **Chance Constraints**: Tightens constraints based on closed-loop covariance prediction
- **Control Law**: `u = v + K*e` (nominal + feedback correction)
- **Risk Levels**: `εx = εu = 0.01` (1% violation probability)

### Method
1. Designs feedback gain `K` via LQR for error stabilization
2. Propagates closed-loop covariance using Lyapunov recursion
3. Computes time-varying constraint tightenings using `β = norminv(1 - ε/2)`
4. Solves nominal MPC with tightened constraints
5. Applies certainty-equivalent control with feedback correction

### Requirements
- MATLAB with Optimization Toolbox
- YALMIP toolbox ([https://yalmip.github.io/](https://yalmip.github.io/))

### Usage
```matlab
run SMPC.m
```

### Output
- **Figure 1**: XY trajectory with 3σ tubes visualizing uncertainty propagation
- **Figure 2**: Position/input plots with ±3σ confidence bounds and constraint tightening
- **Figure 3**: Velocity constraint verification
- **Figure 4**: QP objective and solve time analysis
- **Animation**: Video showing real-time tube propagation (`SMPC_animation.mp4`)

---

## 3. Robust MPC (RMPC.m)

### Description
Tube-based MPC with hard constraint guarantees under worst-case bounded disturbances. Uses Monte Carlo simulation to compute tube size and constraint tightening.

### Key Features
- **Language**: MATLAB
- **Solver**: YALMIP with quadprog
- **Horizon**: Np = 10 steps
- **Disturbance Model**: Bounded additive noise `w ∈ [-0.02, 0.02]` on positions
- **Tube Design**: Monte Carlo simulation (1000 trajectories × 400 steps) to estimate error bounds
- **Control Law**: `u = v + KT*e` (nominal + robust feedback)
- **Guarantees**: 100% constraint satisfaction for all realizations within disturbance bounds

### Method
1. Designs tube gain `KT` via LQR to stabilize error dynamics
2. Estimates tube size `e_max` using Monte Carlo under closed-loop feedback
3. Shrinks tube to ensure reference lies inside tightened feasible region
4. Tightens constraints: `x_tight = x_max - e_max`, `u_tight = u_max - |KT|*e_max`
5. Solves nominal MPC for tube center with tightened constraints
6. Actual state guaranteed to stay within tube around nominal trajectory

### Requirements
- MATLAB with Optimization Toolbox
- YALMIP toolbox

### Usage
```matlab
run RMPC.m
```

### Output
- **Figure 1**: Position/input time histories with tightened constraint visualization
- **Figure 2**: XY trajectory with shaded constraint regions (original and tightened)
- **Figure 3+**: Tube error verification, input decomposition, detailed diagnostics
- **Animation**: Video showing tube-based motion (`mpc_animation.mp4`)
- **Saved Results**: All plots saved to `mpc_results/` folder

---

## Comparison

| Feature | DMPC | SMPC | RMPC |
|---------|------|------|------|
| **Disturbance Handling** | None | Stochastic (probabilistic) | Worst-case bounded |
| **Constraint Guarantee** | Deterministic only | Probabilistic (99%) | Hard (100%) |
| **Conservatism** | Low | Medium | Higher |
| **Tube/Uncertainty** | None | Statistical (covariance) | Deterministic (MC bounds) |
| **Feedback** | Open-loop | Ancillary (K) | Robust (KT) |
| **Horizon** | 10 | 50 | 10 |
| **Computation** | Fastest | Medium | Medium |

## Key Insights

- **DMPC**: Simple and efficient but vulnerable to disturbances; may violate constraints in practice
- **SMPC**: Balances performance and safety with probabilistic guarantees; requires accurate noise statistics
- **RMPC**: Most conservative but provides hard guarantees; suitable for safety-critical applications

## References

This project implements concepts from:
- Model Predictive Control (Rawlings, Mayne, Diehl)
- Stochastic MPC with constraint tightening (Kouvaritakis, Cannon)
- Tube-based robust MPC (Mayne et al., Langson et al.)

## Authors

ENEE662 Students, University of Maryland, College Park

## License

Academic use for ENEE662 course project.
