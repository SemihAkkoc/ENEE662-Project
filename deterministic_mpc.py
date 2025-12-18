import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
import time as tm  # Renamed to avoid conflict with 'time' array variable

# ==========================================
# 1. System Definitions & Constraints
# ==========================================
dt = 0.05

# Dynamics
A = np.array([
    [1, dt, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, dt],
    [0, 0, 0, 1]
])

B = np.array([
    [0.5 * dt**2, 0],
    [dt, 0],
    [0, 0.5 * dt**2],
    [0, dt]
])

n_x = 4
n_u = 2

# Constraints
x_max = np.array([2.0, 1.0, 2.0, 1.0])  # [px, vx, py, vy]
u_max = np.array([0.7, 0.7])            # [ux, uy]

# Reference
p_ref = np.array([1.5, 1.5])
x_ref = np.array([p_ref[0], 0, p_ref[1], 0])

# ==========================================
# 2. MPC Tuning & Terminal Cost
# ==========================================
N = 10 

# Weight matrices
Q = np.diag([5, 1, 5, 1])
R = 0.1 * np.eye(n_u)

# Compute Terminal Cost P via LQR
Q_lqr = np.eye(n_x)
R_lqr = np.eye(n_u)
P_lqr = solve_discrete_are(A, B, Q_lqr, R_lqr)

# ==========================================
# 3. Setup CVXPY Optimization Problem
# ==========================================
x_var = cp.Variable((n_x, N + 1))
u_var = cp.Variable((n_u, N))
x_init = cp.Parameter(n_x)

cost = 0
constraints = [x_var[:, 0] == x_init]

for k in range(N):
    state_error = x_var[:, k] - x_ref
    cost += cp.quad_form(state_error, Q) + cp.quad_form(u_var[:, k], R)
    constraints += [x_var[:, k+1] == A @ x_var[:, k] + B @ u_var[:, k]]
    constraints += [cp.abs(x_var[:, k+1]) <= x_max]
    constraints += [cp.abs(u_var[:, k]) <= u_max]

term_error = x_var[:, N] - x_ref
cost += cp.quad_form(term_error, P_lqr)

problem = cp.Problem(cp.Minimize(cost), constraints)

# ==========================================
# 4. Simulation Loop
# ==========================================
T_sim = 8.0          
steps = int(T_sim / dt)
x_current = np.zeros(n_x)

# Data logging
x_history = [x_current]
u_history = []
obj_history = []    # To store optimal cost
time_history = []   # To store solve time

print("Starting Simulation...")
for t in range(steps):
    x_init.value = x_current
    
    # Timing the solver
    start_timer = tm.perf_counter()
    problem.solve(solver=cp.OSQP, warm_start=True)
    end_timer = tm.perf_counter()
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Error at step {t}: Solver status {problem.status}")
        break

    # Get control input
    u_apply = u_var[:, 0].value
    
    # Apply to system
    x_next = A @ x_current + B @ u_apply
    
    # Update and Log
    x_current = x_next
    x_history.append(x_current)
    u_history.append(u_apply)
    obj_history.append(problem.value)
    time_history.append((end_timer - start_timer) * 1000) # Convert to ms

x_history = np.array(x_history)
u_history = np.array(u_history)
time_axis = np.arange(len(x_history)) * dt
obj_history = np.array(obj_history)
time_history = np.array(time_history)

# ==========================================
# 5. Plotting
# ==========================================
if len(u_history) > 0:
    # --- FIGURE 1: 2D Spatial Trajectory ---
    plt.figure(figsize=(6, 6))
    plt.plot(x_history[:, 0], x_history[:, 2], 'b-', linewidth=2, label='Trajectory')
    plt.plot(x_ref[0], x_ref[2], 'rx', markersize=10, markeredgewidth=3, label='Target')
    plt.plot(0, 0, 'go', label='Start')
    rect = plt.Rectangle((-x_max[0], -x_max[2]), 2*x_max[0], 2*x_max[2], 
                         fill=False, edgecolor='r', linestyle='--', label='Constraint')
    plt.gca().add_patch(rect)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xlabel('Position X [m]')
    plt.ylabel('Position Y [m]')
    plt.title('2D Position Trajectory')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('det_mpc.png') # Uncomment to save for LaTeX

    # --- FIGURE 2: States and Inputs Evolution ---
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # 2a. Position Evolution
    axs2[0].plot(time_axis, x_history[:, 0], 'b-', linewidth=2, label='$p_x$')
    axs2[0].plot(time_axis, x_history[:, 2], 'r--', linewidth=2, label='$p_y$')
    axs2[0].hlines([x_ref[0]], time_axis[0], time_axis[-1], colors='k', linestyles=':', label='$p_{ref}$')
    axs2[0].set_ylabel('Position [m]')
    axs2[0].set_title('Position Evolution')
    axs2[0].legend(loc='lower right')
    axs2[0].grid(True)

    # 2b. Velocity Evolution
    axs2[1].plot(time_axis, x_history[:, 1], 'b-', linewidth=2, label='$v_x$')
    axs2[1].plot(time_axis, x_history[:, 3], 'r--', linewidth=2, label='$v_y$')
    axs2[1].hlines([x_max[1], -x_max[1]], time_axis[0], time_axis[-1], colors='k', linestyles=':', label='Limits')
    axs2[1].set_ylabel('Velocity [m/s]')
    axs2[1].set_title('Velocity Evolution')
    axs2[1].legend(loc='upper right')
    axs2[1].grid(True)

    # 2c. Input Evolution
    axs2[2].step(time_axis[:-1], u_history[:, 0], where='post', color='b', linestyle='-', linewidth=2, label='$u_x$')
    axs2[2].step(time_axis[:-1], u_history[:, 1], where='post', color='r', linestyle='--', linewidth=2, label='$u_y$')
    axs2[2].hlines([u_max[0], -u_max[0]], time_axis[0], time_axis[-1], colors='k', linestyles=':', label='Limits')
    axs2[2].set_ylabel('Control Input [m/s²]')
    axs2[2].set_xlabel('Time [s]')
    axs2[2].set_title('Control Inputs')
    axs2[2].legend(loc='upper right')
    axs2[2].grid(True)
    
    plt.tight_layout()
    # plt.savefig('det_states_inputs.png') # Uncomment to save for LaTeX

    # --- FIGURE 3: Solver Performance ---
    fig3, axs3 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 3a. Objective Function Cost
    axs3[0].plot(np.arange(len(obj_history)), obj_history, markersize=3, linewidth=1.5)
    axs3[0].set_ylabel('Cost Value $J^*$')
    axs3[0].set_title('Objective Function Cost vs MPC Step')
    axs3[0].grid(True)
    axs3[0].set_yscale('log') # Log scale is often better for cost visualization

    # 3b. Solver Time
    axs3[1].plot(np.arange(len(time_history)), time_history, markersize=3, linewidth=1.5)
    axs3[1].set_ylabel('Time [ms]')
    axs3[1].set_xlabel('MPC Step $k$')
    axs3[1].set_title('Solver Computation Time vs MPC Step')
    axs3[1].grid(True)

    plt.tight_layout()
    # plt.savefig('det_solver_stats.png') # Uncomment to save for LaTeX
    
    plt.show()

else:
    print("Simulation failed before any steps could be taken.")