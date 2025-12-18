clc; clear; close all;
yalmip('clear');

%% ===================== 1) Model: 2D double integrator =====================
% State: x = [px; vx; py; vy]'
% Input: u = [ux; uy]'

dt = .05;   % sampling time

A = [1, dt, 0, 0;
     0,  1, 0, 0;
     0,  0, 1, dt;
     0,  0, 0, 1];

B = [1/2*dt^2,  0;
     dt      ,  0;
     0       , 1/2*dt^2;
     0       ,  dt];

n = size(A,1);         % 4
m = size(B,2);         % 2

%% ===================== 2) Constraints and cost ============================
% True state constraints
x_max = [ 2.0;   % |px| <= 2
          1.0;   % |vx| <= 1
          2.0;   % |py| <= 2
          1.0 ]; % |vy| <= 1

% True input constraints
u_max = [0.7;    % |ux| <= 0.7
         0.7];   % |uy| <= 0.7

% MPC weights (more weight on positions than velocities)
Q = diag([5 1 5 1]);      % state cost
R = 0.1*eye(m);           % input cost

% Terminal cost from LQR (for nice convergence)
[~, P, ~] = dlqr(A, B, eye(n), eye(m));    % only P is used here

% Reference state: move to p_ref with zero velocity
p_ref = [1.5; 1.5];
x_ref = [p_ref(1); 0; p_ref(2); 0];

%% ===================== 2.5) Robust tube via MC + shrink ==================
% Assume additive bounded disturbance on positions:
%   w_k = [wx; 0; wy; 0], with |wx|,|wy| <= w_pos_max
w_pos_max = 0.02;   % disturbance bound

% Disturbance sampler
sample_w = @() [w_pos_max*(2*rand-1);   % wx in [-w_pos_max, w_pos_max]
                0;
                w_pos_max*(2*rand-1);
                0];

% ---------- 1) Choose a tube gain KT via LQR (single scale) ---------------
R_lqr_scale = 5;           % you can tune this (5 is a reasonable compromise)
Q_lqr       = eye(n);
R_lqr       = R_lqr_scale*eye(m);

[KT, ~, ~] = dlqr(A, B, Q_lqr, R_lqr);   % u = -KT * e
Acl        = A - B*KT;                   % error dynamics

% ---------- 2) Monte Carlo estimate of error bound e_max_mc ---------------
N_mc = 1000;   % number of trajectories
T_mc = 400;    % length of each trajectory [steps]

e_max_mc = zeros(n,1);
for s = 1:N_mc
    e = zeros(n,1);
    for k = 1:T_mc
        w = sample_w();
        e = Acl*e + w;
        e_max_mc = max(e_max_mc, abs(e));
    end
end

fprintf('\nMC-based error bound (raw): e_max_mc = [%g %g %g %g]^T\n', e_max_mc);

% ---------- 3) Shrink e_max so reference lies inside tightened region -----
% We want: x_tight_max(1) = x_max(1) - e_max(1) >= |p_ref(1)|
%          x_tight_max(3) = x_max(3) - e_max(3) >= |p_ref(2)|
%
% i.e.,   e_max(1) <= x_max(1) - |p_ref(1)|
%         e_max(3) <= x_max(3) - |p_ref(2)|

pos_ref = abs([x_ref(1); x_ref(3)]);   % desired |px_ref|, |py_ref|
pos_slack = [x_max(1); x_max(3)] - pos_ref;

if any(pos_slack <= 0)
    error('Reference is outside the original state bounds: cannot enforce tightening to contain it.');
end

% Scaling factor alpha so that alpha * e_max_mc satisfies the position slack
alpha1 = pos_slack(1) / e_max_mc(1);
alpha3 = pos_slack(2) / e_max_mc(3);
alpha  = min([1.0, 0.9*alpha1, 0.9*alpha3]);   % 0.9 = small margin

if alpha <= 0
    error('Cannot fit reference into tightened region even by shrinking e_max_mc.');
end

e_max = alpha * e_max_mc;

fprintf('Scaling factor alpha = %g (shrink MC bound to fit reference)\n', alpha);
fprintf('Final e_max = [%g %g %g %g]^T\n', e_max);

% ---------- 4) Tightened constraints for nominal MPC ----------------------
x_tight_max = x_max - e_max;            % state tightening
u_tight_max = u_max - abs(KT)*e_max;    % input tightening (2x1)

fprintf('x_tight_max = [%g %g %g %g]^T\n', x_tight_max);
fprintf('u_tight_max = [%g %g]^T\n',      u_tight_max);

% Safety checks
if any(x_tight_max <= 0)
    warning('State tightening too severe: adjust w_pos_max or x_max.');
end
if any(u_tight_max <= 0)
    warning('Input tightening too severe: adjust w_pos_max, u_max, or R_lqr_scale.');
end


%% ===================== 3) MPC formulation (YALMIP) =======================
Np = 10;   % prediction horizon

% Parameter: current state
x0 = sdpvar(n,1);

% Decision variables: nominal state and input trajectories
x = sdpvar(n, Np+1, 'full');   % here x plays the role of nominal state z
u = sdpvar(m, Np,   'full');   % nominal input v

constraints = [];
objective   = 0;

% Initial condition
constraints = [constraints, x(:,1) == x0];

for k = 1:Np
    % Nominal dynamics
    constraints = [constraints, x(:,k+1) == A*x(:,k) + B*u(:,k)];
    
    % Tightened state constraints (on nominal state)
    constraints = [constraints, -x_tight_max <= x(:,k) <= x_tight_max];
    
    % Tightened input constraints (on nominal input)
    constraints = [constraints, -u_tight_max <= u(:,k) <= u_tight_max];
    
    % Stage cost
    objective = objective + (x(:,k) - x_ref)'*Q*(x(:,k) - x_ref) ...
                            + u(:,k)'*R*u(:,k);
end

% method 1 - terminal constriant set
% Terminal state constraints (also tightened)
constraints = [constraints, -x_tight_max <= x(:,Np+1) <= x_tight_max];

% Terminal cost
objective = objective + (x(:,Np+1) - x_ref)'*P*(x(:,Np+1) - x_ref);

% method 2 - terminal constriant set
% % ---------- Terminal set: N_tail-step invariance under LQR ----------
% N_tail = 20;  % how many extra steps you want guaranteed
% 
% % Error at terminal step (around reference)
% sT = x(:,Np+1) - x_ref;   % s_T = x_T - x_ref
% 
% % 0-step condition: terminal error itself must be inside tightened set
% constraints = [constraints, -x_tight_max <= sT <= x_tight_max];
% 
% % j-step conditions: after j steps of LQR, still inside tightened set
% Acl_pow = Acl;
% for j = 1:N_tail
%     constraints = [constraints, ...
%         -x_tight_max <= Acl_pow * sT <= x_tight_max];
%     Acl_pow = Acl_pow * Acl;   % Acl_pow = Acl^j
% end
% 
% % Terminal cost (unchanged)
% objective = objective + sT' * P * sT;


% Solver options
ops = sdpsettings('solver','quadprog','verbose',0);

% Build optimizer: input = x0, output = first nominal input u(:,1)
% mpc = optimizer(constraints, objective, ops, x0, u(:,1));
% Build optimizer: input = x0, output = [first nominal input; optimal objective]
mpc = optimizer(constraints, objective, ops, x0, [u(:,1); objective]);

%% ===================== 3.5) MPC sanity check =============================
[u_test, diag_test] = mpc(zeros(n,1));
if diag_test ~= 0
    warning('MPC test at x0=0 infeasible (code %d). Check tightening / terminal set.', diag_test);
else
    fprintf('MPC test at x0=0: u_opt = [%g %g]^T\n', u_test(1), u_test(2));
end


%% ===================== 4) Simulation (tube MPC) ==========================
T_sim = 8.0;                   % total simulation time [s]
N_sim = round(T_sim/dt);       % number of simulation steps

x_hist = zeros(n, N_sim+1);    % true state trajectory
z_hist = zeros(n, N_sim+1);    % nominal state trajectory (tube center)
u_hist = zeros(m, N_sim);      % applied control
v_hist = zeros(m, N_sim);      % nominal control

% Initial true and nominal states (start aligned at origin)
x_curr = zeros(n,1);
z_curr = x_curr;

x_hist(:,1) = x_curr;
z_hist(:,1) = z_curr;
J_opt      = NaN(1, N_sim);
solve_time = NaN(1, N_sim);

for k = 1:N_sim

    % 1) Solve MPC for the nominal system starting from z_curr
    %    (tube theory: MPC plans z_k, v_k; feedback keeps x_k near z_k)

    t_solve = tic;
    [out, diag] = mpc(z_curr);
    solve_time(k) = toc(t_solve);
    
    if diag ~= 0
        warning('MPC infeasible at step %d (code %d). Stopping.', k, diag);
        break;
    end
    
    v_curr   = out(1:m);
    J_opt(k) = out(m+1);
    
    v_hist(:,k) = v_curr;
        
    % [v_opt, diag] = mpc(z_curr);
    % if diag ~= 0
    %     warning('MPC infeasible at step %d (code %d). Stopping.', k, diag);
    %     break;
    % end
    % v_curr = v_opt(:);    % first nominal input v_0
    % v_hist(:,k) = v_curr;


    % 2) Tube feedback: u_k = v_k - KT * e_k,  e_k = x_k - z_k
    e_curr = x_curr - z_curr;
    u_curr = v_curr - KT * e_curr;

    if k <= 5
        fprintf('k=%d: ||v||=%.4f, ||u||=%.4f, ||x||=%.4f, ||z||=%.4f\n', ...
            k, norm(v_curr), norm(u_curr), norm(x_curr), norm(z_curr));
    end

    u_hist(:,k) = u_curr;

    % 3) Random disturbance (bounded by w_pos_max)
    w_k = [w_pos_max*(2*rand-1);   % uniform in [-w_pos_max, w_pos_max]
           0;
           w_pos_max*(2*rand-1);
           0];

    % 4) True system update with disturbance
    x_curr = A*x_curr + B*u_curr + w_k;

    % (Optional) numerical safety clip to hard bounds (original, not tightened)
    % x_curr = min(max(x_curr, -x_max), x_max);

    % 5) Nominal system update (no disturbance, uses nominal input v_curr)
    z_curr = A*z_curr + B*v_curr;

    % 6) Log trajectories
    x_hist(:,k+1) = x_curr;
    z_hist(:,k+1) = z_curr;
end

time_vec = (0:N_sim)*dt;


%% ===================== 5) Static plots (and save) =========================
outdir = 'mpc_results';
if ~exist(outdir,'dir')
    mkdir(outdir);
end

% ---- Time histories: states & inputs ----
f1 = figure('Name','MPC_time_hist','NumberTitle','off');

%% ------------------ px plot ------------------
subplot(3,1,1);
plot(time_vec, x_hist(1,:), 'b-', 'LineWidth', 1.5); hold on;

% reference
yline(p_ref(1), 'b--', 'LineWidth', 1.2);

% constraints on p_x
yline( x_max(1), 'k--', 'LineWidth', 1.4);
yline(-x_max(1), 'k--', 'LineWidth', 1.4);

% tightened constraints on p_x
yline( x_tight_max(1), 'r--', 'LineWidth', 1.4);
yline(-x_tight_max(1), 'r--', 'LineWidth', 1.4);

% force y-limits so constraints are clearly visible
ylim(1.1*[-x_max(1), x_max(1)]);

xlabel('Time [s]'); ylabel('p_x');
legend('p_x','p_{x,ref}','p_{x,max}','p_{x,min}', 'p_{x,tight}','p_{x,tight}','Location','best');
grid on;

%% ------------------ py plot ------------------
subplot(3,1,2);
plot(time_vec, x_hist(3,:), 'b-', 'LineWidth', 1.5); hold on;

% reference
yline(p_ref(2), 'b--', 'LineWidth', 1.2);

% constraints on p_y
yline( x_max(3), 'k--', 'LineWidth', 1.4);
yline(-x_max(3), 'k--', 'LineWidth', 1.4);

% tightened constraints on p_y
yline( x_tight_max(3), 'r--', 'LineWidth', 1.4);
yline(-x_tight_max(3), 'r--', 'LineWidth', 1.4);

% force y-limits so constraints are clearly visible
ylim(1.1*[-x_max(3), x_max(3)]);

xlabel('Time [s]'); ylabel('p_y');
legend('p_y','p_{y,ref}','p_{y,max}','p_{y,min}', 'p_{y,tight}','p_{y,tight}','Location','best');
grid on;

%% ------------------ Inputs u_x, u_y ------------------
subplot(3,1,3);
% plot(time_vec(1:end-1), u_hist(1,:), 'm-', 'LineWidth', 1.5); hold on;
% plot(time_vec(1:end-1), u_hist(2,:), 'g-', 'LineWidth', 1.5);
plot(time_vec(1:end-1), u_hist(1,:), '-', 'Color', [0 0.5 0.5], 'LineWidth', 1.5); hold on;
plot(time_vec(1:end-1), u_hist(2,:), '-', 'Color', [0.75 0 0], 'LineWidth', 1.5);




% Optionally overlay nominal inputs
% plot(time_vec(1:end-1), v_hist(1,:), 'm--', 'LineWidth', 1.0);
% plot(time_vec(1:end-1), v_hist(2,:), 'g--', 'LineWidth', 1.0);

% input constraints (same for ux, uy here)
yline( u_max(1), 'k--', 'LineWidth', 1.4);
yline(-u_max(1), 'k--', 'LineWidth', 1.4);

% input constraints (same for ux, uy here)
yline( u_tight_max(1), 'r--', 'LineWidth', 1.4);
yline(-u_tight_max(1), 'r--', 'LineWidth', 1.4);

% force y-limits for inputs too
ylim(1.1*[-u_max(1), u_max(1)]);

xlabel('Time [s]'); ylabel('u_x, u_y');
legend('u_x','u_y','u_{max}','u_{min}','u_{tight}','u_{tight}','Location','best');
grid on;


saveas(f1, fullfile(outdir,'mpc_time_hist.png'));
savefig(f1, fullfile(outdir,'mpc_time_hist.fig'));

% ---- XY trajectory with shaded position constraint region ----
f2 = figure('Name','MPC_xy_traj','NumberTitle','off'); hold on; grid on; axis equal;
xlabel('x'); ylabel('y');
title('2D trajectory with robust tightening and disturbance');

% Shaded position constraint box (feasible region)
px_min = -x_max(1);
px_max =  x_max(1);
py_min = -x_max(3);
py_max =  x_max(3);

px_tight_min = -x_tight_max(1);
px_tight_max =  x_tight_max(1);
py_tight_min = -x_tight_max(3);
py_tight_max =  x_tight_max(3);

X_box = [px_min px_max px_max px_min];
Y_box = [py_min py_min py_max py_max];

fill(X_box, Y_box, [0.9 0.9 1.0], 'FaceAlpha', 0.4, ...
     'EdgeColor','k','LineStyle','--','LineWidth',1.0);  % light bluish area


X_tight_box = [px_tight_min px_tight_max px_tight_max px_tight_min];
Y_tight_box = [py_tight_min py_tight_min py_tight_max py_tight_max];

fill(X_tight_box, Y_tight_box, [0.9 0.9 1.0], 'FaceAlpha', 0.7, ...
     'EdgeColor','r','LineStyle','--','LineWidth',1.0);  % light bluish area


% Trajectory and reference
plot(x_hist(1,:), x_hist(3,:), 'b-', 'LineWidth', 1.5);
plot(p_ref(1), p_ref(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

legend({'position constraints', 'Tightened constraints','reference', 'trajectory'}, 'Location','best');

saveas(f2, fullfile(outdir,'mpc_xy_traj.png'));
savefig(f2, fullfile(outdir,'mpc_xy_traj.fig'));



%% ===================== 6) Animation (and save as .mp4) ====================
videoFile = fullfile(outdir,'mpc_animation.mp4');
v = VideoWriter(videoFile,'MPEG-4');
% If you step every 2 samples, effective frame period ~ 2*dt
v.FrameRate = 1/(2*dt);
open(v);

f3 = figure('Name','MPC_animation','NumberTitle','off');
hold on; grid on; axis equal;
xlabel('x'); ylabel('y');
title('MPC-controlled motion with disturbance (animation)');

% Shaded position constraint region (same as above)
fill(X_box, Y_box, [0.9 0.9 1.0], 'FaceAlpha', 0.4, ...
     'EdgeColor','k','LineStyle','--','LineWidth',1.0);

fill(X_tight_box, Y_tight_box, [0.9 0.9 1.0], 'FaceAlpha', 0.7, ...
     'EdgeColor','r','LineStyle','--','LineWidth',1.0);  % light bluish area

% Set axes a bit larger than constraints
ax_margin = 0.3;
xlim([px_min-ax_margin, px_max+ax_margin]);
ylim([py_min-ax_margin, py_max+ax_margin]);

% Reference point
plot(p_ref(1), p_ref(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

% Handles for trajectory and current position
traj_line = plot(NaN, NaN, 'b-', 'LineWidth', 1.5);
agent     = plot(NaN, NaN, 'bo', 'MarkerSize', 8, 'MarkerFaceColor','b');

legend({'position constraints', 'Tightened constraints','reference', 'trajectory'}, 'Location','best');

for k = 1:2:N_sim+1   % show every 2nd sample
    px = x_hist(1,1:k);
    py = x_hist(3,1:k);
    
    set(traj_line, 'XData', px, 'YData', py);
    set(agent,     'XData', px(end), 'YData', py(end));
    
    drawnow limitrate;
    frame = getframe(f3);
    writeVideo(v, frame);
end

close(v);

% Also save the final animation frame as image/fig
saveas(f3, fullfile(outdir,'mpc_animation_frame.png'));
savefig(f3, fullfile(outdir,'mpc_animation_frame.fig'));

disp(['Saved plots and animation in folder: ' outdir]);

%% ===================== 7) Report-ready plots + QP diagnostics =============
% Uses: outdir, time_vec, x_hist, z_hist, u_hist, v_hist,
%       x_max, x_tight_max, u_max, u_tight_max, e_max, KT, p_ref,
%       J_opt, solve_time, A, B, Q, R, P, x_ref, Np

outdir2 = fullfile(outdir,'report_plots');
if ~exist(outdir2,'dir'); mkdir(outdir2); end

% --- Effective horizon (robust to early break) -----------------------------
K_eff = find(isfinite(J_opt), 1, 'last');   % last solved MPC step
if isempty(K_eff), K_eff = 0; end

N_eff = K_eff + 1;                          % states logged up to k+1
t     = time_vec(1:N_eff);
Ns    = K_eff;                               % inputs exist for 1:K_eff
tt    = time_vec(1:Ns);                      % time for inputs

xH = x_hist(:,1:N_eff);                      % true state
zH = z_hist(:,1:N_eff);                      % nominal state
eH = xH - zH;                                % tube error

if Ns > 0
    uH = u_hist(:,1:Ns);                     % applied
    vH = v_hist(:,1:Ns);                     % nominal
    u_fb = -KT * eH(:,1:Ns);                 % feedback correction
else
    uH = zeros(size(u_hist,1),0);
    vH = zeros(size(v_hist,1),0);
    u_fb = zeros(size(u_hist,1),0);
end

% --- Plot style defaults ---------------------------------------------------
set(groot,'defaultFigureColor','w');
set(groot,'defaultAxesFontSize',12);
set(groot,'defaultAxesLineWidth',1.0);
set(groot,'defaultLineLineWidth',1.8);

% --- Helper: shaded band (tube band) --------------------------------------
bandplot = @(x, lo, hi) fill([x fliplr(x)], [lo fliplr(hi)], ...
    [0.85 0.85 0.85], 'EdgeColor','none', 'FaceAlpha',0.35,...
    'HandleVisibility','off');

%% -------------------------------------------------------------------------
%% Plot 1) Positions + inputs: TRUE x, NOMINAL z, tube band, bounds, u_x/u_y
%% -------------------------------------------------------------------------
f1 = figure('Name','Tube_Positions_Inputs_Time','NumberTitle','off');

% ---- p_x ----
subplot(3,1,1); hold on; grid on;
lo = zH(1,:) - e_max(1);
hi = zH(1,:) + e_max(1);
bandplot(t, lo, hi);
plot(t, zH(1,:), 'k--', 'DisplayName','z_x (nominal)');
plot(t, xH(1,:), 'b-',  'DisplayName','x_x (true)');
yline(p_ref(1), 'b--',  'DisplayName','p_{x,ref}');
yline( x_max(1), 'k-',  'DisplayName','p_{x,max}');
yline(-x_max(1), 'k-',  'HandleVisibility','off');
yline( x_tight_max(1), 'r--', 'DisplayName','p_{x,tight}');
yline(-x_tight_max(1), 'r--', 'HandleVisibility','off');
ylim(1.15*[-x_max(1), x_max(1)]);
xlabel('Time [s]'); ylabel('p_x');
legend('Location','best');

% ---- p_y ----
subplot(3,1,2); hold on; grid on;
lo = zH(3,:) - e_max(3);
hi = zH(3,:) + e_max(3);
bandplot(t, lo, hi);
plot(t, zH(3,:), 'k--', 'DisplayName','z_y (nominal)');
plot(t, xH(3,:), 'b-',  'DisplayName','x_y (true)');
yline(p_ref(2), 'b--',  'DisplayName','p_{y,ref}');
yline( x_max(3), 'k-',  'DisplayName','p_{y,max}');
yline(-x_max(3), 'k-',  'HandleVisibility','off');
yline( x_tight_max(3), 'r--', 'DisplayName','p_{y,tight}');
yline(-x_tight_max(3), 'r--', 'HandleVisibility','off');
ylim(1.15*[-x_max(3), x_max(3)]);
xlabel('Time [s]'); ylabel('p_y');
legend('Location','best');

% ---- inputs u_x, u_y (applied) ----
subplot(3,1,3); hold on; grid on;

if Ns > 0
    plot(tt, uH(1,:), 'b-', 'DisplayName','u_x (applied)');
    plot(tt, uH(2,:), 'k--','DisplayName','u_y (applied)');

    % original bounds
    yline( u_max(1), 'k-',  'DisplayName','u_{max}');
    yline(-u_max(1), 'k-',  'HandleVisibility','off');

    % tightened nominal bounds (shown as reference)
    yline( u_tight_max(1), 'r--', 'DisplayName','u_{tight}');
    yline(-u_tight_max(1), 'r--', 'HandleVisibility','off');

    ylim(1.15*[-u_max(1), u_max(1)]);
    xlabel('Time [s]'); ylabel('u_x, u_y');
    legend('Location','best');
else
    text(0.1,0.5,'No inputs available (simulation ended early).');
    axis off;
end

sgtitle('True vs nominal positions (tube) and applied inputs');
export_fig(f1, outdir2, 'pos_true_nominal_tube_with_inputs');


%% -------------------------------------------------------------------------
%% Plot 2) Tube verification: e = x - z within Â±e_max
%% -------------------------------------------------------------------------
f2 = figure('Name','Tube_Error_Verification','NumberTitle','off');
names = {'p_x','v_x','p_y','v_y'};

for i = 1:4
    subplot(2,2,i); hold on; grid on;
    plot(t, eH(i,:), 'b-', 'DisplayName',['e_' names{i}]);
    yline( e_max(i), 'r--', 'DisplayName','+e_{max}');
    yline(-e_max(i), 'r--', 'DisplayName','-e_{max}');
    xlabel('Time [s]'); ylabel(['e_' names{i}]);
    title(['Error ' names{i} ' within \pm e_{max}']);
    if i==1, legend('Location','best'); end
end
sgtitle('Tube error bound check');
export_fig(f2, outdir2, 'error_vs_emax');

%% -------------------------------------------------------------------------
%% Plot 3) Inputs decomposition: v, feedback -K e, applied u + bounds
%% -------------------------------------------------------------------------
f3 = figure('Name','Inputs_Decomposition','NumberTitle','off');

if Ns > 0
    subplot(2,1,1); hold on; grid on;
    plot(tt, vH(1,:),   'k--', 'DisplayName','v_x (nominal)');
    plot(tt, u_fb(1,:), 'm-',  'DisplayName','u_{fb,x}=-K_T e');
    plot(tt, uH(1,:),   'b-',  'DisplayName','u_x (applied)');
    yline( u_max(1), 'k-', 'DisplayName','u_{max}');
    yline(-u_max(1), 'k-', 'HandleVisibility','off');
    yline( u_tight_max(1), 'r--', 'DisplayName','u_{tight}');
    yline(-u_tight_max(1), 'r--', 'HandleVisibility','off');
    ylim(1.15*[-u_max(1), u_max(1)]);
    xlabel('Time [s]'); ylabel('u_x');
    legend('Location','best');

    subplot(2,1,2); hold on; grid on;
    plot(tt, vH(2,:),   'k--', 'DisplayName','v_y (nominal)');
    plot(tt, u_fb(2,:), 'm-',  'DisplayName','u_{fb,y}=-K_T e');
    plot(tt, uH(2,:),   'b-',  'DisplayName','u_y (applied)');
    yline( u_max(2), 'k-', 'DisplayName','u_{max}');
    yline(-u_max(2), 'k-', 'HandleVisibility','off');
    yline( u_tight_max(2), 'r--', 'DisplayName','u_{tight}');
    yline(-u_tight_max(2), 'r--', 'HandleVisibility','off');
    ylim(1.15*[-u_max(2), u_max(2)]);
    xlabel('Time [s]'); ylabel('u_y');
    legend('Location','best');

    sgtitle('Input decomposition (tube MPC): u = v - K_T(x-z)');
    export_fig(f3, outdir2, 'inputs_decomposition');
else
    clf(f3); text(0.1,0.5,'No input samples available.'); axis off;
end

%% -------------------------------------------------------------------------
%% Plot 4) XY view: X box, X_tight box, true x vs nominal z, tube rectangles
%% -------------------------------------------------------------------------
f4 = figure('Name','XY_Tube_View','NumberTitle','off');
hold on; grid on; axis equal;
xlabel('p_x'); ylabel('p_y');
title('XY view: X, X_{tight}, nominal z, true x, and tube');

% Original position box
px_min = -x_max(1); px_max = x_max(1);
py_min = -x_max(3); py_max = x_max(3);
patch([px_min px_max px_max px_min], [py_min py_min py_max py_max], ...
    [0.90 0.90 1.00], 'FaceAlpha',0.18, 'EdgeColor','k', ...
    'LineStyle','-', 'LineWidth',1.2, 'DisplayName','X (pos)');

% Tightened position box
px_tmin = -x_tight_max(1); px_tmax = x_tight_max(1);
py_tmin = -x_tight_max(3); py_tmax = x_tight_max(3);
patch([px_tmin px_tmax px_tmax px_tmin], [py_tmin py_tmin py_tmax py_tmax], ...
    [1.00 0.90 0.90], 'FaceAlpha',0.22, 'EdgeColor','r', ...
    'LineStyle','--', 'LineWidth',1.4, 'DisplayName','X_{tight} (pos)');

% Tube rectangles at a few snapshots (z Â± emax on positions)
idx = unique(round(linspace(1, N_eff, min(10,N_eff))));
for kk = idx
    cx = zH(1,kk); cy = zH(3,kk);
    patch([cx-e_max(1) cx+e_max(1) cx+e_max(1) cx-e_max(1)], ...
          [cy-e_max(3) cy-e_max(3) cy+e_max(3) cy+e_max(3)], ...
          [0.85 0.85 0.85], 'FaceAlpha',0.10, ...
          'EdgeColor',[0.2 0.2 0.2], 'LineWidth',1.0, ...
          'HandleVisibility','off');
end
plot(NaN,NaN,'-','Color',[0.2 0.2 0.2],'LineWidth',1.0,'DisplayName','tube (z Â± e_{max})');

% True and nominal trajectories (REAL values shown)
plot(zH(1,:), zH(3,:), 'k--', 'LineWidth',1.8, 'DisplayName','z trajectory (nominal)');
plot(xH(1,:), xH(3,:), 'b-',  'LineWidth',2.0, 'DisplayName','x trajectory (true)');

plot(xH(1,1), xH(3,1), 'ko', 'MarkerFaceColor','k', 'DisplayName','start');
plot(p_ref(1), p_ref(2), 'rx', 'MarkerSize',10, 'LineWidth',2, 'DisplayName','reference');

xlim([px_min-0.3, px_max+0.3]);
ylim([py_min-0.3, py_max+0.3]);
legend('Location','bestoutside');

export_fig(f4, outdir2, 'xy_true_nominal_tube_boxes');

%% -------------------------------------------------------------------------
%% Plot 5) Feasibility margins: min slack vs time (certificate-style plot)
%% -------------------------------------------------------------------------
margin_x_true  = min(x_max       - abs(xH), [], 1);
margin_z_tight = min(x_tight_max - abs(zH), [], 1);

if Ns > 0
    margin_u_true  = min(u_max       - abs(uH), [], 1);
    margin_v_tight = min(u_tight_max - abs(vH), [], 1);
else
    margin_u_true  = [];
    margin_v_tight = [];
end

f5 = figure('Name','Constraint_Margins','NumberTitle','off');

subplot(2,1,1); hold on; grid on;
plot(t, margin_x_true,  'b-', 'DisplayName','min(x_{max}-|x|) true');
plot(t, margin_z_tight, 'k--','DisplayName','min(x_{tight}-|z|) nominal');
yline(0,'r-','DisplayName','0');
xlabel('Time [s]'); ylabel('state margin');
title('State feasibility margins');
legend('Location','best');

subplot(2,1,2); hold on; grid on;
if Ns > 0
    plot(tt, margin_u_true,  'b-', 'DisplayName','min(u_{max}-|u|) applied');
    plot(tt, margin_v_tight, 'k--','DisplayName','min(u_{tight}-|v|) nominal');
end
yline(0,'r-','DisplayName','0');
xlabel('Time [s]'); ylabel('input margin');
title('Input feasibility margins');
legend('Location','best');

sgtitle('Margins: direct visual certificate of constraint satisfaction');
export_fig(f5, outdir2, 'constraint_margins');

%% -------------------------------------------------------------------------
%% Plot 6) QP behavior in closed-loop: J_opt(k) and solve_time(k)
%% -------------------------------------------------------------------------
if K_eff > 0
    kidx = 1:K_eff;

    f6 = figure('Name','QP_Behavior_ClosedLoop','NumberTitle','off');

    subplot(2,1,1); hold on; grid on;
    plot(kidx, J_opt(kidx), 'LineWidth', 1.8);
    xlabel('MPC step k'); ylabel('J^*(k)');
    title('Closed-loop optimal QP objective');

    subplot(2,1,2); hold on; grid on;
    plot(kidx, solve_time(kidx), 'LineWidth', 1.8);
    xlabel('MPC step k'); ylabel('solve time [s]');
    title('QP solve time per step');

    sgtitle('How the QP behaves online (objective + timing)');
    export_fig(f6, outdir2, 'qp_closedloop_objective_time');
end

%% -------------------------------------------------------------------------
%% Plot 7) "How quadprog works": solver summary + KKT residuals for one QP
%% -------------------------------------------------------------------------
x0_diag = z_hist(:,1);  % representative initial nominal state

% Build condensed prediction matrices: X = Sx*x0 + Su*U for X=[x1;..;xN]
[Sx, Su] = build_pred_mats(A, B, n, m, Np);

% Reference stack for x1..xN
Xref = repmat(x_ref, Np, 1);

% Q block for x1..x_{N-1} with Q, and xN with P
if Np == 1
    Qblk = P;
else
    Qblk = blkdiag(kron(eye(Np-1), Q), P);
end

% R block for u0..u_{N-1}
Rblk = kron(eye(Np), R);

% Condensed convex QP: 0.5*U'HU + f'U
H = 2*(Su' * Qblk * Su + Rblk);
f = 2*(Su' * Qblk * (Sx*x0_diag - Xref));

% Inequality constraints (tightened):
% -x_tight <= X <= x_tight,  -u_tight <= U <= u_tight
xT = repmat(x_tight_max, Np, 1);
uT = repmat(u_tight_max, Np, 1);

Aineq = [ Su;
         -Su;
          eye(m*Np);
         -eye(m*Np)];

bineq = [ xT - Sx*x0_diag;
          xT + Sx*x0_diag;
          uT;
          uT ];

% Solve ONE condensed QP with quadprog (no OutputFcn needed)
optsQP = optimoptions('quadprog', ...
    'Algorithm','interior-point-convex', ...
    'Display','off');

[U_star, fval_star, exitflag_star, output_star, lambda_star] = quadprog( ...
    H, f, Aineq, bineq, [], [], [], [], [], optsQP);

% ---- Build KKT-style residuals (certificate) ------------------------------
primal_violation = max(0, max(Aineq*U_star - bineq));     % should be ~0
slack = bineq - Aineq*U_star;

if isstruct(lambda_star) && isfield(lambda_star,'ineqlin') && ~isempty(lambda_star.ineqlin)
    lam = lambda_star.ineqlin;
    stationarity = norm(H*U_star + f + Aineq'*lam, inf);   % should be small
    complementarity = max(abs(lam .* slack));              % should be small
else
    stationarity = NaN;
    complementarity = NaN;
end

% ---- Plot: solver summary + residuals -------------------------------------
f7 = figure('Name','Quadprog_Summary_KKT','NumberTitle','off');

subplot(2,1,1); axis off;
txt = {
    sprintf('quadprog condensed QP (one instance)')
    sprintf('exitflag = %d', exitflag_star)
    sprintf('iterations = %d', output_star.iterations)
    sprintf('fval = %.6g', fval_star)
    output_star.message
    };
text(0.01, 0.95, txt, 'VerticalAlignment','top', 'FontSize', 11);

subplot(2,1,2); grid on; hold on;
vals = [primal_violation, stationarity, complementarity];
bar(vals);
set(gca,'XTickLabel',{'primal viol','stationarity','complementarity'});
ylabel('magnitude');
title('KKT-style residuals (smaller is better)');
set(gca,'YScale','log');   % usually these span orders of magnitude

export_fig(f7, outdir2, 'qp_quadprog_kkt_certificate');

%% -------------------------------------------------------------------------
%% Plot 8) Sparsity patterns of the condensed QP (H and Aineq)
%% -------------------------------------------------------------------------
f8 = figure('Name','QP_Sparsity','NumberTitle','off');

subplot(1,2,1);
spy(H);
title('Sparsity of Hessian H');

subplot(1,2,2);
spy(Aineq);
title('Sparsity of inequality matrix A_{ineq}');

sgtitle('Condensed QP structure (convex + sparse)');
export_fig(f8, outdir2, 'qp_sparsity');

disp(['Saved report plots + QP diagnostics in: ' outdir2]);

%% ===================== Local functions (script-safe) ======================
function export_fig(figH, outDir, baseName)
%EXPORT_FIG Robust export to PDF+PNG with permission/lock fallback (Windows-safe)

    if ~exist(outDir,'dir')
        mkdir(outDir);
    end

    % --- Test write permission in outDir -----------------------------------
    testfile = fullfile(outDir, ['__write_test__' char(java.util.UUID.randomUUID) '.tmp']);
    canWrite = true;
    fid = fopen(testfile,'w');
    if fid < 0
        canWrite = false;
    else
        fclose(fid);
        delete(testfile);
    end

    if ~canWrite
        % Fallback to a guaranteed-writable folder
        outDir = fullfile(tempdir, 'mpc_report_plots');
        if ~exist(outDir,'dir'); mkdir(outDir); end
        warning('No write permission. Saving plots to: %s', outDir);
    end

    pdfPath = fullfile(outDir, [baseName '.pdf']);
    pngPath = fullfile(outDir, [baseName '.png']);

    % --- If PDF is locked (open), write a timestamped name -----------------
    if exist(pdfPath,'file')
        try
            % quick lock test: try opening for append
            fid = fopen(pdfPath,'a');
            if fid < 0
                error('locked');
            end
            fclose(fid);
        catch
            stamp  = datestr(now,'yyyymmdd_HHMMSS');
            pdfPath = fullfile(outDir, [baseName '_' stamp '.pdf']);
        end
    end

    % --- Export ------------------------------------------------------------
    try
        exportgraphics(figH, pdfPath, 'ContentType','vector');
    catch ME
        % If vector PDF fails for any reason, fall back to PNG-only
        warning('PDF export failed (%s). Saving PNG only.', ME.message);
    end

    % Always save a high-res PNG (usually never locked)
    exportgraphics(figH, pngPath, 'Resolution', 300);
end


function [Sx, Su] = build_pred_mats(A, B, n, m, N)
% Build prediction matrices for X = [x1;...;xN] = Sx*x0 + Su*U, U=[u0;...;u_{N-1}]
    Sx = zeros(n*N, n);
    Su = zeros(n*N, m*N);
    A_pow = eye(n);
    for i = 1:N
        A_pow = A_pow * A;  % A^i
        Sx((i-1)*n+1:i*n, :) = A_pow;
        for j = 1:i
            A_ij = A^(i-j);
            Su((i-1)*n+1:i*n, (j-1)*m+1:j*m) = A_ij * B;
        end
    end
end

