%% SMPC Implementation for 2D Double Integrator System
clc, clearvars, close all

%% ================================
% 1) System model
%% ================================
dt = 0.05;

A = [1 dt 0  0;
     0 1  0  0;
     0 0  1 dt;
     0 0  0  1];

B = [0.5*dt^2 0;
     dt       0;
     0 0.5*dt^2;
     0       dt];

n = 4; m = 2;


%% ================================
% 2) Costs and hard constraints
%% ================================
x_max = [2; 1; 2; 1];
u_max = [0.65; 0.65];

Q = diag([5 1 5 1]);
R = 0.1*eye(m);
[~, P, ~] = dlqr(A, B, Q, R);

p_ref = [1.5; 1.5];
x_ref = [p_ref(1); 0; p_ref(2); 0];


%% ================================
% 3) Disturbance model + chance tightening
%% ================================
w_pos_max = 0.02;
sigma_w   = w_pos_max / sqrt(3);
W = diag([sigma_w^2 0 sigma_w^2 0]);

eps_x = 0.01;
eps_u = 0.01;
beta_x = norminv(1 - eps_x/2);
beta_u = norminv(1 - eps_u/2);


%% ================================
% 4) Feedback gain for closed-loop SMPC
%% ================================
Q_fb = eye(n);
R_fb = 5*eye(m);
[K,~,~] = dlqr(A, B, Q_fb, R_fb);
K = -K;

Acl = A + B*K;


%% ================================
% 5) Closed-loop covariance prediction
%% ================================
Np = 50;
T_sim = 8;
N_sim = round(T_sim/dt);
N_cov = max(Np, N_sim) + 1;

Sigma = cell(N_cov,1);
Sigma{1} = zeros(n);

for k = 1:N_cov-1
    Sigma{k+1} = Acl * Sigma{k} * Acl' + W;
end

sigma_x = zeros(n, N_cov);
sigma_u = zeros(m, N_cov);

for k = 1:N_cov
    S = Sigma{k};
    sigma_x(:,k) = sqrt(diag(S));
    sigma_u(:,k) = sqrt(diag(K*S*K'));
end


%% ================================
% 6) SMPC optimization (nominal problem)
%% ================================
x0 = sdpvar(n,1);
z  = sdpvar(n, Np+1, 'full');
v  = sdpvar(m, Np,   'full');

constraints = [];
objective   = 0;

constraints = [constraints, z(:,1) == x0];

for k = 1:Np
    constraints = [constraints, z(:,k+1) == A*z(:,k) + B*v(:,k)];

    x_tight = x_max - beta_x * sigma_x(:,k);
    constraints = [constraints, -x_tight <= z(:,k) <= x_tight];

    u_tight = u_max - beta_u * sigma_u(:,k);
    constraints = [constraints, -u_tight <= v(:,k) <= u_tight];

    objective = objective + (z(:,k)-x_ref)'*Q*(z(:,k)-x_ref) + v(:,k)'*R*v(:,k);
end

x_tight_T = x_max - beta_x * sigma_x(:,Np+1);
constraints = [constraints, -x_tight_T <= z(:,Np+1) <= x_tight_T];
objective   = objective + (z(:,Np+1)-x_ref)'*P*(z(:,Np+1)-x_ref);

ops = sdpsettings('solver','quadprog','verbose',0);
mpc = optimizer(constraints, objective, ops, x0, {v(:,1), objective});


%% ================================
% 7) Closed-loop simulation
%% ================================
x_hist = zeros(n,N_sim+1);
z_hist = zeros(n,N_sim+1);
v_hist = zeros(m,N_sim);
u_hist = zeros(m,N_sim);

x_curr = zeros(n,1);
z_curr = zeros(n,1);

x_hist(:,1) = x_curr;
z_hist(:,1) = z_curr;

J_hist   = zeros(1, N_sim);
solve_t  = zeros(1, N_sim);

for k = 1:N_sim

    tic;
    out = mpc(z_curr);
    solve_t(k) = toc;
    
    v_nom = out{1};
    J_hist(k) = out{2};

    v_hist(:,k) = v_nom;

    e = x_curr - z_curr;
    u_true = v_nom + K*e;
    u_hist(:,k) = u_true;

    w = [w_pos_max*(2*rand-1); 0; w_pos_max*(2*rand-1); 0];

    x_curr = A*x_curr + B*u_true + w;
    z_curr = A*z_curr + B*v_nom;

    x_hist(:,k+1) = x_curr;
    z_hist(:,k+1) = z_curr;
end

time = (0:N_sim)*dt;


%% ================================
% 8) 3σ tube trajectory plot
%% ================================
figure
set(gcf, 'Color', 'w')
hold on
grid on
axis equal
title('XY view: nominal z, true x and 3σ tube')
xlim([-2.2 2.2])
ylim([-2.2 2.2])
xlabel('p_x')
ylabel('p_y')

% Shaded constraints
x_c = [-x_max(1),  x_max(1),  x_max(1), -x_max(1)];
y_c = [-x_max(3), -x_max(3),  x_max(3),  x_max(3)];
h1 = patch(x_c, y_c, [0.9 0.9 1.0], 'FaceAlpha',0.4, ...
           'EdgeColor','k','LineWidth',1);
angles = linspace(0, 2*pi, 80);

for k = 1:3:N_sim-90
    idx = min(k, Np+1);
    sx = sigma_x(1, idx);
    sy = sigma_x(3, idx);

    cx = z_hist(1,k) + 3*sx*cos(angles);
    cy = z_hist(3,k) + 3*sy*sin(angles);

    patch(cx, cy, [1 0 0], 'FaceAlpha',0.075, 'EdgeColor','none')
end

hTube = patch(NaN,NaN,[1 0 0], 'FaceAlpha',0.12,'EdgeColor','none');

h2 = plot(z_hist(1,:), z_hist(3,:), 'k--', 'LineWidth',2);
h3 = plot(x_hist(1,:), x_hist(3,:), 'b-', 'LineWidth',1.5);
h4 = plot(p_ref(1), p_ref(2), 'rx','MarkerSize',10,'LineWidth',2);
h5 = plot(0, 0, 'k.','MarkerSize',15,'LineWidth',2);

legend([h1, hTube, h2, h3, h5, h4], ...
       {'X (pos)','3σ tube around z','z trajectory (nominal)','x  trajectory (true)', 'start', 'reference'}, ...
       'Location','northwest')

%% ================================
% 9) Position plots with ±3σ bounds and Input plots
%% ================================
figure
subplot(3,1,1)
hold on
grid on
px_nom = z_hist(1,:);
px_true = x_hist(1,:);
px_sigma = 3 * sigma_x(1,1:N_sim+1);

h_fill = fill([time, fliplr(time)], [px_nom - px_sigma, fliplr(px_nom + px_sigma)], ...
     [1 0 0], 'FaceAlpha', 0.15, 'EdgeColor','none');

h_nom = plot(time, px_nom,'k--','LineWidth',1.6);
h_true = plot(time, px_true,'b','LineWidth',1.4);
h1 = yline(p_ref(1),'b--');
ylim([-2.2 2.2])
h2 = yline(2, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5);
yline(-2, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5)

px_tight = x_max(1) - beta_x * sigma_x(1,1:N_sim+1);
h3 = plot( time, px_tight, 'r--', 'LineWidth', 1);
plot(time, -px_tight(end), 'r--', 'LineWidth', 1)

legend([h_fill, h_nom, h_true, h1, h2, h3], ...
       {'3$\sigma$ tube', ...
        '$z_x$ (nominal)', ...
        '$x_x$ (true)', ...
        '$p_{x,\mathrm{ref}}$', ...
        '$p_{x,\max}$', ...
        '$p_{x,\mathrm{tight}}$'}, ...
       'Interpreter','latex');
xlabel('Time [s]')
ylabel('p_x')

subplot(3,1,2)
hold on
grid on
py_nom = z_hist(3,:);
py_true = x_hist(3,:);
py_sigma = 3 * sigma_x(3,1:N_sim+1);

h_fill = fill([time, fliplr(time)], [py_nom - py_sigma, fliplr(py_nom + py_sigma)], ...
     [1 0 0], 'FaceAlpha', 0.15, 'EdgeColor','none');

h_nom = plot(time, py_nom,'k--','LineWidth',1.6);
h_true = plot(time, py_true,'b','LineWidth',1.4);
h1 = yline(p_ref(2),'b--');
ylim([-2.2 2.2])
h2 = yline(2, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5);
yline(-2, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5)

py_tight = x_max(3) - beta_x * sigma_x(3,1:N_sim+1);
h3 = plot(time, py_tight, 'r--', 'LineWidth', 1);
plot(time, -py_tight, 'r--', 'LineWidth', 1)

legend([h_fill, h_nom, h_true, h1, h2, h3], ...
       {'3$\sigma$ tube', ...
        '$z_y$ (nominal)', ...
        '$x_y$ (true)', ...
        '$p_{y,\mathrm{ref}}$', ...
        '$p_{y,\max}$', ...
        '$p_{y,\mathrm{tight}}$'}, ...
       'Interpreter','latex');
xlabel('Time [s]')
ylabel('p_y')

subplot(3,1,3)
ylim([-0.77 0.77])
hold on
grid on
h1 = plot(time(1:end-1), v_hist(1,:), 'k--','LineWidth',1.6);
h2 = plot(time(1:end-1), u_hist(1,:), 'b','LineWidth',1.4);
plot(time(1:end-1), v_hist(2,:), 'k--','LineWidth',1.6)
h3 = plot(time(1:end-1), u_hist(2,:), 'r','LineWidth',1.4);
h4 = yline(0.7,'Color',[0.7 0.7 0.7], 'LineWidth', 1.5);
yline(-0.7,'Color',[0.7 0.7 0.7], 'LineWidth', 1.5)
u_tight = 0.7 - beta_u * sigma_u(:,1:N_sim);
h5 = plot(time(1:end-1), u_tight(1,:), 'r--', 'LineWidth', 1);
plot(time(1:end-1), -u_tight(1,:), 'r--', 'LineWidth', 1)
legend([h1, h2, h3, h4, h5], ...
       {'$v_x$, $v_y$ (nominal)', ...
        '$u_x$ (true)', ...
        '$u_y$ (true)', ...
        '$u_{x,max}, u_{y,max}$', ...
        '$u_{x,tight}, u_{y,tight}$'}, ...
       'Interpreter','latex', 'Location','best');
xlabel('Time [s]')
ylabel('p_y')
set(gcf, 'Color','w')
sgtitle('True vs nominal positions and applied inputs')

%% ================================
% 12) Velocity Constraints Plots
%% ================================
figure

% v_x
subplot(2,1,1)
hold on
grid on
plot(time, x_hist(2,:), 'b', 'LineWidth', 1.5)

% Hard bounds (reuse x_max(2))
yline( x_max(2), 'k--', 'LineWidth', 1.3)
yline(-x_max(2), 'k--', 'LineWidth', 1.3)

xlabel('Time [s]')
ylabel('v_x')
title('Velocity Constraint Visualization')
legend('v_x','v_{x,max}','v_{x,min}', ...
       'Location','northeast')

% v_y 
subplot(2,1,2)
hold on
grid on
plot(time, x_hist(4,:), 'b', 'LineWidth', 1.5)

yline( x_max(4), 'k--', 'LineWidth', 1.3)
yline(-x_max(4), 'k--', 'LineWidth', 1.3)

xlabel('Time [s]')
ylabel('v_y')

legend('v_y','v_{y,max}','v_{y,min}', ...
       'Location','northeast')


%% ================================
% 13) Objective Value and Solution Time Plots
%% ================================
figure
set(gcf, 'Color', 'w')
subplot(2,1,1)
hold on
grid on
plot(1:N_sim, J_hist,'b', 'LineWidth', 2)
xlabel('MPC step k')
ylabel('J^*(k)')
title('Closed-loop optimal QP objective')
set(gca,'FontSize',11)

subplot(2,1,2)
hold on
grid on
plot(1:N_sim, solve_t,'b', 'LineWidth', 1.8)
ylabel('solve time [s]')
xlabel('MPC step k')
title('QP solve time per step')
set(gca,'FontSize',11)
sgtitle('How the QP behaves online (objective + timing)')

%% ================================
% 14) Animation
%% ================================
figure('Name','SMPC Animation','NumberTitle','off')
hold on
grid on
axis equal
title('Closed-Loop SMPC Animation')
xlabel('p_x'); ylabel('p_y')
xlim([-2 2]); ylim([-2 2])

rectangle('Position',[-x_max(1), -x_max(3), 2*x_max(1), 2*x_max(3)], ...
    'EdgeColor','k','LineWidth',1.4)
plot(p_ref(1), p_ref(2), 'rx','MarkerSize',12,'LineWidth',2)

traj_nom  = plot(NaN, NaN, 'b-', 'LineWidth', 1.8);
traj_true = plot(NaN, NaN, 'm-', 'LineWidth', 1.8);
agent_nom = plot(NaN, NaN, 'bo','MarkerFaceColor','b');
agent_true= plot(NaN, NaN, 'mo','MarkerFaceColor','m');

makeVideo = true;
if makeVideo
    v = VideoWriter('SMPC_animation.mp4','MPEG-4');
    v.FrameRate = 1/(2*dt);
    open(v)
end

angles = linspace(0, 2*pi, 100);

for k = 1:2:N_sim+1

    set(traj_nom,  'XData', z_hist(1,1:k), 'YData', z_hist(3,1:k));
    set(traj_true, 'XData', x_hist(1,1:k), 'YData', x_hist(3,1:k));
    set(agent_nom, 'XData', z_hist(1,k), 'YData', z_hist(3,k));
    set(agent_true,'XData', x_hist(1,k), 'YData', x_hist(3,k));

    idx = min(k, Np+1);
    sx = sigma_x(1, idx);
    sy = sigma_x(3, idx);

    cx = z_hist(1,k) + 3*sx*cos(angles);
    cy = z_hist(3,k) + 3*sy*sin(angles);

    if k <= 50
        alpha_val = 0.05;
    elseif k >= 100
        alpha_val = 0;
    else
        alpha_val = 0.02;
    end

    patch(cx, cy, [1 0 0], 'FaceAlpha', alpha_val, 'EdgeColor','none')

    drawnow
    if makeVideo
        writeVideo(v, getframe(gcf))
    end
end

if makeVideo
    close(v)
end

