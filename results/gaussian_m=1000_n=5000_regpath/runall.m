%% runall script
%   This script compares the BPDN_inclusions_solver, lasso_fista_solver,
%   and glmnet solver at different hyperparameter values on Gaussian data.
%
%   gaussian data (fixed grid, regularization path)


%% TODO
% 1) Create a separate test function for the basis pursuit solution!
%    Apparently something may be wrong?

%% Input
% Nb of samples and features
m = 1000;
n = 5000;

% Signal-to-noise ratio, value of nonzero coefficients, and
% proportion of nonzero coefficients in the signal.
SNR = 1;
val_nonzero = 1;
prop = 0.02;

% Tolerance levels
tol = 1e-08;
tol_glmnet = 1e-04;
tol_fista = tol;

% Grid of hyperparameters
spacing = -0.01;
max = 0.95;
min = 0.01;

% Max nb of iterations for the FISTA solver
max_iters = 50000;
min_iters = 100;



%% Generate data
% Set random seed and generate Gaussian data
rng('default')
[A,b] = generate_gaussian_data(m,n,SNR,val_nonzero,prop);

% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution
t0 = norm(A.'*b,inf);
x0 = zeros(n,1);
p0 = -b/t0;

% Generate desired grid of hyperparameters
t = t0 * (max:spacing:min);
kmax = length(t);


%% Solve BPDN using glmnet, fista, and differential inclusions
%   Note: The solve are done sequentially but do NOT reuse old solutions
%   as warm starts. We resolve each problem from scratch separately.
sol_incl_x = zeros(n,kmax);
sol_fista_x = zeros(n,kmax);
sol_glmnet_x = zeros(n,kmax);

sol_incl_p = zeros(m,kmax);
sol_fista_p = zeros(m,kmax);
sol_glmnet_p = zeros(m,kmax); 


%%%%%%%%%%%%%%%%%%%%
% GLMNET
%%%%%%%%%%%%%%%%%%%%
%   Note 1: The matrix A and response vector b must be rescaled
%   by a factor of sqrt(m) due to how its implemented.

%   Note 2: The lasso function returns the solutions of the
%   regularization path starting from its lowest value.

%   Note 3: Unlike the exact lasso algorithm, the dual solution is Ax-b.

%   Note 4: The GLMNET algorithm is more sensitive to the tolerance, so one
%   should raise it.

warning('off');
time_glmnet_alg = 0;

% Run MATLAB's native lasso solver, flip it, and rescale the dual solution
disp('Running GLMNET algorithm...')
tic
sol_glmnet_x = lasso(sqrt(m)*A,sqrt(m)*b, 'lambda', t, ...
    'Intercept', false, 'RelTol', tol_glmnet);
sol_glmnet_x = flip(sol_glmnet_x,2);
for k=1:1:kmax
    sol_glmnet_p(:,k) = (A*sol_glmnet_x(:,k)-b)/t(k);
end
time_glmnet_alg = time_glmnet_alg + toc;
disp('Done.')
disp(' ')



%%%%%%%%%%%%%%%%%%%%%%%%%
% FISTA + selection rule
%%%%%%%%%%%%%%%%%%%%%%%%%
% % Compute the L22 norm of the matrix A and tau parameter for FISTA
% disp('Running FISTA algorithm...')
% tic
% L22 = svds(A,1)^2;
% time_fista_L22 = toc;
% 
% % Computer some parameters and options for FISTA
% tau = 1/L22;
% 
% % Run the FISTA solver and rescale the dual solution
% tic
% 
% % k == 1
% [sol_fista_x(:,1),sol_fista_p(:,1),num_iters] = ...
%         lasso_fista_solver(x0,p0,t(1),...
%         A,b,tau,max_iters,tol_fista,min_iters);
% sol_fista_p(:,1) = sol_fista_p(:,1)/t(1);
% 
% % k >= 2
% for k=2:1:kmax
%     % Selection rule
%     ind = lasso_screening_rule(t(k)/t(k-1),sol_fista_p(:,k-1),A);
% 
%     % Compute solution
%     [sol_fista_x(ind,k),sol_fista_p(:,k),num_iters] = ...
%         lasso_fista_solver(sol_fista_x(ind,k-1),sol_fista_p(:,k-1),t(k),...
%         A(:,ind),b,tau,max_iters,tol_fista,min_iters);
%     sol_fista_p(:,k) = sol_fista_p(:,k)/t(k);
% end
% time_fista_alg = toc;
% time_fista_total = time_fista_alg + time_fista_L22;
% disp('Done.')
% disp(' ')



%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions
%%%%%%%%%%%%%%%%%%%%
disp('Running differential inclusions algorithm...')
tic
% k == 1
[sol_incl_x(:,1), sol_incl_p(:,1)] = ...
    BPDN_inclusions_solver(A,b,p0,t(1),tol);

% k >= 2
for k=2:1:kmax
    % Selection rule
    ind = lasso_screening_rule(t(k)/t(k-1),sol_incl_p(:,k-1),A);

    [sol_incl_x(ind,k), sol_incl_p(:,k)] = ...
        BPDN_inclusions_solver(A(:,ind),b,sol_incl_p(:,k-1),t(k),tol);
end
disp('Done.')
time_incl_alg = toc;

% Basis Pursuit solution
% tic
% disp('Running BP algorithm with warm start')
% 
% [sol_bp_x, sol_bp_p] = ...
%     BPDN_inclusions_solver(A,b,sol_incl_p(:,kmax),0,tol);
% time_bp_alg = toc;
% disp('Done.')