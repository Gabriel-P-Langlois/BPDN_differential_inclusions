%% runall script
%   This script runs a simple numerical experiment to showcase all the 
%   algorithms.
%
%   Part 2/2.
%
%   This runs Algorithm 1-2, glmnet, mlass, fista.
%   This is the same as in Part 1/2, except the tolerances for glmnet,
%   fista and mlasso are harsher (1e-13, 1e-08, 1e-08).
%
%   This example uses instance 548 of the SPEAR dataset from
%   Lorenz, Dirk A., Marc E. Pfetsch, and Andreas M. Tillmann. 
%   "Solving basis pursuit: Heuristic optimality check and 
%   solver comparison." ACM Transactions on Mathematical Software (TOMS) 
%   41, no. 2 (2015): 1-29.
%
%   FISTA: ~2000s
%
%   Run from the project directory with
%   run ./results/1_calibration/run_spear548_2_2.m


%% Initialization
name = "dense_spear548_2_2";

% BPDN
tol = 1e-08;

% GLMNET
% Use default parameters
options = glmnetSet;
options.thresh = 1e-13; options.maxit = 10^8;

% Others
tol_mlasso = 1e-8;
tol_fista = 1e-8;


%% Generate data
load './../../../LOCAL_DATA/l1_testset_data/spear_inst_548.mat'
[m,n] = size(A);

% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution
t0 = norm(A.'*b,inf);
x0 = zeros(n,1);
p0 = -b/t0;


%% Solve BPDN using various algorithms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BPDN_incl_homotopy algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_nb_pts = 10*m;
disp('0. Running the BPND homotopy solver...')
tic
[sol_hBPDN_x, sol_hBPDN_p, t, hBPDN_count] = ...
    BPDN_incl_homotopy(A,b,max_nb_pts,tol);
time_hBPDN_alg = toc;
disp(['Done. Total time = ', num2str(time_hBPDN_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(hBPDN_count), '.'])
disp(' ')

kmax = length(t);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: BPDN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('1. Running the differential inclusions BPDN algorithm...')
tic
[sol_incl_x,sol_incl_p, bpdn_count, bpdn_linsolve] = ...
    BPDN_incl_regpath(A,b,p0,t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bpdn_count), '.'])
disp(['Total number of linsolve calls: ', num2str(bpdn_linsolve)])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GLMNET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning('off');
sol_glmnet_p = zeros(m,kmax); 
options.lambda = t;
options.alpha = 1;
options.intr = false;
options.standardize = false;


% Run the external GLMNET package
disp('4. Running GLMNET for the BPDN problem \w Reltol = tol')
tic
fit = glmnet(sqrt(m)*A,sqrt(m)*b, 'gaussian', options);

% Trim x solution and compute the dual solution
sol_glmnet_x = fit.beta;
for k=1:1:kmax
    sol_glmnet_p(:,k) = (A*sol_glmnet_x(:,k)-b)/(t(k));
end
time_glmnet_alg_1 = toc;
disp(['Done. Total time = ', num2str(time_glmnet_alg_1), ' seconds.'])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB's integrated LASSO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sol_mlasso_p = zeros(m,kmax); 
warning('off');

% Run MATLAB's native lasso solver, flip it, and rescale the dual solution
disp("5. Running MATLAB's integrated LASSO algorithm for the BPDN problem...")
tic
sol_mlasso_x = lasso(sqrt(m)*A,sqrt(m)*b, 'lambda', t, ...
    'Intercept', false, 'RelTol', tol_mlasso);
sol_mlasso_x = flip(sol_mlasso_x,2);
for k=1:1:kmax
    sol_mlasso_p(:,k) = (A*sol_mlasso_x(:,k)-b)/t(k);
end
time_mlasso_alg = toc;
disp(['Done. Total time = ', num2str(time_mlasso_alg), ' seconds.'])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%
% FISTA + selection rule
%%%%%%%%%%%%%%%%%%%%%%%%%

sol_fista_x = zeros(n,kmax); sol_fista_p = zeros(m,kmax);
min_iters = 100; max_iters = 50000;

% Compute the L22 norm of the matrix A and tau parameter for FISTA
disp('Final. Running FISTA algorithm...')
tic
L22 = svds(A,1)^2;
time_fista_L22 = toc;
vecA2 = vecnorm(A,2);

% Computer some parameters and options for FISTA
tau = 1/L22;

% Run the FISTA solver and rescale the dual solution
tic

% k == 1
[sol_fista_x(:,1),sol_fista_p(:,1),num_iters] = ...
        lasso_fista_solver(x0,p0,t(1),...
        A,b,tau,max_iters,tol_fista,min_iters);
sol_fista_p(:,1) = sol_fista_p(:,1)/t(1);

% 2 <= k <= kmax-1
for k=2:1:kmax-1
    % Selection rule
    Atimesp = A.'*sol_fista_p(:,k-1);
    ind = lasso_screening_rule(t(k)/t(k-1),sol_fista_p(:,k-1),vecA2,Atimesp);

    % Compute solution
    [sol_fista_x(ind,k),sol_fista_p(:,k),num_iters] = ...
        lasso_fista_solver(sol_fista_x(ind,k-1),sol_fista_p(:,k-1),t(k),...
        A(:,ind),b,tau,max_iters,tol_fista,min_iters);
    sol_fista_p(:,k) = sol_fista_p(:,k)/t(k);
end
time_fista_alg = toc;
time_fista_total = time_fista_alg + time_fista_L22;
disp(['Done. Total time = ', num2str(time_fista_total), ' seconds.'])
disp(' ')



% Set summarize flag to true
summarize_1_dense = true;
summarize_1_calculations = false;