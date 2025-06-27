%% runall script
%   This script compares the BPDN_inclusions_solver and glmnet solver at 
%   different hyperparameter values on Gaussian data. 
%
%   Specifications: Gaussian data \w fixed grid.
%                   GLMNET uses a regularization path strategy.
%                   BPDN use a regularization path strategy.
%                   A primal-dual method with log barrier is used too.
%
% Run from the project directory with
% run ./results/gaussian_m=1000_n=5000_BP/runall.m

%% Initialization
% Nb of samples and features
m = 100;
n = 500;
use_fista = false;

% Signal-to-noise ratio, value of nonzero coefficients, and
% proportion of nonzero coefficients in the signal.
SNR = 1;
val_nonzero = 1;
prop = 0.05;

% Tolerance levels
tol = 1e-08;
tol_glmnet = 1e-08;

% Spacing for the BPDN solver and GLMNET
spacing = -0.01;
max = 0.99;
min = 0.0;


%% Generate data
% Set random seed and generate Gaussian data
rng('default')
[A,b] = generate_gaussian_data(m,n,SNR,val_nonzero,prop);

% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution
t0 = norm(A.'*b,inf);
x0 = zeros(n,1);
p0 = -b/t0;


%% Solve using the inclusion solvers,  GLMNET, and an interior method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Direct Basis Pursuit Solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the differential inclusions Basis Pursuit solver...')
tic
[sol_incl_BP_x, sol_incl_BP_p, bp_nnls_count, bp_lsq_count] = ...
    BP_inclusions_solver(A,b,p0,tol);
time_incl_BP_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_BP_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bp_nnls_count), '.'])
disp(['Total number of linsolve calls: ', num2str(bp_lsq_count)])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: BPDN with regularization path
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = t0 * (max:spacing:min);
kmax = length(t);

disp(' ')
disp(['Running the differential inclusions BPDN solver ' ...
    'with regularization path...'])
tic
[sol_incl_BPDN_x,sol_incl_BPDN_p, bpdn_nnls_count, bpdn_lsq_count] = ...
    BPDN_inclusions_regpath_solver(A,b,p0,t,tol);
time_incl_bpdn_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_bpdn_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bpdn_nnls_count), '.'])
disp(['Total number of linsolve calls: ', num2str(bpdn_lsq_count)])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run the greedy algorithm and use it as a warm start
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the greedy algorithm \w thresholding...')
tic
[~, sol_g_p, ~, sol_g_xf, sol_g_eqset] = ...
    greedy_homotopy_threshold(A,b,tol);
time_greedy = toc;


% 1) Warm start
tic
disp(['Running the BP solver using the greedy dual solution' ...
    ' as a warm start.'])
[~,~, warm_nnls_count, warm_lsq_count] = ...
    BP_inclusions_solver(A,b,sol_g_p(:,end),tol);
time_warm_total = time_greedy + toc;
disp(['Done. Total time = ', num2str(time_warm_total), ' seconds.'])
disp(['Total number of NNLS solves (BP solver): ', num2str(warm_nnls_count)])
disp(['Total number of linsolve calls: ', num2str(m+warm_lsq_count)])
disp(' ')


% 2) Inversion formula.
tic
disp(['Running the BP solver using the greedy dual solution' ...
    ' + inversion as a warm start.'])
z = A(:,sol_g_eqset).'*sol_g_p(:,end);
[~,~, inv_nnls_count, inv_lsq_count] = ...
    BP_inclusions_solver(inv(A(:,sol_g_eqset))*A,sol_g_xf(sol_g_eqset),z,tol);
time_inv_total = time_greedy + toc;
disp(['Done. Total time = ', num2str(time_inv_total), ' seconds.'])
disp(['Total number of NNLS solves (BP solver): ', num2str(inv_nnls_count)])
disp(['Total number of linsolve calls: ', num2str(m+inv_lsq_count)])
disp(' ')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Primal-dual \w log barrier (WIP)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Running a primal-dual \w log barrier method...')
tic
sol = PDLB_algorithm(A, b, n, 50, 0, tol);
time_pdco = toc;
disp(['Done. Total time = ', num2str(time_pdco), ' seconds.'])


%% Set summarize flag to true
summarize_1000_5000_BP = true;