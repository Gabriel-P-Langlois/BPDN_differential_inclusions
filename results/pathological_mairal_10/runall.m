%% runall script
% Invoke this script from the main directory by calling
%     run ./results/gaussian_greedy/runall.m


%% Initialization
% Nb of samples and features
m = 10;
n = 10;

% Tolerance levels
tol = 1e-08;
tol_glmnet = 1e-08;
tol_fista = tol;


%% Load data
load './../../data/mairal_pathological/A_pathological_10.mat'
load './../../data/mairal_pathological/b_pathological_10.mat'


% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution -b/t0
t0 = norm(A.'*b,inf);
x0 = zeros(n,1);
p0 = -b/t0;


%% Run 1: Direct Basis Pursuit (BP) Solver
disp(' ')
disp('1. Running the differential inclusions Basis Pursuit solver...')
tic
[sol_inclBP_x, sol_inclBP_p, bp_count] = BP_inclusions_solver(A,b,p0,tol);
time_inclBP_alg = toc;
disp(['Done. Total time = ', num2str(time_inclBP_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bp_count)])
disp(' ')


%% Run 2: Homotopy algorithm
disp(' ')
disp('Running the BP homotopy solver...')
tic
[sol_homoBP_x, sol_homoBP_p, count] = BP_homotopy_solver(A,b,tol);
time_homotopy_alg = toc;
disp(['Done. Total time = ', num2str(time_homotopy_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(count), '.'])
disp(' ')


%% Run 3: Warm start via greedy homotopy algorithm via the matroid property 
% Note: This computes sol_g_xf and sol_g_eqset such that
%    A*sol_g_xf(sol_g_eqset) = b
disp(' ')
disp(['3. Running the greedy algorithm \w thresholding and using it ' ...
    'as a warm-start for the BP solver.'])
tic
[~, sol_g_p, ~, sol_g_xf, sol_g_eqset] = ...
    greedy_homotopy_threshold(A,b,tol);
[sol_warm_x,sol_warm_p, warm_count] = ...
    BP_inclusions_solver(A,b,sol_g_p(:,end),tol);
time_warm = toc;
disp(['Done. Total time = ', num2str(time_warm), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(warm_count)])
disp(' ')


%% Run 4: Use the greedy dual solution and a change of variable
disp(' ')
disp(['4. Running the BP solver using the dual greedy solution' ...
    ' with a change of variable as a warm start'])
tic

% Compute the feasible point and the equicorrelation set
b_new = -sol_g_xf(sol_g_eqset);
p_new = -(A(:,sol_g_eqset).')*sol_g_p(:,end);
A_new = -inv(A(:,sol_g_eqset))*A;


% Invoke the bp solver on the modified problem
[sol_test_x, sol_test_p, test_count] = ...
    BP_inclusions_solver(A_new,b_new,p_new,tol);
time_new = toc;

% Invert the solution to get the corresponding vector p.
% Check that it is the same as the dual solution
sol_test_p = -inv(A(:,sol_g_eqset).')*sol_test_p;
disp(['Done. Total time = ', num2str(time_new), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(test_count)])
disp(['Furthermore, norm(sol_test_p-sol_inclBP_p) = ', ...
    num2str(norm(sol_test_p-sol_inclBP_p))])
disp(' ')