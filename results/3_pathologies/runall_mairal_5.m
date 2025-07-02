%% runall script
% Invoke this script from the main directory by calling
%     run ./results/pathological_mairal_10/runall.m


%% Initialization

% Setters
use_mlasso = true;
use_homotopy = true;
using_spear = false;
use_fista = false;
use_mlinprog = false;

% Tolerance levels
tol = 1e-08;
tol_glmnet = 1e-04;
tol_mlasso = 1e-04;
tol_fista = tol;


%% Load data
load './../../data/mairal_pathological/A_pathological_5.mat'
load './../../data/mairal_pathological/b_pathological_5.mat'
[m,n] = size(A);


% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution -b/t0
t0 = norm(A.'*b,inf);
x0 = zeros(n,1);
p0 = -b/t0;


%% Run 1: Direct Basis Pursuit (BP) Solver
disp(' ')
disp('1. Running the differential inclusions Basis Pursuit solver...')
tic
[sol_inclBP_x, sol_inclBP_p, bp_count] = BP_incl_direct(A,b,p0,tol);
time_inclBP_alg = toc;
disp(['Done. Total time = ', num2str(time_inclBP_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bp_count)])
disp(' ')


%% Run 2: Homotopy algorithm
total_t = 30000;
disp(' ')
disp('Running the BPDN homotopy solver...')
tic
[sol_hBPDN_x, sol_hBPDN_p, t, count_NNLS, count_LSQ] = ...
    BPDN_incl_homotopy(A,b,total_t,tol);
time_homotopy_alg = toc;
disp(['Done. Total time = ', num2str(time_homotopy_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(count_NNLS), '.'])
disp(['Total number of linsolves: ', num2str(count_LSQ), '.'])
disp(' ')

%% Run 3: BPDN algorithm
disp('3. Running the differential inclusions BPDN algorithm...')
tic
[sol_incl_x,sol_incl_p, bpdn_count, bpdn_linsolve] = ...
    BPDN_incl_regpath(A,b,p0,t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bpdn_count), '.'])
disp(['Total number of linsolve calls: ', num2str(bpdn_linsolve)])
disp(' ')


%% Run 4: Warm start via greedy homotopy algorithm via the matroid property 
disp(' ')
disp('4. Running the differential inclusions BP solver \w warm start')
tic
[~,~, warm_nnls_count, warm_linsolve_count] = ...
    BP_incl_greedy(A,b,tol);
time_warm = toc;
disp(['Done. Total time = ', num2str(time_warm), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(warm_nnls_count)])
disp(['Total number of linsolve calls: ', num2str(warm_linsolve_count)])
disp(' ')


%% Run 5. GLMNET
warning('off');
options = glmnetSet;
options.alpha = 1;
options.lambda = t;
options.intr = false;
options.thresh = tol_glmnet;
options.maxit = 10^8;
options.standardize = false;
kmax = length(t);

sol_glmnet_p = zeros(m,kmax); 

% Run the external GLMNET package
options.thresh = tol_glmnet;
disp('5. Running GLMNET for the BPDN problem \w Reltol = tol')
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


%% Run 6: MATLAB's LASSO
if(use_mlasso)
    sol_mlasso_p = zeros(m,kmax); 
    warning('off');
    
    % Run MATLAB's native lasso solver, flip it, and rescale the dual solution
    disp("6. Running MATLAB's integrated LASSO algorithm for the BPDN problem...")
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
end