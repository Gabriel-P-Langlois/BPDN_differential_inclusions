%% runall script
%   This script compares the BPDN_inclusions_solver and glmnet solver at 
%   different hyperparameter values on Gaussian data. 
%
%   If enabled, the fista solver is also used.
%
%   Specifications: Gaussian data \w fixed grid.
%                   GLMNET and FISTA use a regularization path strategy.
%                   BPDN use a semi-regularization path strategy (only
%                   uses warm starts in a crude way).


%% Initialization
% Nb of samples and features
m = 1000;
n = 5000;
use_fista = false;

% Signal-to-noise ratio, value of nonzero coefficients, and
% proportion of nonzero coefficients in the signal.
SNR = 1;
val_nonzero = 1;
prop = 0.05;

% Tolerance levels
tol = 1e-08;
tol_glmnet = 1e-08;


%% Generate data
% Set random seed and generate Gaussian data
rng('default')
[A,b] = generate_gaussian_data(m,n,SNR,val_nonzero,prop);

% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution
t0 = norm(A.'*b,inf);
x0 = zeros(n,1);
p0 = -b/t0;


%% Solve BPDN using BPDN, GLMNET and, if enabled, FISTA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Direct Basis Pursuit Solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Running the differential inclusions algorithm for the BP problem...')
tic
[sol_incl_BP_x, sol_incl_BP_p] = BP_inclusions_solver(A,b,p0,tol);
disp('Done.')
time_incl_BP_alg = toc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: BPDN with regularization path
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

spacing = -0.01;
max = 0.99;
min = 0.0;
t = t0 * (max:spacing:min);
kmax = length(t);

disp('Running the differential inclusions algorithm for the BPDN problem')
tic
[sol_incl_BPDN_x,sol_incl_BPDN_p] = ...
    BPDN_inclusions_regpath_solver(A,b,p0,t,tol);
time_incl_BPDN_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_BPDN_alg), ' seconds.'])


%%%%%%%%%%%%%%%%%%%%
% GLMNET
%%%%%%%%%%%%%%%%%%%%
%   Note 1: The matrix A and response vector b must be rescaled
%   by a factor of sqrt(m) due to how its implemented.

%   Note 2: The lasso function returns the solutions of the
%   regularization path starting from its lowest value.

%   Note 3: Unlike the exact lasso algorithm, the dual solution is Ax-b.

sol_glmnet_p = zeros(m,kmax); 
warning('off');
time_glmnet_alg = 0;

% Run MATLAB's native lasso solver, flip it, and rescale the dual solution
disp('Running the GLMNET algorithm for the BPDN problem...')
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