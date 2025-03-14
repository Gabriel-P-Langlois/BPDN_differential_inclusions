%% runall script
%   This script solves the BPDN problem via homotopy and then have the
%   glmnet solver solve it along the same path.
%
%   If enabled, the fista solver is also used.
%
%   Specifications: Gaussian data \w homotopy obtained from the BPDN
%                   problem GLMNET and FISTA use a 
%                   regularization path strategy.


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


%% Generate data
% Set random seed and generate Gaussian data
rng('default')
[A,b] = generate_gaussian_data(m,n,SNR,val_nonzero,prop);

% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution
t0 = norm(A.'*b,inf);
x0 = zeros(n,1);
p0 = -b/t0;


%% Solve BPDN using the homotopy solver, GLMNET and, if enabled, FISTA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Homotopy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Running the BPDN homotopy solver...')
tic
[sol_homotopy_x, sol_homotopy_p, t] = BPDN_homotopy_solver(A,t0,p0,tol);
disp('Done.')
time_incl_homotopy_alg = toc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Direct BPDN solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Running the differential inclusions algorithm for the BPDN problem...')
tic
[sol_incl_x,sol_incl_p] = BPDN_inclusions_regpath_solver(A,b,p0,t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])


%% Summarize
summarize_1000_2500_homotopy = true;