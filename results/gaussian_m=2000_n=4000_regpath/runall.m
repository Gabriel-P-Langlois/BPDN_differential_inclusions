%% runall script
%   This script compares the BPDN_inclusions_solver and glmnet solver at 
%   different hyperparameter values on Gaussian data. 
%
%   If enabled, the fista solver is also used.
%
%   Specifications: Gaussian data \w fixed grid.
%                   GLMNET and FISTA use a regularization path strategy.
%                   BPDN_inclusions uses a highly optimized regularization
%                   path strategy.

% Run from the project directory with
% run ./results/gaussian_m=2000_n=4000_regpath/runall.m

%% Initialization
% Nb of samples and features
m = 2000;
n = 4000;
use_fista = false;

% Signal-to-noise ratio, value of nonzero coefficients, and
% proportion of nonzero coefficients in the signal.
SNR = 1;
val_nonzero = 1;
prop = 0.05;

% Tolerance levels
tol = 1e-08;
tol_glmnet = 1e-08;
tol_fista = tol;


% Grid of hyperparameters
spacing = -0.005;
max = 0.995;
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

% Generate desired grid of hyperparameters
t = t0 * (max:spacing:min);
kmax = length(t);


%% Solve BPDN using BPDN, GLMNET and, if enabled, FISTA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: BPDN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Running the differential inclusions algorithm for the BPDN problem...')
tic
[sol_incl_x,sol_incl_p, count] = ...
    BPDN_incl_regpath(A,b,p0,t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(count), '.'])
disp(' ')

% Set summarize flag to true
summarize_2000_20000_regpath = true;