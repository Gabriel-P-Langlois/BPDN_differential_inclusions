%% runall script
%   This script compares the BPDN_inclusions_solver and glmnet solver at 
%   different hyperparameter values on Gaussian data. 
%
%   If enabled, the fista solver is also used.
%
%   Specifications: Gaussian data \w fixed grid.
%                   GLMNET uses a regularization path strategy.
%                   BPDN_inclusions uses a highly optimized regularization
%                   path strategy.

% Run from the project directory with
% run ./results/gaussian_m=4000_n=40000_regpath/runall.m
%
% Note: This will take a while.

%% Initialization
% Nb of samples and features
m = 4000;
n = 40000;

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
spacing = -0.01;
max = 0.99;
min = 0.01;


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
[sol_incl_x,sol_incl_p,count] = BPDN_inclusions_regpath_solver(A,b,p0,t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(count), '.'])
disp(' ')


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

% Run MATLAB's native lasso solver, flip it, and rescale the dual solution
disp(' ')
disp('Running the GLMNET algorithm for the BPDN problem...')
tic
sol_glmnet_x = lasso(sqrt(m)*A,sqrt(m)*b, 'lambda', t, ...
    'Intercept', false, 'RelTol', tol_glmnet);
sol_glmnet_x = flip(sol_glmnet_x,2);
for k=1:1:kmax
    sol_glmnet_p(:,k) = (A*sol_glmnet_x(:,k)-b)/t(k);
end
time_glmnet_alg = toc;
disp(['Done. Total time = ', num2str(time_glmnet_alg), ' seconds.'])
disp(' ')


% Set summarize flag to true
summarize_4000_40000_regpath = true;