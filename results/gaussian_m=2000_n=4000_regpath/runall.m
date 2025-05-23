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
    BPDN_inclusions_regpath_solver(A,b,p0,t,tol);
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



%%%%%%%%%%%%%%%%%%%%%%%%%
% FISTA + selection rule
%%%%%%%%%%%%%%%%%%%%%%%%%
if(use_fista)
    sol_fista_x = zeros(n,kmax);
    sol_fista_p = zeros(m,kmax);
    max_iters = 50000;
    min_iters = 100;

    % Compute the L22 norm of the matrix A and tau parameter for FISTA
    disp('Running FISTA algorithm...')
    tic
    L22 = svds(A,1)^2;
    time_fista_L22 = toc;
    
    % Computer some parameters and options for FISTA
    tau = 1/L22;
    
    % Run the FISTA solver and rescale the dual solution
    tic
    
    % k == 1
    [sol_fista_x(:,1),sol_fista_p(:,1),num_iters] = ...
            lasso_fista_solver(x0,p0,t(1),...
            A,b,tau,max_iters,tol_fista,min_iters);
    sol_fista_p(:,1) = sol_fista_p(:,1)/t(1);
    
    % k >= 2
    for k=2:1:kmax
        % Selection rule
        ind = lasso_screening_rule(t(k)/t(k-1),sol_fista_p(:,k-1),A);
    
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
end


% Set summarize flag to true
summarize_2000_20000_regpath = true;