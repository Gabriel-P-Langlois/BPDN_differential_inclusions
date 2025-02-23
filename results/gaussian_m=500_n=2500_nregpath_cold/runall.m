%% runall script
%   This script compares the BPDN_inclusions_solver and glmnet solver at 
%   different hyperparameter values on Gaussian data. 
%
%   If enabled, the fista solver is also used.
%
%   Specifications: Gaussian data \w fixed grid.
%                   All algorithms use the same initial starts
%                   p0 = -b/t0 and x0= 0.


%% Initialization
% Nb of samples and features
m = 500;
n = 2500;
use_fista = true;

% Signal-to-noise ratio, value of nonzero coefficients, and
% proportion of nonzero coefficients in the signal.
SNR = 1;
val_nonzero = 1;
prop = 0.05;

% Tolerance levels
tol = 1e-10;
tol_glmnet = 1e-10;
tol_fista = tol;

% Grid of hyperparameters
spacing = -0.05;
max = 0.95;
min = 0.05;


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
% Diff. Inclusions: BPDN + BP \w selection rule
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sol_incl_x = zeros(n,kmax);
sol_incl_p = zeros(m,kmax);
disp('Running the differential inclusions algorithm for the BPDN problem...')

tic
for k=1:1:kmax
    [sol_incl_x(:,k), sol_incl_p(:,k)] = ...
        BPDN_inclusions_qr_solver(A,b,p0,t(k),tol);
end
disp('Done.')
time_incl_alg = toc;


%%%%%%%%%%%%%%%%%%%%
% GLMNET
%%%%%%%%%%%%%%%%%%%%
%   Note 1: The matrix A and response vector b must be rescaled
%   by a factor of sqrt(m) due to how its implemented.

%   Note 2: The lasso function returns the solutions of the
%   regularization path starting from its lowest value.

%   Note 3: Unlike the exact lasso algorithm, the dual solution is Ax-b.

sol_glmnet_x = zeros(n,kmax);
sol_glmnet_p = zeros(m,kmax); 
warning('off');
time_glmnet_alg = 0;


% Run MATLAB's native lasso solver, flip it, and rescale the dual solution
disp('Running the GLMNET algorithm for the BPDN problem...')
tic
for k=1:1:kmax
    sol_glmnet_x(:,k) = lasso(sqrt(m)*A,sqrt(m)*b, 'lambda', t(k), ...
        'Intercept', false, 'RelTol', tol_glmnet);
    sol_glmnet_x(:,k) = flip(sol_glmnet_x(:,k),2);
    sol_glmnet_p(:,k) = (A*sol_glmnet_x(:,k)-b)/t(k);
end
time_glmnet_alg = time_glmnet_alg + toc;
disp('Done.')
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
    for k=1:1:kmax

        % Compute solution
        [sol_fista_x(:,k),sol_fista_p(:,k),num_iters] = ...
            lasso_fista_solver(sol_fista_x(:,1),p0,t(k),...
            A,b,tau,max_iters,tol_fista,min_iters);
        sol_fista_p(:,k) = sol_fista_p(:,k)/t(k);
    end
    time_fista_alg = toc;
    time_fista_total = time_fista_alg + time_fista_L22;
    disp('Done.')
    disp(' ')
end


% Set summarize flag to true
summarize_500_2500_nregpath_cold = true;