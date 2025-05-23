%% runall script
%   This script runs the BP inclusion solver, the BPDN solver, and GLMNET
%   on Sparco problem 12.

% Run from the project directory with
% run ./results/sparco_prob12/runall.m


%% Initialization
% Options
use_fista = false;
tol = 1e-08;
tol_glmnet = 1e-08;
tol_fista = tol;

% Load the data
load './../../../LOCAL_DATA/A_sparco_12.mat'
load './../../../LOCAL_DATA/b_sparco_12.mat'
[m,n] = size(A_sparco_12);

% BPDN and GLMNET: Grid of hyperparameters
spacing = -0.005;
max = 0.995;
min = 0.0;

% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution.
t0 = norm(A_sparco_12.'*b_sparco_12,inf);
x0 = zeros(n,1);
p0 = -b_sparco_12/t0;

% Generate desired grid of hyperparameters
t = t0 * (max:spacing:min);
kmax = length(t);


%% Solve BPDN using BP/BPDN solvers, GLMNET and, if enabled, FISTA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Direct Basis Pursuit Solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the differential inclusions Basis Pursuit solver...')
tic
[sol_incl_BP_x, sol_incl_BP_p, bp_count] = ...
    BP_inclusions_solver(A_sparco_12,b_sparco_12,p0,tol);
time_incl_BP_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_BP_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bp_count), '.'])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: BPDN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Running the differential inclusions algorithm for the BPDN problem...')
tic
[sol_incl_x,sol_incl_p, bpdn_count] = ...
    BPDN_inclusions_regpath_solver(A_sparco_12,b_sparco_12,p0,t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bpdn_count), '.'])
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
sol_glmnet_x = lasso(sqrt(m)*A_sparco_12,sqrt(m)*b_sparco_12, 'lambda', t, ...
    'Intercept', false, 'RelTol', tol_glmnet);
sol_glmnet_x = flip(sol_glmnet_x,2);
for k=1:1:kmax
    sol_glmnet_p(:,k) = (A_sparco_12*sol_glmnet_x(:,k)-b_sparco_12)/t(k);
end
time_glmnet_alg = toc;
disp(['Done. Total time = ', num2str(time_glmnet_alg), ' seconds.'])
disp(' ')



% %%%%%%%%%%%%%%%%%%%%%%%%%
% % FISTA + selection rule
% %%%%%%%%%%%%%%%%%%%%%%%%%
% if(use_fista)
%     sol_fista_x = zeros(n,kmax);
%     sol_fista_p = zeros(m,kmax);
%     max_iters = 50000;
%     min_iters = 100;
% 
%     % Compute the L22 norm of the matrix A and tau parameter for FISTA
%     disp('Running FISTA algorithm...')
%     tic
%     L22 = svds(A,1)^2;
%     time_fista_L22 = toc;
% 
%     % Computer some parameters and options for FISTA
%     tau = 1/L22;
% 
%     % Run the FISTA solver and rescale the dual solution
%     tic
% 
%     % k == 1
%     [sol_fista_x(:,1),sol_fista_p(:,1),num_iters] = ...
%             lasso_fista_solver(x0,p0,t(1),...
%             A,b,tau,max_iters,tol_fista,min_iters);
%     sol_fista_p(:,1) = sol_fista_p(:,1)/t(1);
% 
%     % k >= 2
%     for k=2:1:kmax
%         % Selection rule
%         ind = lasso_screening_rule(t(k)/t(k-1),sol_fista_p(:,k-1),A);
% 
%         % Compute solution
%         [sol_fista_x(ind,k),sol_fista_p(:,k),num_iters] = ...
%             lasso_fista_solver(sol_fista_x(ind,k-1),sol_fista_p(:,k-1),t(k),...
%             A(:,ind),b,tau,max_iters,tol_fista,min_iters);
%         sol_fista_p(:,k) = sol_fista_p(:,k)/t(k);
%     end
%     time_fista_alg = toc;
%     time_fista_total = time_fista_alg + time_fista_L22;
%     disp(['Done. Total time = ', num2str(time_fista_total), ' seconds.'])
%     disp(' ')
% end
% 
% 
% Set summarize flag to true
summarize_sparco = true;