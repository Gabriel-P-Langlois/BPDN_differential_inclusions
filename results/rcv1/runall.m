%% runall script
%   This script runs the BP inclusion sparse solver, the BPDN solver, 
% and GLMNET on RCV1

% Run from the project directory with
% run ./results/rcv1/runall.m

%% Notes:
% With
%
% spacing = -0.01;
% max = 0.99;
% min = 0.01;
%
% It takes about 347s (itertion = )for BPDN and 260s for FISTA

%% Initialization
% Options
use_fista = true;
tol = 1e-08;
tol_glmnet = 1e-08;
tol_fista = tol;

% Load the data
load './../../../LOCAL_DATA/A_rcv1.mat'
load './../../../LOCAL_DATA/b_rcv1.mat'
[m,n] = size(A_rcv1);

% BPDN and GLMNET: Grid of hyperparameters
spacing = -0.01;
max = 0.99;
min = 0.01;

% Note: Takes about 3 minutes with the BPDN solver.

% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution.
t0 = norm(A_rcv1.'*b_rcv1,inf);
x0 = zeros(n,1);
p0 = -b_rcv1/t0;

% Generate desired grid of hyperparameters
t = t0 * (max:spacing:min);
%t = [t,t0*(0.001)];
kmax = length(t);


%% Solve BPDN using BP/BPDN solvers, GLMNET and, if enabled, FISTA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Direct Basis Pursuit Solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the differential inclusions Basis Pursuit solver...')
tic
[sol_incl_BP_x, sol_incl_BP_p, bp_count] = ...
    BP_inclusions_solver_s(A_rcv1,b_rcv1,p0,tol);
time_incl_BP_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_BP_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bp_count), '.'])
disp(' ')


% disp(' ')
% disp('1. Running the greedy algorithm \w thresholding.')
% tic
% [~, sol_g_p, ~, sol_g_xf, sol_g_eqset] = ...
%     greedy_homotopy_threshold_s(A_rcv1,b_rcv1,tol);
% time_greedy_alg = toc;
% disp(['Done. Total time = ', num2str(time_greedy_alg), ' seconds.'])
% disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: BPDN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Running the differential inclusions algorithm for the BPDN problem...')
tic
[sol_incl_x,sol_incl_p, bpdn_count] = ...
    BPDN_inclusions_regpath_solver_s(A_rcv1,b_rcv1,p0,t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bpdn_count), '.'])
disp(' ')

disp(norm(b_rcv1))
disp(norm(A_rcv1*sol_incl_x(:,end)-b_rcv1))


% %%%%%%%%%%%%%%%%%%%%
% % GLMNET
% %%%%%%%%%%%%%%%%%%%%
% %   Note 1: The matrix A and response vector b must be rescaled
% %   by a factor of sqrt(m) due to how its implemented.
% 
% %   Note 2: The lasso function returns the solutions of the
% %   regularization path starting from its lowest value.
% 
% %   Note 3: Unlike the exact lasso algorithm, the dual solution is Ax-b.
% 
% sol_glmnet_p = zeros(m,kmax); 
% warning('off');
% 
% % Run MATLAB's native lasso solver, flip it, and rescale the dual solution
% disp('Running the GLMNET algorithm for the BPDN problem...')
% tic
% sol_glmnet_x = lasso(sqrt(m)*A_rcv1,sqrt(m)*b_rcv1, 'lambda', t, ...
%     'Intercept', false, 'RelTol', tol_glmnet);
% sol_glmnet_x = flip(sol_glmnet_x,2);
% for k=1:1:kmax
%     sol_glmnet_p(:,k) = (A_rcv1*sol_glmnet_x(:,k)-b_rcv1)/t(k);
% end
% time_glmnet_alg = toc;
% disp(['Done. Total time = ', num2str(time_glmnet_alg), ' seconds.'])
% disp(' ')


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
%     L22 = svds(A_rcv1,1)^2;
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
%             A_rcv1,b_rcv1,tau,max_iters,tol_fista,min_iters);
%     sol_fista_p(:,1) = sol_fista_p(:,1)/t(1);
% 
%     % k >= 2
%     for k=2:1:kmax
%         disp(k)
%         % Selection rule
%         ind = lasso_screening_rule(t(k)/t(k-1),sol_fista_p(:,k-1),A_rcv1);
% 
%         % Compute solution
%         [sol_fista_x(ind,k),sol_fista_p(:,k),num_iters] = ...
%             lasso_fista_solver(sol_fista_x(ind,k-1),sol_fista_p(:,k-1),t(k),...
%             A_rcv1(:,ind),b_rcv1,tau,max_iters,tol_fista,min_iters);
%         sol_fista_p(:,k) = sol_fista_p(:,k)/t(k);
%     end
%     time_fista_alg = toc;
%     time_fista_total = time_fista_alg + time_fista_L22;
%     disp(['Done. Total time = ', num2str(time_fista_total), ' seconds.'])
%     disp(' ')
% end
% 
% 
% % Set summarize flag to true
% summarize_rcv1 = true;