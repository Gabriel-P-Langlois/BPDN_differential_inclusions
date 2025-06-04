%% runall script
%   This script runs the BP inclusion solver, the BPDN solver, and GLMNET
%   on the l1_testset data taken from 
%   https://wwwopt.mathematik.tu-darmstadt.de/spear/
%   which was used in the following paper:

% "Solving basis pursuit: Heuristic optimality check and solver comparison"
%   by Lorenz, Dirk A and Pfetsch, Marc E and Tillmann, Andreas M.


%% Initialization
% Options
use_fista = false;
tol = 1e-08;
tol_glmnet = 1e-08;
tol_fista = tol;

% Load the data
% Use `low dynamic range" data, which I think makes it harder to
% identify the true solution?

% m = 512, n = 4096: spear_inst_73.mat and spear_inst_74.mat
% m = 1024, n = 8192: spear_inst_147.mat and spear_inst_148.mat
% m = 2048, n = 12288: spear_inst_173.mat and spear_inst_174.mat
%   BP incl solver takes ~0.33/4.3s for these.
% m = 8192, n = 49152: spear_inst_199.mat and spear_inst_200.mat
%   BP incl solver takes ~2s/60s for these. 

load './../../../LOCAL_DATA/l1_testset_data/spear_inst_74.mat'
[m,n] = size(A);

% BPDN and GLMNET: Grid of hyperparameters
spacing = -0.005;
max = 0.995;
min = 0.00;

% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution.
t0 = norm(A.'*b,inf);
x0 = zeros(n,1);
p0 = -b/t0;

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
    BP_inclusions_solver(A,b,p0,tol);
time_incl_BP_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_BP_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bp_count), '.'])
disp(['||x_true-sol_incl_BP_x||_{2} = ', num2str(norm(x-sol_incl_BP_x))])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: BPDN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Running the differential inclusions algorithm for the BPDN problem...')
tic
[sol_incl_x,sol_incl_p, bpdn_count] = ...
    BPDN_inclusions_regpath_solver(A,b,p0,t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bpdn_count), '.'])
disp(['||x_true-sol_incl_x(:,end)||_{2} = ', num2str(norm(x-sol_incl_x(:,end)))])
disp(' ')


%% Greedy homotopy algorithm via the matroid property
% Note: This computes sol_g_xf and sol_g_eqset such that
%    A*sol_g_xf(sol_g_eqset) = b
disp(' ')
disp('Running the greedy algorithm \w thresholding...')
tic
[sol_g_x, sol_g_p, sol_g_b, sol_g_xf, sol_g_eqset] = ...
    greedy_homotopy_threshold(A,b,tol);
time_greedy_alg = toc;
disp(['Done. Total time = ', num2str(time_greedy_alg), ' seconds.'])
disp(' ')

disp(' ')
disp(['Running the BP solver using the dual greedy solution' ...
    ' as a warm start...'])
tic
[~,~, warm_count] = ...
    BP_inclusions_solver(A,b,sol_g_p(:,end),tol);
time_warm = toc;
disp(['Done. Total time = ', num2str(time_warm), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(warm_count)])
disp(' ')


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
% sol_glmnet_x = lasso(sqrt(m)*A,sqrt(m)*b, 'lambda', t, ...
%     'Intercept', false, 'RelTol', tol_glmnet);
% sol_glmnet_x = flip(sol_glmnet_x,2);
% for k=1:1:kmax
%     sol_glmnet_p(:,k) = (A*sol_glmnet_x(:,k)-b)/t(k);
% end
% time_glmnet_alg = toc;
% disp(['Done. Total time = ', num2str(time_glmnet_alg), ' seconds.'])
% disp(' ')





% %%%%%%%%%%%%%%%%%%%%%%%%
% FISTA + selection rule
% %%%%%%%%%%%%%%%%%%%%%%%%
if(use_fista)
    sol_fista_x = zeros(n,kmax);
    sol_fista_p = zeros(m,kmax);
    max_iters = 50000;
    min_iters = 100;

%   Compute the L22 norm of the matrix A and tau parameter for FISTA
    disp('Running FISTA algorithm...')
    tic
    L22 = svds(A,1)^2;
    time_fista_L22 = toc;

%   Computer some parameters and options for FISTA
    tau = 1/L22;

%   Run the FISTA solver and rescale the dual solution
    tic

%   k == 1
    [sol_fista_x(:,1),sol_fista_p(:,1),num_iters] = ...
            lasso_fista_solver(x0,p0,t(1),...
            A,b,tau,max_iters,tol_fista,min_iters);
    sol_fista_p(:,1) = sol_fista_p(:,1)/t(1);

%   k >= 2
    for k=2:1:kmax
%       Selection rule
        ind = lasso_screening_rule(t(k)/t(k-1),sol_fista_p(:,k-1),A);

%       Compute solution
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
