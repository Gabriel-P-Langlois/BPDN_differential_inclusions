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
tol_glmnet = tol;    % Default is 1e-04
tol_fista = tol;

load './../../../LOCAL_DATA/l1_testset_data/spear_inst_548.mat'
[m,n] = size(A);

% BPDN and GLMNET: Grid of hyperparameters
spacing = -0.01;
max_val = 0.99;
min_val = 0.00;

% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution.
t0 = norm(A.'*b,inf);
x0 = zeros(n,1);
p0 = -b/t0;

% Generate desired grid of hyperparameters
t = t0 * (max_val:spacing:min_val);
kmax = length(t);


%% Solvers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BP differential inclusions solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('1. Running the BP inclusions solver...')
tic
[sol_incl_BP_x, sol_incl_BP_p, bp_nnls_count,bp_linsolve_count] = ...
    BP_incl_direct(A,b,p0,tol);
time_incl_BP_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_BP_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bp_nnls_count), '.'])
disp(['Total number of linsolve calls: ', num2str(bp_linsolve_count)])
disp(['||x_true-sol_incl_BP_x||_{2} = ', num2str(norm(x-sol_incl_BP_x))])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BPDN differential inclusions solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('2. Running the BPDN inclusions solver...')
tic
[sol_incl_x,sol_incl_p, bpdn_nnls_count,bpdn_linsolve_count] = ...
    BPDN_incl_regpath(A,b,p0,t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bpdn_nnls_count), '.'])
disp(['Total number of linsolve calls: ', num2str(bpdn_linsolve_count)])
disp(['||x_true-sol_incl_x(:,end)||_{2} = ', num2str(norm(x-sol_incl_x(:,end)))])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GLMNET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sol_glmnet_p = zeros(m,kmax); 

warning('off');
options = glmnetSet;
options.alpha = 1;
options.lambda = t;
options.intr = false;
options.maxit = 10^7;


% Run the external GLMNET package with thresh = 1e-4
options.thresh = tol_glmnet;
disp('3. Running GLMNET for the BPDN problem \w Reltol = tol')
tic
fit = glmnet(sqrt(m)*A,sqrt(m)*b, 'gaussian', options);
sol_glmnet_x = fit.beta;
for k=1:1:kmax
    sol_glmnet_p(:,k) = (A*sol_glmnet_x(:,k)-b)/t(k);
end
time_glmnet_alg_1 = toc;
disp(['Done. Total time = ', num2str(time_glmnet_alg_1), ' seconds.'])
disp(' ')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Greedy homotopy algorithm via the matroid property
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: This computes sol_g_xf and sol_g_eqset such that
%    A*sol_g_xf(sol_g_eqset) = b
disp(' ')
disp('4. Running the greedy algorithm \w thresholding...')
tic
[sol_g_x, sol_g_p, sol_g_b, sol_g_xf, sol_g_eqset] = ...
    greedy_homotopy_threshold(A,b,tol);
time_greedy_alg = toc;
disp(['Running the BP solver using the dual greedy solution' ...
    ' as a warm start...'])
tic
[~,~, warm_nnls_count,warm_linsolve_count] = ...
    BP_incl_direct(A,b,sol_g_p(:,end),tol);
time_warm = time_greedy_alg + toc;
disp(['Done. Total time = ', num2str(time_warm), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(warm_nnls_count)])
disp(['Total number of linsolve calls: ', num2str(warm_linsolve_count)])
disp(' ')


% %%%%%%%%%%%%%%%%%%%%%%%%
% FISTA + selection rule
% %%%%%%%%%%%%%%%%%%%%%%%%
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

    % Compute some parameters and options for FISTA
    tau = 1/L22;

    % Run the FISTA solver and rescale the dual solution
    tic

    % k == 1
    [sol_fista_x(:,1),sol_fista_p(:,1),num_iters] = ...
            lasso_fista_solver(x0,p0,t(1),...
            A,b,tau,max_iters,tol_fista,min_iters);
    sol_fista_p(:,1) = sol_fista_p(:,1)/t(1);

    % k >= 2
    for k=2:1:kmax-1
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
