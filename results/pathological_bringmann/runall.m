%% runall script
% Run from the project directory with
% run ./results/pathological_bringmann/runall.m


%% CASE 2: Bringmann's example
% Tolerance levels
tol = 1e-08;
use_fista = true;
tol_fista = tol;

% Nb of samples and features
m = 2;
n = 4;

A = [1,1,1,0;
    0,0,0,1];
b = [2;1];

% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution
t0 = norm(A.'*b,inf);
p0 = -b/t0;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Direct Basis Pursuit Solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the differential inclusions Basis Pursuit solver...')
tic
[sol_inclBP2_x, sol_inclBP2_p, BP2_count] = ...
    BP_inclusions_solver(A,b,p0,tol);
time_incl_BP2_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_BP2_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(BP2_count), '.'])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Homotopy for BP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the BPDN homotopy solver...')
tic
[sol_hBP2_x, sol_hBP2_p, count2] = BP_homotopy_solver(A,b,tol);
time_homotopy2_alg = toc;
disp(['Done. Total time = ', num2str(time_homotopy2_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(count2), '.'])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Homotopy for BPDN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the BPND homotopy solver...')
tic
[sol_hBPDN2_x, sol_hBPDN2_p, sol_hBPDN2_t, hBPDN_count2] = ...
    BPDN_homotopy_solver(A,b,tol);
time_hBPDN2_alg = toc;
disp(['Done. Total time = ', num2str(time_hBPDN2_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(hBPDN_count2), '.'])
disp(' ')

l = length(sol_hBPDN2_t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: BPDN solver \w homotopy points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the BPDN differential inclusions algorithm...')
tic
[sol_incl2_x, sol_incl2_p, BPDN_count2] = ...
    BPDN_inclusions_regpath_solver(A,b,p0,sol_hBPDN2_t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(BPDN_count2), '.'])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GLMNET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Note 1: The matrix A and response vector b must be rescaled
%   by a factor of sqrt(m) due to how its implemented.

%   Note 2: The lasso function returns the solutions of the
%   regularization path starting from its lowest value.

%   Note 3: Unlike the exact lasso algorithm, the dual solution is Ax-b.


sol_glmnet2_p = zeros(m,l); 
warning('off');

% Run MATLAB's native lasso solver, flip it, and rescale the dual solution
disp('Running the GLMNET algorithm for the BPDN problem...')
tic
sol_glmnet2_x = lasso(sqrt(m)*A,sqrt(m)*b, 'lambda', sol_hBPDN2_t, ...
    'Intercept', false, 'RelTol', tol);
sol_glmnet2_x = flip(sol_glmnet2_x,2);
for k=1:1:l
    sol_glmnet2_p(:,k) = (A*sol_glmnet2_x(:,k)-b)/sol_hBPDN2_t(k);
end
time_glmnet_alg = toc;
disp(['Done. Total time = ', num2str(time_glmnet_alg), ' seconds.'])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%
% FISTA + selection rule
%%%%%%%%%%%%%%%%%%%%%%%%%
if(use_fista)
    sol_fista_x = zeros(n,l);
    sol_fista_p = zeros(m,l);
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
            lasso_fista_solver(zeros(n,1),p0,sol_hBPDN2_t(1),...
            A,b,tau,max_iters,tol_fista,min_iters);
    sol_fista_p(:,1) = sol_fista_p(:,1)/sol_hBPDN2_t(1);
    
    % k >= 2
    for k=2:1:l
        % Selection rule
        ind = lasso_screening_rule(sol_hBPDN2_t(k)/sol_hBPDN2_t(k-1),...
            sol_fista_p(:,k-1),A);
    
        % Compute solution
        [sol_fista_x(ind,k),sol_fista_p(:,k),num_iters] = ...
            lasso_fista_solver(sol_fista_x(ind,k-1),...
            sol_fista_p(:,k-1),sol_hBPDN2_t(k),...
            A(:,ind),b,tau,max_iters,tol_fista,min_iters);
        sol_fista_p(:,k) = sol_fista_p(:,k)/sol_hBPDN2_t(k);
    end
    time_fista_alg = toc;
    time_fista_total = time_fista_alg + time_fista_L22;
    disp(['Done. Total time = ', num2str(time_fista_total), ' seconds.'])
    disp(' ')
end



%% Summarize some results
disp('Differences between basis pursuit solutions (primal/dual)...')
disp(norm(sol_inclBP2_x-sol_hBP2_x))
disp(norm(sol_inclBP2_p-sol_hBP2_p))
disp(norm(sol_inclBP2_x-sol_hBPDN2_x(:,end)))
disp(norm(sol_inclBP2_p-sol_hBPDN2_p(:,end)))

% Calculate difference between homotopy solution and the BPDN solver
disp(' ')
disp('Calculate deviation between homotopy/BPDN solver solutions...')
norm_primal_diff = zeros(l,1);
norm_dual_diff = zeros(l,1);
norm_primal_glmnet_bpdn = zeros(l,1);
norm_dual_glmnet_bpdn = zeros(l,1);

for i=2:1:l
    norm_primal_diff(i) = norm(sol_incl2_x(:,i) - sol_hBPDN2_x(:,i),2)/...
        norm(sol_incl2_x(:,i));
    norm_dual_diff(i) = norm(sol_incl2_p(:,i) - sol_hBPDN2_p(:,i),2)/...
        norm(sol_incl2_p(:,i));
    norm_primal_glmnet_bpdn(i) = norm(sol_incl2_x(:,i) - sol_glmnet2_x(:,i),2)/...
        norm(sol_incl2_x(:,i));
    norm_dual_glmnet_bpdn(i) = norm(sol_incl2_p(:,i) - sol_glmnet2_p(:,i),2)/...
        norm(sol_incl2_p(:,i));
end

disp('Sum of relative norms between homotopy and BPDN solvers')
disp(' ')
disp(sum(norm_primal_diff))
disp(sum(norm_dual_diff))
disp(' ')

disp('Sum of relative norms between glmnet and BPDN solvers (except t=0)')
disp(' ')
disp(sum(norm_primal_glmnet_bpdn(1:end-1)))
disp(sum(norm_dual_glmnet_bpdn(1:end-1)))
disp(' ')

%% Notes:
% The l2 norm of the solutions obtained from FISTA differ from 
% all the rest; their l2 norm is lower.
% However, their l1 norm are ALL the same.