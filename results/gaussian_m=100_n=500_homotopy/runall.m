%% runall script
% Call from the project directory as follows:
% run ./results/gaussian_m=100_n=500_homotopy/runall.m


%% CASE 1: Gaussian data
% Tolerance levels
tol = 1e-08;
tol_glmnet = 1e-04;    % Default is 1e-04

% Nb of samples and features
m = 500;
n = 1000;

% Signal-to-noise ratio, value of nonzero coefficients, and
% proportion of nonzero coefficients in the signal.
SNR = 1;
val_nonzero = 1;
prop = 0.05;

% Set random seed and generate Gaussian data
rng('default')
[A,b] = generate_gaussian_data(m,n,SNR,val_nonzero,prop);

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
[sol_inclBP_x, sol_inclBP_p, BP_count] = ...
    BP_inclusions_solver(A,b,p0,tol);
time_incl_BP_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_BP_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(BP_count), '.'])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Homotopy for BPDN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the BPND homotopy solver...')
tic
[sol_hBPDN_x, sol_hBPDN_p, sol_hBPDN_t, hBPDN_count] = ...
    BPDN_homotopy_solver(A,b,tol);
time_hBPDN_alg = toc;
disp(['Done. Total time = ', num2str(time_hBPDN_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(hBPDN_count), '.'])
disp(' ')

l = length(sol_hBPDN_t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: BPDN solver \w homotopy points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the BPDN differential inclusions algorithm...')
tic
[sol_incl_x, sol_incl_p, BPDN_count] = ...
    BPDN_inclusions_regpath_solver(A,b,p0,sol_hBPDN_t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(BPDN_count), '.'])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GLMNET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Note 1: The matrix A and response vector b must be rescaled
%   by a factor of sqrt(m) due to how its implemented.

%   Note 2: The lasso function returns the solutions of the
%   regularization path starting from its lowest value.

%   Note 3: Unlike the exact lasso algorithm, the dual solution is Ax-b.


sol_glmnet_p = zeros(m,l); 
warning('off');

% Run MATLAB's native lasso solver, flip it, and rescale the dual solution
disp('Running the GLMNET algorithm for the BPDN problem...')
tic
sol_glmnet_x = lasso(sqrt(m)*A,sqrt(m)*b, 'lambda', sol_hBPDN_t, ...
    'Intercept', false, 'RelTol', tol_glmnet);
sol_glmnet_x = flip(sol_glmnet_x,2);
for k=1:1:l
    sol_glmnet_p(:,k) = (A*sol_glmnet_x(:,k)-b)/sol_hBPDN_t(k);
end
time_glmnet_alg = toc;
disp(['Done. Total time = ', num2str(time_glmnet_alg), ' seconds.'])
disp(' ')



%% Summarize some results
disp('Differences between basis pursuit solutions (primal/dual)...')
disp(['norm(xBP-xhBPDN) = ', num2str(norm(sol_inclBP_x-sol_hBPDN_x(:,end)))])
disp(['norm(pBP-phBPDN) = ', num2str(norm(sol_inclBP_p-sol_hBPDN_p(:,end)))])

% Calculate difference between homotopy solution and the BPDN solver
disp(' ')
disp('Calculate deviation between homotopy/BPDN solver solutions...')
norm_primal_diff = zeros(l,1);
norm_dual_diff = zeros(l,1);

norm_primal_glmnet_bpdn = zeros(l,1);
norm_dual_glmnet_bpdn = zeros(l,1);

norm_primal_glmnet_hBPDN = zeros(l,1);
norm_dual_glmnet_hBPDN = zeros(l,1);



for i=2:1:l
    norm_primal_diff(i) = norm(sol_incl_x(:,i) - sol_hBPDN_x(:,i),2)/...
        norm(sol_incl_x(:,i));
    norm_dual_diff(i) = norm(sol_incl_p(:,i) - sol_hBPDN_p(:,i),2)/...
        norm(sol_incl_p(:,i));

    norm_primal_glmnet_bpdn(i) = norm(sol_incl_x(:,i) - sol_glmnet_x(:,i),2)/...
        norm(sol_incl_x(:,i));
    norm_dual_glmnet_bpdn(i) = norm(sol_incl_p(:,i) - sol_glmnet_p(:,i),2)/...
        norm(sol_incl_p(:,i));

    norm_primal_glmnet_hBPDN(i) = norm(sol_hBPDN_x(:,i) - sol_glmnet_x(:,i),2)/...
        norm(sol_hBPDN_x(:,i));
    norm_dual_glmnet_hBPDN(i) = norm(sol_hBPDN_p(:,i) - sol_glmnet_p(:,i),2)/...
        norm(sol_hBPDN_p(:,i));
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

disp('Sum of relative norms between glmnet and hBPDN solvers (except t=0)')
disp(' ')
disp(sum(norm_primal_glmnet_hBPDN(1:end-1)))
disp(sum(norm_dual_glmnet_hBPDN(1:end-1)))
disp(' ')