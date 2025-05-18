%% runall script


%% CASE 1: Gaussian data
% Tolerance levels
tol = 1e-08;

% Nb of samples and features
m = 100;
n = 200;

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


%% Solve BPDN using the homotopy solver, GLMNET and, if enabled, FISTA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Homotopy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the BPDN homotopy solver...')
tic
[sol_homoBP_x, sol_homoBP_p, count] = BP_homotopy_solver(A,b,tol);
time_homotopy_alg = toc;
disp(['Done. Total time = ', num2str(time_homotopy_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(count), '.'])
disp(' ')


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


% Summarize results so far...
disp(norm(sol_inclBP_x-sol_homoBP_x))
disp(norm(sol_inclBP_p-sol_homoBP_p))
disp('&&&&&&&&&&')
disp('&&&&&&&&&&')
disp('&&&&&&&&&&')



%% CASE 2: Bringmann's example
% Tolerance levels
tol = 1e-08;

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

%% TESTS
% Atop_times_p = (p0.'*A).';
% vec_of_signs = sign(-Atop_times_p);
% eq_set = (abs(Atop_times_p) >= 1-tol);
% K = A(:,eq_set).*vec_of_signs(eq_set).';
% u = K\b;
% test = [2;2;2]/3;

% Note: There is a problem with cycling and the T- time. This is
% problematic...

%% Solve BPDN using the homotopy solver, GLMNET and, if enabled, FISTA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Diff. Inclusions: Homotopy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' ')
disp('Running the BPDN homotopy solver...')
tic
[sol_homoBP2_x, sol_homoBP2_p, count2] = BP_homotopy_solver(A,b,tol);
time_homotopy2_alg = toc;
disp(['Done. Total time = ', num2str(time_homotopy2_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(count2), '.'])
disp(' ')



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

