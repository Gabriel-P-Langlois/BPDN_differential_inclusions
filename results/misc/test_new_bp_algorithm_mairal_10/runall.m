%% RUNALL Script
% From the project directory, run
% run ./results/test_new_bp_algorithm_mairal_5/runall.m

% This script solves the Basis Pursuit Problem
%   min_{u \in \Rn} \normone{u}  subject to Au = b
% using the exact BPDN and modified BPDN algorithms

% We use the data from Mairal et al. (2012) for this. Will the modified 
% BPDN algorithm works in this case?


%% m=n=10
tol_bp = 1e-8;
disp_output_bp = false;

% Display some information
load('../../data/mairal_pathological/A_pathological_10.mat', 'A') % saved as A
load('../../data/mairal_pathological/b_pathological_10.mat', 'b') % saved as b
disp('The pathological example of Mairal and Bin (2012) with m = n = 10.')


%% Algorithm 1: Exact BPDN algorithm
tol_exact = 1e-10;
tic
disp(' ')
disp('----------')
disp('Algorithm: Exact solution to BPDN via gradient inclusions')

% Call the solver
max_iter = 30000*5;
[sol_x_exact,sol_p_exact,sol_path] = ...
    BPDN_exact_algorithm(A,b,tol_exact,max_iter,true,false);
time_total_exact = toc;

% Display some information
disp(['Total time elasped for the homotopy algorithm: ',...
    num2str(time_total_exact), ' seconds.'])
disp(["norm(Ax-b): ",num2str(norm(A*sol_x_exact(:,end) - b))])
disp('----------')
disp(' ')


%% Algorithm 2: Modified BPDN algorithm
disp(' ')
disp('----------')
disp('Modified BP algorithm (BPDN + changes in data)')

% Call the BP solver
tic
[sol_x, sol_bp, sol_t, xtot, btot] = BP_silly_algorithm(A,b,tol_bp,disp_output_bp);
time_bp = toc;

% Display some information
disp(['Total time elasped for the exact BP algorithm: ',...
    num2str(time_bp), ' seconds.'])
disp(['Norm of A(x+xtot)-b: ', num2str(norm(A*(sol_x(:,end) + xtot) - b))])
disp(["Residual between the sols of algorithms 1 and 2: ", ...
    num2str(norm(sol_x(:,end) + xtot - sol_x_exact(:,end)))])


% LSQ solution, since we can do it.
disp(' ')
disp('----------')
disp('Direct solution (since m=n)')
tic
xlq = A \ b;
time_lq = toc;
disp(['Total time elasped using LSQ: ',...
num2str(time_lq), ' seconds.'])
disp(["Residual norm between the LSQ solution + modified alg: ", ...
    num2str(norm(sol_x(:,end) + xtot - xlq))])