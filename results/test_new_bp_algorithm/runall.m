%% RUNALL Script
% From the project directory, run
% run ./results/test_new_bp_algorithm/runall.m

% This script solves the Basis Pursuit Problem
%   min_{u \in \Rn} \normone{u}  subject to Au = b
% using the algorithm Ciril et al. (2023) developed.

% It then compares the output with the tentative approach
% that GPL developed, which solves
%   min_{x \in \Rn} \normone{x} subject to A(x+v) = b

% The question is: Do we have x+v = u? Or at least normone(x+v) = norm(u)


%% Example 1 -- Random Gaussian design -- Setup
% Options
rng('default')
tol_bp = 1e-10;
disp_output_bp = false;

% Parameters
m = 100; n = 300;
k_num = floor(n/50);    % Number of true nonzero coefficients
val = 1;                % Value of nonzero coefficients
SNR = 1;                % Signal to noise ratio

% Initialize the design matrix and normalize it
A = randn(m,n);
A = A./sqrt(sum(A.^2)/m);
xsol = zeros(n,1); xsol(randsample(n,k_num)) = val; 

% Generate the noisy observation
sigma = norm(A*xsol)/sqrt(SNR);
b = (A*xsol + sqrt(sigma)*randn(m,1));


%% Example 1 -- Random Gaussian design
%% Algorithm 1 -- Exact BP from Ciril et al. (2023)
disp(' ')
disp('----------')
disp('Basis Pursuit algorithm (Ciril et al. (2023)')

% Call the BP solver
tic
[sol_x, ~] = BP_exact_algorithm(A,b,tol_bp,disp_output_bp);
time_bp = toc;

% Display some information
disp(['Total time elasped for the exact BP algorithm: ',...
    num2str(time_bp), ' seconds.'])
disp('----------')
disp(' ')

% Display optimality conditions
disp(['Euclidan norm of Ax-b: ', num2str(norm(A*sol_x-b))])
disp(['l1 Norm of the bp solution: ', num2str(norm(sol_x,1))])
disp('----------')
disp(' ')


%% Algorithm 2 -- Modified BPDN
disp(' ')
disp('----------')
disp('Testing a new, tentative algorithm...')

% Call the BP solver
tic
[sol_x_2, sol_bp_2, sol_t_2, xtot, btot] = BP_test_algorithm(A,b,tol_bp,disp_output_bp);
time_bp_2 = toc;

% Display some information
disp(['Total time elasped for the exact BP algorithm: ',...
    num2str(time_bp_2), ' seconds.'])
disp('----------')
disp(' ')

% Display optimality conditions
disp(['Euclidan norm of Ax-btot: ', num2str(norm(A*sol_x_2(:,end) - btot))])
tmp = sol_x_2(:,end) + xtot;
disp(['Euclidan norm of A(x+xtot)-b: ', num2str(norm(A*tmp - b))])
disp(['l1 Norm of x + xtot: ', num2str(norm(tmp,1))])
disp(['residual of sol_x - (sol_x_2 + xtot): ', num2str(norm(tmp - sol_x))])
% disp('----------')
% disp(' ')
