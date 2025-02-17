%% runall script for testing diferent nnls solvers
% this script tests Meyers's method of hinges and compare 
% its efficiency vs the MATLAB implementation


%% NOTES
%
%   1)  Using (m < n, active_set = true(n,1)) makes the hinge algorithm
%       behave badly. This is expected; this result in solving an over
%       determined system, so the linear solver will behave poorly.
%
%       In practice, we will always have n_eff < m, so we're fine.
%
%   2)  Using active_set = false(n,1) in the hinge algorithm yields the
%       Lawson-Hanson algorithm.



%% Input and options
% Nb of samples and features
m = 200;
n = 1000;

% Initial active set of the hinge algorithm
active_set = false(n,1);
tol = 1e-08;


%% Generate data
rng('default');
A = randn(m,n);
b = randn(m,1); 


%% Solver the NNLS problem min_{x >= 0} ||A*x-b||_{2}^{2} 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) MATLAB's nnls solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
x_lsqnonneg = lsqnonneg(A,b);
p_lsqnonneg = A*x_lsqnonneg - b;
time_nnls1 = toc;

disp(["Total time for MATLAB's lsqnonneg algorithm: ", ...
    num2str(time_nnls1)])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) Meyers's nnls solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[x_hinge,p_hinge] = hinge_lsqnonneg(A,b,active_set,tol);
time_nnls2 = toc;

disp(["Total time for Meyers's algorithm: ", ...
    num2str(time_nnls2)])


%% Comparison between the two solvers
disp(["norm(p_lsqnonneg-p_hinge,inf) = ", ...
    num2str(norm(p_lsqnonneg-p_hinge,inf))])

disp(["norm(x_lsqnonneg-x_hinge,inf) = ", ...
    num2str(norm(x_lsqnonneg-x_hinge))])