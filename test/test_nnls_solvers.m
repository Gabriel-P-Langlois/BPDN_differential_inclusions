%% runall script for testing diferent nnls solvers
% this script tests Meyers's method of hinges and compare 
% its efficiency vs the MATLAB implementation


%% Input and options
% Nb of samples and features
m = 1000;
n = 5000;

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

disp(["Total time for MATLAB's lsqnonneg algorithm: ", num2str(time_nnls1)])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2) Meyers's nnls solver with full active set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[x_hinge,p_hinge] = hinge_lsqnonneg(A,b,tol);
time_nnls2 = toc;

disp(["Total time for Meyers's algorithm: ", num2str(time_nnls2)])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3) Meyers's nnls solver with full active set (MEX file)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
[x_hinge_mex,p_hinge_mex] = hinge_lsqnonneg_mex(A,b,tol);
time_nnls2_mex = toc;

disp(["Total time for Meyers's algorithm (MEX): ", num2str(time_nnls2_mex)])


%% Comparison between the two solvers
% Compare relative magnitude of the dual solutions 
disp(["norm(p_lsqnonneg-p_hinge,inf) = ", ...
    num2str(norm(p_lsqnonneg-p_hinge,inf)/norm(p_lsqnonneg,inf))])

% Check that norm(p_lsqnonneg) and norm(p_hinge) = 0.
disp(["norm(p_lsqnonneg,inf) = ", num2str(norm(p_lsqnonneg,inf))])
disp(["norm(p_hinge,inf) = ", num2str(norm(p_hinge,inf))])

% Verify that A.'*p >= 0
disp(["min(A.'*p_lsqnonneg + tol): ", num2str(min(A.'*p_lsqnonneg + tol))])
disp(["min(A.'*p_hinge + tol): ", num2str(min(A.'*p_hinge + tol))])