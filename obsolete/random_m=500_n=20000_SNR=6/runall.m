%% Script: Random data, m=100 and n=1000
% This script runs the homotopy method on randomized data with
% m = 500 and n = 20000 and high signal to noise ratio.


%% Random setting and parameters
%rng('default')
tol_exact = 1e-10;                      % Tolerance of the homotopy method
m = 500; n = 20000;                      % Dimensions
val = 1;                                % Value of nonzero coefficients
num_nonzero_coeffs = 100;            % Number of coefficients = val
SNR = 6;                                % Signal to noise ratio

display_iterations = true;


%% Example design: Synthetic data with noise
% Taken from ``Sparse Regression: Scalable Algorithms and Empirical
% Performance" by Dimitris Bertsimas, Jean Pauphilet and Bart Van Parys

% Random matrix design
A = randn(m,n);
A = A./sqrt(sum(A.^2)/m);

% Create the true sparse solution
true_sol = zeros(n,1);
true_sol(randsample(n,num_nonzero_coeffs)) = val; 
support_true_sol = find(true_sol);
ind_zero_xsol = find(~true_sol);

% Add some noise to the solution
sigma = norm(A*true_sol)/sqrt(SNR);
b = (A*true_sol + sqrt(sigma)*randn(m,1));


%% Solver the Lasso problem with the homotopy solver
[homotopy_sol_x,sol_homotopy_sol_p,homotopy_path] = ...
    lasso_solver_homotopy(A,b,m,n,tol_exact,display_iterations);


% %% Compare solution found with the true solution
% % Store solution of Basis Pursuit and identify the support
% bp_sol = homotopy_sol_x(:,end);
% support_bp_sol = find(bp_sol);
% 
% % Compare the support of BP with that of the true solution
% C = intersect(support_true_sol,support_bp_sol);
% 
% % Calculate the number of false positives...
% 
% % Calculate the number of false negatives....