%% Description
% This script compares the performance of my exact homotopy method.
% Written by Gabriel Provencher Langlois


%% Options for the script

% Option to output the result at each iteration of the exact
% lasso algorithm implementations.
display_iterations = true;

% Option to output the result at each iteration of the FISTA algorithm.
display_output_fista = false;

% Run diagnostics
run_diagnostics = true;

% Tolerances + minimum number of iterations for the FISTA algorithm
tol_exact = 1e-10;

% Random seed
rng('default')


%% Examples
% Example 1 -- Synthetic data with noise
% Taken from ``Sparse Regression: Scalable Algorithms and Empirical
% Performance" by Dimitris Bertsimas, Jean Pauphilet and Bart Van Parys
%
% Fix n = 10000, use m = 500 or 2500 (this will vary)
% Matrix are columns from N(0,I)
% k_num = Number of nonzero coefficients
% SNR = 1 (medium noise)
% Values are all set to 1

m = 50; n = 500;     % Number of samples and features
k_num = m;              % Number of true nonzero coefficients
val = 1;                % Value of nonzero coefficients
prop = 0.005*n;         % Proportion of coefficients that are equal to val.
SNR = 1;                % Signal to noise ratio

% Design matrix + Normalize it
A = randn(m,n);
A = A./sqrt(sum(A.^2)/m);

xsol = zeros(n,1);
xsol(randsample(n,k_num)) = val; 
ind_nonzero_xsol = find(xsol);
ind_zero_xsol = find(~xsol);

sigma = norm(A*xsol)/sqrt(SNR);
b = (A*xsol + sqrt(sigma)*randn(m,1));


%% Homotopy algorithm
disp(' ')
disp('----------')
disp('The homotopy method (via gradient inclusions) for the Lasso.')

tic
[sol_exact_x,sol_exact_p,exact_path] = ...
    lasso_solver_homotopy(A,b,m,n,tol_exact,display_iterations,run_diagnostics);
time_exact_total = toc;

disp(['Total time elasped for the exact Lasso algorithm: ',...
    num2str(time_exact_total), ' seconds.'])
disp('----------')

% Remove superfluous components of the path at the end
length_path = 1 + length(find(exact_path));
sol_path = exact_path(1:length_path); 
sol_exact_x(:,length_path+1:end) = [];
sol_exact_p(:,length_path+1:end) = [];