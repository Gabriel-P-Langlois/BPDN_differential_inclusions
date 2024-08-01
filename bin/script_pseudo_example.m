function script_pseudo_example(m,n,tol_exact,disp_output_exact,...
    run_opt_cond_checks,use_bp,tol_bp,disp_output_bp,...
    use_pseudo,tol_pseudo,disp_output_pseudo,run_diagnostics)
%%  Script for testing the pseudo/greedy algorithm that GPL has
%   Written by Gabriel Provencher Langlois.


%% Example
% Synthetic data with noise
% Taken from ``Sparse Regression: Scalable Algorithms and Empirical
% Performance" by Dimitris Bertsimas, Jean Pauphilet and Bart Van Parys
%
% Matrix are columns from N(0,I)
% k_num = Number of nonzero coefficients
% SNR = 1 (medium noise)
% Values are all set to 1

% Random seed
rng('default')

% Dimensions and parameters
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



%% Homotopy algorithm
tic
disp(' ')
disp('----------')
disp('Algorithm: Exact solution to BPDN via gradient inclusions')

% Call the solver
max_iter = 10*m;
[sol_x_exact,sol_p_exact,sol_path] = ...
    BPDN_exact_algorithm(A,b,tol_exact,max_iter,disp_output_exact,run_opt_cond_checks);
time_total_exact = toc;

% Calculate the length of the path
length_path = length(sol_path);

% Display some information
disp(['Total time elasped for calculating the solution path to BPDN via gradient inclusions: ',...
    num2str(time_total_exact), ' seconds.'])
disp('----------')
disp(' ')
disp(['Length of the path: ',num2str(length(sol_path))])
disp(['Basis Pursuit solution (residual -- Exact solution to BPDN via gradient inclusions): ||A*sol_x_exact(:,end)-b||_2 = ',num2str(norm(A*sol_x_exact(:,end)-b))])
disp(['Basis Pursuit solution (obj function -- Exact solution to BPDN via gradient inclusions): ||sol_x_exact(:,end)||_{1} = ',num2str(norm(sol_x_exact(:,end),1))])



%% Basis Pursuit algorithm
% This solves the problem
%   min_{x \in \Rn} \norm{x}_{1}   s.t. Ax = b
% and outputs both the primal and dual solutions.
if(use_bp)
    disp(' ')
    disp('----------')
    disp('Basis Pursuit algorithm')
    
    % Call the BP solver
    tic
    [sol_x_bp, sol_p_bp] = BP_exact_algorithm(A,b,tol_bp,disp_output_bp);
    time_bp = toc;

    % Display some information
    disp(['Total time elasped for the exact Lasso algorithm: ',...
        num2str(time_bp), ' seconds.'])
    disp('----------')
    disp(' ')
end



%% Pseudo algorithm
% 
tic
disp(' ')
disp('----------')
disp('Algorithm: Incorrect/pseudo algorithm')

% Call the solver
max_iter = 10*m;
[sol_x_pseudo,sol_p_pseudo] = ...
    pseudo_exact_algorithm(A,b,tol_pseudo,max_iter,disp_output_pseudo);
time_total_pseudo = toc;

% Display some information
disp(['Total time elasped for calculating the solution path to BPDN via gradient inclusions: ',...
    num2str(time_total_pseudo), ' seconds.'])
disp('----------')
disp(' ')
disp(['||A*sol_x_pseudo(:,end)-b||_2 = ',num2str(norm(A*sol_x_pseudo(:,end)-b))])
disp(['||sol_x_pseudo(:,end)||_{1} = ',num2str(norm(sol_x_pseudo(:,end),1))])




%% Diagnostics -- These are run if the variable run_diagnostics is == true
% Diagnostics involving the Basis Pursuit Algorithm
if(run_diagnostics && use_bp)
    disp(' ')
    disp('--------------------')
    disp('Diagnostics involving the Basis Pursuit Algorithm from Ciril et al. (2021)')
    disp('--------------------')
    disp(' ')

    % Solution obtained from the method from Ciril et al. (2021)
    disp(['Basis Pursuit solution (residual -- method from Ciril et al. (2021)): ||A*sol_x_bp-b||_2 = ',num2str(norm(A*sol_x_bp(:,end) -b))])
    disp(['Basis Pursuit solution (obj function -- method from Ciril et al. (2021)): ||sol_x_bp||_{1} = ',num2str(norm(sol_x_bp,1))])
    disp(['\ell_{2} norm residual between the methods = ',num2str(norm(sol_x_bp-sol_x_exact(:,end)))])
end

if(run_diagnostics && use_pseudo)
    disp(' ')
    disp('--------------------')
    disp('Diagnostics involving the pseudo/greedy algorithm')
    disp('--------------------')
    disp(' ')

    disp(['\ell_{2} norm residual between the methods = ',num2str(norm(sol_x_pseudo(:,end)-sol_x_exact(:,end)))])
end

end