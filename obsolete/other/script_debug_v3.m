%Description
%   This script compares the performance of my exact homotopy method.
%
%   INPUT:
%
%       use_fista:  Boolean variable. Set to true to use the FISTA
%                   algorithm and compare the output with the homotopy
%                   algorithm.
%
%       run_diagnostics:    Boolean variable. Run additional tests if
%                           set to true.
%
%   Written by Gabriel Provencher Langlois.

%% Notes
% The issue stems from the zero solution to the NNLS problem...
% Everything breaks down when one of the components is equal to zero!


%%
%% Options for the script
display_iterations = false;
run_diagnostics = true;
use_fista = false;


% Options for the FISTA algorithm
if(use_fista)
    display_output_fista = true;
    tol_fista = 1e-08;
    min_iters_fista = 40;
end

% Tolerance for the homotopy algorithm
tol_exact = 1e-10;

% Random seed
rng('default')


%% Examples
% Synthetic data with noise
% Taken from ``Sparse Regression: Scalable Algorithms and Empirical
% Performance" by Dimitris Bertsimas, Jean Pauphilet and Bart Van Parys
%
% Fix n = 10000, use m = 500 or 2500 (this will vary)
% Matrix are columns from N(0,I)
% k_num = Number of nonzero coefficients
% SNR = 1 (medium noise)
% Values are all set to 1

m = 50; n = 500;        % Number of samples and features
k_num = floor(n/50);     % Number of true nonzero coefficients
val = 1;                % Value of nonzero coefficients
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
    lasso_solver_homotopy_debug_v3(A,b,m,n,tol_exact,display_iterations,run_diagnostics);
time_exact_total = toc;

disp(['Total time elasped for the exact Lasso algorithm: ',...
    num2str(time_exact_total), ' seconds.'])
disp('----------')
disp(' ')

% Remove superfluous components of the path at the end
length_path = length(find(exact_path));
sol_path = exact_path(1:length_path); 
sol_exact_x(:,length_path+1:end) = [];
sol_exact_p(:,length_path+1:end) = [];

% Display a few details
disp(['Length of the path: ',num2str(length(sol_path))])
disp(length(sol_exact_x(1,:)))


%% FISTA algorithm
% Solve min_{x} 0.5*normsq{Ax-b} + t*normone(x)
if(use_fista)
    disp(' ')
    disp('----------')
    disp('The FISTA method \w improved screening rule.')

    % Define initial quantities and placeholders for solutions
    max_iters = 500000;
    sol_fista_x = zeros(n,length_path);     % Primal solutions
    sol_fista_p = zeros(m,length_path);     % Dual solutions 
    sol_fista_p(:,1) = -b;

    % Timings
    tic
    L22 = svds(A,1)^2;
    time_fista = toc;

    % Compute the solution at each kink found by the exact lasso method.
    for i=2:1:length_path-1
        % Improved screening rule
        ind = lasso_screening_rule(sol_path(i),sol_path(i-1),...
            sol_fista_x(:,i-1),sol_fista_p(:,i-1),A);

        % Display percentage of zero coefficients
        if(display_output_fista)
            disp(['Iteration ',num2str(i),'/',num2str(length_path-1),': Percentage of coefficients found to be zero: ',...
                num2str(100-100*sum(ind)/n)])
        end

        % Call the FISTA solver
        tau = 1/L22;
        tic 
        [sol_fista_x(ind,i),sol_fista_p(:,i),num_iters] = ...
            lasso_fista_solver(sol_fista_x(ind,i-1),sol_fista_p(:,i-1),...
            sol_path(i),A(:,ind),b,tau,max_iters,tol_fista,min_iters_fista);
        time_iter = toc;
        time_fista = time_fista + time_iter;

        % If enabled, the code below will print info about the iteration
        if(display_iterations)
            disp(['Solution computed for t/t0 = ', num2str(sol_path(i)/sol_path(1),'%.4e'), '. Number of primal-dual steps = ', num2str(num_iters), '.'])
            disp(['Time elapsed for solving the Lasso problem = ',...
                num2str(time_iter),' seconds.'])
            disp(' ')
        end
    end
    disp(['Total time elasped for constructing the regularization path with the FISTA algorithm: ',num2str(time_fista), ' seconds.'])
    disp(' ')
    disp('----------')
    disp(' ')

    % Rescale the dual solution
    for(i=1:1:length(sol_path)-1)
        sol_fista_p(:,i) = sol_fista_p(:,i)/sol_path(i);
    end
end

%% Run diagnostics if enabled

% Further diagnostics if the FISTA method was used
if(use_fista && run_diagnostics)
    % Compute the MSE between homotopy sol - FISTA sol -- Primal 
    MSE_primal_homotopy_fista = 0;
    for i=1:1:length(sol_path)-1
        var = norm(sol_exact_x(:,i)-sol_fista_x(:,i))^2;
        MSE_primal_homotopy_fista = MSE_primal_homotopy_fista + var;
        disp(['Iteration: ',num2str(i),': (Norm of homotopy sol - FISTA sol)/norm(FISTA sol): ',...
            num2str(sqrt(var)/norm(sol_fista_x(:,i)))])
    end
    MSE_primal_homotopy_fista = MSE_primal_homotopy_fista/...
        (length(sol_path)-1);
    disp(' ')
    disp(['MSE between homotopy sol and FISTA sol (primal): ',...
        num2str(MSE_primal_homotopy_fista)])
    disp(' ')


    % Compute the MSE between homotopy sol - FISTA sol -- Dual
    MSE_dual_homotopy_fista = 0;
    for i=1:1:length(sol_path)-1
        var = norm(sol_exact_p(:,i)-sol_fista_p(:,i))^2;
        MSE_dual_homotopy_fista = MSE_dual_homotopy_fista + var;
        disp(['Iteration: ',num2str(i),': (Norm of homotopy sol - FISTA sol)/norm(FISTA sol): ',...
            num2str(sqrt(var)/norm(sol_fista_p(:,i)))])
    end
    MSE_dual_homotopy_fista = MSE_dual_homotopy_fista/(length(sol_path)-1);
    disp(' ')
    disp(['MSE between homotopy sol and FISTA sol: ',...
        num2str(MSE_dual_homotopy_fista)])
    disp(' ')
end
