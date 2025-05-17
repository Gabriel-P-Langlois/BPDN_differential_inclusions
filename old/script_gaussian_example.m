function script_gaussian_example(m,n,tol_exact,disp_output_exact,...
    run_opt_cond_checks,use_bp,tol_bp,disp_output_bp,...
    use_fista,tol_fista,disp_output_fista,min_iters_fista,...
    use_glmnet,tol_glmnet,run_diagnostics)

%%  Script for the work on the BPDN solution via Differential Inclusions
%   This script is part of the BPDN project. It performs
%   several numerical experiments to validate a new approach
%   to Basis Pursuit Denoising.

%   This script is called from an appropriate runall.m function.

%   Written by Gabriel Provencher Langlois.

% Input (TBC.):
% m:
% n:
% tol_exact:
% disp_output_exact:
% run_opt_cond_checks:
% use_bp:
% tol_bp:
% disp_output_bp:
% use_fista:
% tol_fista:
% disp_output_fista:
% min_iters_fista:
% use_glmnet:
% tol_glmnet:
% run_diagnostics:



%% Example
% Synthetic data with Gaussian noise
% Taken from ``Sparse Regression: Scalable Algorithms and Empirical
% Performance" by Dimitris Bertsimas, Jean Pauphilet and Bart Van Parys

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
    [sol_x_bp, ~] = BP_exact_algorithm(A,b,-b/norm(A.'*b,inf),tol_bp,disp_output_bp);
    time_bp = toc;

    % Display some information
    disp(['Total time elasped for the exact Lasso algorithm: ',...
        num2str(time_bp), ' seconds.'])
    disp('----------')
    disp(' ')
end


%% FISTA algorithm
% Note: FISTA solves the problem
%       min_{x} 0.5*normsq{Ax-b} + t*normone(x).
% Note: The dual solution is rescaled at the end to match it 
if(use_fista)
    disp(' ')
    disp('----------')
    disp('Algorithm: The FISTA method with (improved) screening rule.')

    % Define initial quantities and placeholders for solutions
    % Note: We take length_path-1 because the FISTA algorithm
    % cannot compute the solution to Basis Pursuit adequately.
    max_iters = 500000;
    sol_x_fista = zeros(n,length_path-1);     % Primal solutions
    sol_p_fista = zeros(m,length_path-1);     % Dual solutions 
    sol_p_fista(:,1) = -b;

    % Timings
    tic
    L22 = svds(A,1)^2;
    time_fista = toc;

    % Compute the solution at each kink found by the exact lasso method.
    for i=2:1:length_path-1
        % Improved screening rule
        ind = lasso_screening_rule(sol_path(i),sol_path(i-1),...
            sol_x_fista(:,i-1),sol_p_fista(:,i-1),A);

        % Display percentage of zero coefficients
        if(disp_output_fista)
            disp(['Iteration ',num2str(i),'/',num2str(length_path-1),': Percentage of coefficients found to be zero: ',...
                num2str(100-100*sum(ind)/n)])
        end

        % Call the FISTA solver
        tau = 1/L22;
        tic 
        [sol_x_fista(ind,i),sol_p_fista(:,i),num_iters] = ...
            lasso_fista_solver(sol_x_fista(ind,i-1),sol_p_fista(:,i-1),...
            sol_path(i),A(:,ind),b,tau,max_iters,tol_fista,min_iters_fista);
        time_iter = toc;
        time_fista = time_fista + time_iter;

        % If enabled, the code below will print info about the iteration
        if(disp_output_fista)
            disp(['Solution computed for t/t0 = ', num2str(sol_path(i)/sol_path(1),'%.4e'), '. Number of primal-dual steps = ', num2str(num_iters), '.'])
            disp(['Time elapsed for solving the Lasso problem = ',...
                num2str(time_iter),' seconds.'])
            disp(' ')
        end
    end

    % Display some information
    disp(['Total time elasped for constructing the regularization path with the FISTA algorithm: ',num2str(time_fista), ' seconds.'])
    disp(' ')
    disp('----------')
    disp(' ')

    % Rescale the dual solution
    for(i=1:1:length(sol_path)-1)
        sol_p_fista(:,i) = sol_p_fista(:,i)/sol_path(i);
    end
end



%% GLMNET algorithm
if(use_glmnet)
    disp(' ')
    disp('----------')
    disp('Algorithm: GLMNET')

    tic
    warning('off');
    % Warning 1: The matrix A and response vector b must be rescaled
    % by a factor of sqrt(m) due to how its implemented
    sol_x_glmnet = lasso(sqrt(m)*A,sqrt(m)*b,'lambda',sol_path(1:end-1),...
        'Intercept',false,'RelTol',tol_glmnet);

    % Warning 2: The lasso function returns the solutions of the
    % regularization path starting from its lowest value.
    sol_x_glmnet = flip(sol_x_glmnet,2);

    % Warning 3: Unlike the exact lasso algorithm, the dual solution is
    % Ax-b
    sol_p_glmnet = (A*sol_x_glmnet - b);
    for(i=1:1:length(sol_path)-1)
        sol_p_glmnet(:,i) = sol_p_glmnet(:,i)/sol_path(i);
    end
    time_glmnet = toc;

    disp(['Total time elasped for constructing the regularization path with the GLMNET algorithm: ',num2str(time_glmnet), ' seconds.'])
    disp('----------')
end



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


% Diagnostics involving FISTA
if(run_diagnostics && use_fista)
    disp(' ')
    disp('--------------------')
    disp('Diagnostics involving the FISTA method')
    disp('--------------------')
    disp(' ')

    % Compute the MSE between the primal solutions obtained by the exact algorithm
    % and the FISTA algorithm. Here, the MSE is averaged over the solution
    % path (except at t = 0).
    MSE_primal_homotopy_fista = 0;
    for i=2:1:length(sol_path)-1
        var = norm(sol_x_exact(:,i)-sol_x_fista(:,i))^2;
        MSE_primal_homotopy_fista = MSE_primal_homotopy_fista + var;
        disp(['Iteration: ',num2str(i-1),': (Norm of exact solution - FISTA solution)/norm(FISTA solution) (Primal): ',...
            num2str(sqrt(var)/norm(sol_x_fista(:,i)))])
    end
    
    MSE_primal_homotopy_fista = MSE_primal_homotopy_fista/...
        (length(sol_path)-1);
    disp(' ')
    disp(['MSE between exact solution and FISTA solution (Primal): ',...
        num2str(MSE_primal_homotopy_fista)])
    disp(' ')


    % Compute the MSE between the dual solutions obtained by the exact algorithm
    % and the FISTA algorithm. Here, the MSE is averaged over the solution
    % path (except at t = 0).
    MSE_dual_homotopy_fista = 0;
    for i=2:1:length(sol_path)-1
        var = norm(sol_p_exact(:,i)-sol_p_fista(:,i))^2;
        MSE_dual_homotopy_fista = MSE_dual_homotopy_fista + var;
        disp(['Iteration: ',num2str(i-1),': (Norm of exact solution - FISTA solution)/norm(FISTA solution) (Dual): ',...
            num2str(sqrt(var)/norm(sol_p_fista(:,i)))])
    end
    MSE_dual_homotopy_fista = MSE_dual_homotopy_fista/(length(sol_path)-1);
    disp(' ')
    disp(['MSE between exact solution and FISTA solution (Dual): ',...
        num2str(MSE_dual_homotopy_fista)])
    disp(' ')
end


% Diagnostics involving glmnet
if(run_diagnostics && use_glmnet)
    disp(' ')
    disp('--------------------')
    disp('Diagnostics involving the GLMNET method')
    disp('--------------------')
    disp(' ')

    % Compute the MSE between the primal solutions obtained by the exact algorithm
    % and the FISTA algorithm. Here, the MSE is averaged over the solution
    % path (except at t = 0).
    MSE_primal_homotopy_glmnet = 0;
    for i=2:1:length(sol_path)-1
        var = norm(sol_x_exact(:,i)-sol_x_glmnet(:,i))^2;
        MSE_primal_homotopy_glmnet = MSE_primal_homotopy_glmnet + var;
        disp(['Iteration: ',num2str(i-1),': (Norm of exact solution - glmnet solution)/norm(glmnet solution) (Primal): ',...
            num2str(sqrt(var)/norm(sol_x_glmnet(:,i)))])
    end
    
    MSE_primal_homotopy_glmnet = MSE_primal_homotopy_glmnet/...
        (length(sol_path)-1);
    disp(' ')
    disp(['MSE between exact solution and glmnet solution (Primal): ',...
        num2str(MSE_primal_homotopy_glmnet)])
    disp(' ')


    % Compute the MSE between the dual solutions obtained by the exact algorithm
    % and the glmnet algorithm. Here, the MSE is averaged over the solution
    % path (except at t = 0).
    MSE_dual_homotopy_glmnet = 0;
    for i=2:1:length(sol_path)-1
        var = norm(sol_p_exact(:,i)-sol_p_glmnet(:,i))^2;
        MSE_dual_homotopy_glmnet = MSE_dual_homotopy_glmnet + var;
        disp(['Iteration: ',num2str(i-1),': (Norm of exact solution - glmnet solution)/norm(glmnet solution) (Dual): ',...
            num2str(sqrt(var)/norm(sol_p_glmnet(:,i)))])
    end
    MSE_dual_homotopy_glmnet = MSE_dual_homotopy_glmnet/(length(sol_path)-1);
    disp(' ')
    disp(['MSE between exact solution and glmnet solution (Dual): ',...
        num2str(MSE_dual_homotopy_glmnet)])
    disp(' ')
end

% Diagnostics involving FISTA and glmnet
if(use_fista && use_glmnet)
    disp(' ')
    disp('--------------------')
    disp('Comparing FISTA vs GLMNET...')
    disp('--------------------')
    disp(' ')

    % Compute the MSE between the primal solutions obtained by FISTA and
    % GLMNET. Here, the MSE is averaged over the solution
    % path (except at t = 0).
    MSE_primal_fista_glmnet = 0;
    for i=2:1:length(sol_path)-1
        var = norm(sol_x_fista(:,i)-sol_x_glmnet(:,i))^2;
        MSE_primal_fista_glmnet = MSE_primal_fista_glmnet + var;
        disp(['Iteration: ',num2str(i-1),': (Norm of fista solution - glmnet solution)/norm(glmnet solution) (Primal): ',...
            num2str(sqrt(var)/norm(sol_x_glmnet(:,i)))])
    end
    
    MSE_primal_fista_glmnet = MSE_primal_fista_glmnet/...
        (length(sol_path)-1);
    disp(' ')
    disp(['MSE between fista solution and glmnet solution (Primal): ',...
        num2str(MSE_primal_fista_glmnet)])
    disp(' ')


    % Compute the MSE between the dual solutions obtained by the fista algorithm
    % and the glmnet algorithm. Here, the MSE is averaged over the solution
    % path (except at t = 0).
    MSE_dual_fista_glmnet = 0;
    for i=2:1:length(sol_path)-1
        var = norm(sol_p_fista(:,i)-sol_p_glmnet(:,i))^2;
        MSE_dual_fista_glmnet = MSE_dual_fista_glmnet + var;
        disp(['Iteration: ',num2str(i-1),': (Norm of fista solution - glmnet solution)/norm(glmnet solution) (Dual): ',...
            num2str(sqrt(var)/norm(sol_p_glmnet(:,i)))])
    end
    MSE_dual_fista_glmnet = MSE_dual_fista_glmnet/(length(sol_path)-1);
    disp(' ')
    disp(['MSE between fista solution and glmnet solution (Dual): ',...
        num2str(MSE_dual_fista_glmnet)])
    disp(' ')
end
end