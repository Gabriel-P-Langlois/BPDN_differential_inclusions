%%  Script for the work on BPDN
%   This script is part of the BPDN project. It performs
%   several numerical experiments to validate a new approach
%   to Basis Pursuit Denoising.

%   Written by Gabriel Provencher Langlois.


%% Options for the script
% Exact homotopy algorithm
disp_output_exact = true;
tol_exact = 1e-10;

% Basis Pursuit algorithm of Ciril et al. (2021)
use_bp = true;
if(use_bp)
    tol_bp = tol_exact;
    disp_output_bp = true;
end

% Fista (robust 1st order optimization method)
use_fista = true;
if(use_fista)
    tol_fista = 1e-10;
    disp_output_fista = false;
    min_iters_fista = 200;      % Minimum # of iterations before 
                                % checking for convergence
end

% Glmnet
use_glmnet = false;

% Run diagnostics. If set to true, lots of checks and diagnostics will be
% run at the end of the script after the methods above are ran (if enabled)
run_diagnostics = true;


%% Examples
% Random seed
rng('default')

% Synthetic data with noise
% Taken from ``Sparse Regression: Scalable Algorithms and Empirical
% Performance" by Dimitris Bertsimas, Jean Pauphilet and Bart Van Parys
%
% Fix n = 10000, use m = 500 or 2500 (this will vary)
% Matrix are columns from N(0,I)
% k_num = Number of nonzero coefficients
% SNR = 1 (medium noise)
% Values are all set to 1


m = 50; n = 250;        % Number of samples and features

k_num = floor(n/50);    % Number of true nonzero coefficients
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
tic
disp(' ')
disp('----------')
disp('The gradient inclusion method for the Lasso.')

[sol_x_exact,sol_p_exact,sol_path] = ...
    homotopy_exp_27_07_2024(A,b,m,n,tol_exact,disp_output_exact,run_diagnostics);

length_path = length(sol_path);
time_total_exact = toc;

% Display a few details
disp(['Total time elasped for the exact Lasso algorithm: ',...
    num2str(time_total_exact), ' seconds.'])
disp('----------')
disp(' ')
disp(['Length of the path: ',num2str(length(sol_path))])
disp(['Basis Pursuit solution: ||A*sol_x_exact(:,end)-b||_2 = ',num2str(norm(A*sol_x_exact(:,end)-b))])


%% Basis Pursuit algorithm
% This solves the problem
%   min_{x \in \Rn} \norm{x}_{1}   s.t. Ax = b
% and outputs both the primal and dual solutions.
if(use_bp)
    tic
    disp(' ')
    disp('----------')
    disp('Basis Pursuit algorithm')
    
    [sol_x_bp, sol_p_bp] = bp_algorithm(A,b,tol_exact,disp_output_bp);
    
    time_bp = toc;
    disp(['Total time elasped for the exact Lasso algorithm: ',...
        num2str(time_bp), ' seconds.'])
    disp('----------')
    disp(' ')
    disp(['Basis Pursuit solution: ||A*sol_x_bp-b||_2 = ',num2str(norm(A*sol_x_exact(:,end)-b))])
    disp(['norm{sol_x_exact(:,end)}_{1}: ',num2str(norm(sol_x_exact(:,end),1))])
    disp(['norm{sol_x_bp}_{1}: ',num2str(norm(sol_x_bp,1))])
    disp(['||sol_x_bp - sol_x_exact(:,end)||_{inf}: ',num2str(norm(sol_x_exact(:,end)-sol_x_bp,inf))])
end

%% FISTA algorithm
% Note: FISTA solves the problem
%       min_{x} 0.5*normsq{Ax-b} + t*normone(x).
% The dual solution is then rescaled to match it to the exact version.
if(use_fista)
    disp(' ')
    disp('----------')
    disp('The FISTA method \w improved screening rule.')

    % Define initial quantities and placeholders for solutions
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
    disp(['Total time elasped for constructing the regularization path with the FISTA algorithm: ',num2str(time_fista), ' seconds.'])
    disp(' ')
    disp('----------')
    disp(' ')

    % Rescale the dual solution
    for(i=1:1:length(sol_path)-1)
        sol_p_fista(:,i) = sol_p_fista(:,i)/sol_path(i);
    end
end


%% Run diagnostics if enabled

% Further diagnostics if the FISTA method was used
if(use_fista && run_diagnostics)
    % Compute the MSE between homotopy sol - FISTA sol -- Primal 
    MSE_primal_homotopy_fista = 0;
    for i=2:1:length(sol_path)-1
        var = norm(sol_x_exact(:,i)-sol_x_fista(:,i))^2;
        MSE_primal_homotopy_fista = MSE_primal_homotopy_fista + var;
        disp(['Iteration: ',num2str(i-1),': (Norm of homotopy sol - FISTA sol)/norm(FISTA sol) (primal): ',...
            num2str(sqrt(var)/norm(sol_x_fista(:,i)))])
    end
    MSE_primal_homotopy_fista = MSE_primal_homotopy_fista/...
        (length(sol_path)-1);
    disp(' ')
    disp(['MSE between homotopy sol and FISTA sol (primal): ',...
        num2str(MSE_primal_homotopy_fista)])
    disp(' ')


    % Compute the MSE between homotopy sol - FISTA sol -- Dual
    MSE_dual_homotopy_fista = 0;
    for i=2:1:length(sol_path)-1
        var = norm(sol_p_exact(:,i)-sol_p_fista(:,i))^2;
        MSE_dual_homotopy_fista = MSE_dual_homotopy_fista + var;
        disp(['Iteration: ',num2str(i-1),': (Norm of homotopy sol - FISTA sol)/norm(FISTA sol) (dual): ',...
            num2str(sqrt(var)/norm(sol_p_fista(:,i)))])
    end
    MSE_dual_homotopy_fista = MSE_dual_homotopy_fista/(length(sol_path)-1);
    disp(' ')
    disp(['MSE between homotopy sol and FISTA sol (dual): ',...
        num2str(MSE_dual_homotopy_fista)])
    disp(' ')
end


% %%%%%%%%%
% %%  Description
% %   Written by Gabriel Provencher Langlois.
% 
% %% DESCRIPTION
% % This script attempts to investigate the behavior of the solution
% % to the LASSO differential inclusion when we reach a direction
% % DkAtopk x dk > 0. This causes issues with the equicorrelation set
% % for the future direction, but I think it can be mitigated.
% 
% %% Notes (25/07/2024)
% % I've made some progress by using the least squares solver
% % and incorporating the second time step, the one that we use to drop
% % the index from the equicorrelation set
% 
% % There is still a bug, though, at iterate 41. But I am getting closer.
% 
% % The issue is that I found a case where uhat < 0 and utilde = 0.
% % I suppose it means we just drop out the index? Is that what we should do?
% 
% 
% %% (26/07/2024)
% % Almost there...
% % There is a bug with the indexing sol(52) vs sol(53) that I need to squash
% 
% 
% %% Options for the script
% display_iterations = false;
% run_diagnostics = true;
% use_fista = true;
% use_glmnet = false;
% 
% % Options for the FISTA algorithm
% if(use_fista)
%     display_output_fista = true;
%     tol_fista = 1e-10;
%     min_iters_fista = 200;
% end
% 
% % Tolerance for the homotopy algorithm
% tol_exact = 1e-10;
% 
% % Random seed
% rng('default')
% 
% 
% %% Examples
% % Synthetic data with noise
% % Taken from ``Sparse Regression: Scalable Algorithms and Empirical
% % Performance" by Dimitris Bertsimas, Jean Pauphilet and Bart Van Parys
% %
% % Fix n = 10000, use m = 500 or 2500 (this will vary)
% % Matrix are columns from N(0,I)
% % k_num = Number of nonzero coefficients
% % SNR = 1 (medium noise)
% % Values are all set to 1
% 
% 
% %m = 40; n = 200;        % Number of samples and features
% m = 40; n = 200;        % Number of samples and features
% 
% k_num = floor(n/50);     % Number of true nonzero coefficients
% val = 1;                % Value of nonzero coefficients
% SNR = 1;                % Signal to noise ratio
% 
% % Design matrix + Normalize it
% A = randn(m,n);
% A = A./sqrt(sum(A.^2)/m);
% 
% xsol = zeros(n,1);
% xsol(randsample(n,k_num)) = val; 
% ind_nonzero_xsol = find(xsol);
% ind_zero_xsol = find(~xsol);
% 
% sigma = norm(A*xsol)/sqrt(SNR);
% b = (A*xsol + sqrt(sigma)*randn(m,1));
% 
% 
% %% Homotopy algorithm
% tic
% disp(' ')
% disp('----------')
% disp('The homotopy method (via gradient inclusions) for the Lasso.')
% 
% [sol_exact_x,sol_exact_p,sol_path] = ...
%     homotopy_exp_27_07_2024(A,b,m,n,tol_exact,display_iterations,run_diagnostics);
% 
% length_path = length(sol_path);
% time_exact_total = toc;
% 
% % Display a few details
% disp(['Total time elasped for the exact Lasso algorithm: ',...
%     num2str(time_exact_total), ' seconds.'])
% disp('----------')
% disp(' ')
% disp(['Length of the path: ',num2str(length(sol_path))])
% disp(length(sol_exact_x(1,:)))
% 
% disp(['Basis Pursuit solution: ||Ax_bp-b||_2 = ',num2str(norm(A*sol_exact_x(:,end)-b))])
% 
% 
% %% Basis Pursuit algorithm
% % tic
% % disp(' ')
% % disp('----------')
% % disp('Basis Pursuit algorithm')
% % 
% % % TODO: Add vanilla basis pursuit algorithm
% % time_bp = toc;
% % disp(['Total time elasped for the exact Lasso algorithm: ',...
% %     num2str(time_exact_total), ' seconds.'])
% % disp('----------')
% % disp(' ')
% % TODO: Compute basis pursuit solution
% % TODO: Compute difference in the solutions
% 
% %% FISTA algorithm
% % Solve min_{x} 0.5*normsq{Ax-b} + t*normone(x)
% if(use_fista)
%     disp(' ')
%     disp('----------')
%     disp('The FISTA method \w improved screening rule.')
% 
%     % Define initial quantities and placeholders for solutions
%     max_iters = 500000;
%     sol_fista_x = zeros(n,length_path-1);     % Primal solutions
%     sol_fista_p = zeros(m,length_path-1);     % Dual solutions 
%     sol_fista_p(:,1) = -b;
% 
%     % Timings
%     tic
%     L22 = svds(A,1)^2;
%     time_fista = toc;
% 
%     % Compute the solution at each kink found by the exact lasso method.
%     for i=2:1:length_path-1
%         % Improved screening rule
%         ind = lasso_screening_rule(sol_path(i),sol_path(i-1),...
%             sol_fista_x(:,i-1),sol_fista_p(:,i-1),A);
% 
%         % Display percentage of zero coefficients
%         if(display_output_fista)
%             disp(['Iteration ',num2str(i),'/',num2str(length_path-1),': Percentage of coefficients found to be zero: ',...
%                 num2str(100-100*sum(ind)/n)])
%         end
% 
%         % Call the FISTA solver
%         tau = 1/L22;
%         tic 
%         [sol_fista_x(ind,i),sol_fista_p(:,i),num_iters] = ...
%             lasso_fista_solver(sol_fista_x(ind,i-1),sol_fista_p(:,i-1),...
%             sol_path(i),A(:,ind),b,tau,max_iters,tol_fista,min_iters_fista);
%         time_iter = toc;
%         time_fista = time_fista + time_iter;
% 
%         % If enabled, the code below will print info about the iteration
%         if(display_iterations)
%             disp(['Solution computed for t/t0 = ', num2str(sol_path(i)/sol_path(1),'%.4e'), '. Number of primal-dual steps = ', num2str(num_iters), '.'])
%             disp(['Time elapsed for solving the Lasso problem = ',...
%                 num2str(time_iter),' seconds.'])
%             disp(' ')
%         end
%     end
%     disp(['Total time elasped for constructing the regularization path with the FISTA algorithm: ',num2str(time_fista), ' seconds.'])
%     disp(' ')
%     disp('----------')
%     disp(' ')
% 
%     % Rescale the dual solution
%     for(i=1:1:length(sol_path)-1)
%         sol_fista_p(:,i) = sol_fista_p(:,i)/sol_path(i);
%     end
% end
% 
% 
% %% Run diagnostics if enabled
% 
% % Further diagnostics if the FISTA method was used
% if(use_fista && run_diagnostics)
%     % Compute the MSE between homotopy sol - FISTA sol -- Primal 
%     MSE_primal_homotopy_fista = 0;
%     for i=2:1:length(sol_path)-1
%         var = norm(sol_exact_x(:,i)-sol_fista_x(:,i))^2;
%         MSE_primal_homotopy_fista = MSE_primal_homotopy_fista + var;
%         disp(['Iteration: ',num2str(i-1),': (Norm of homotopy sol - FISTA sol)/norm(FISTA sol) (primal): ',...
%             num2str(sqrt(var)/norm(sol_fista_x(:,i)))])
%     end
%     MSE_primal_homotopy_fista = MSE_primal_homotopy_fista/...
%         (length(sol_path)-1);
%     disp(' ')
%     disp(['MSE between homotopy sol and FISTA sol (primal): ',...
%         num2str(MSE_primal_homotopy_fista)])
%     disp(' ')
% 
% 
%     % Compute the MSE between homotopy sol - FISTA sol -- Dual
%     MSE_dual_homotopy_fista = 0;
%     for i=2:1:length(sol_path)-1
%         var = norm(sol_exact_p(:,i)-sol_fista_p(:,i))^2;
%         MSE_dual_homotopy_fista = MSE_dual_homotopy_fista + var;
%         disp(['Iteration: ',num2str(i-1),': (Norm of homotopy sol - FISTA sol)/norm(FISTA sol) (dual): ',...
%             num2str(sqrt(var)/norm(sol_fista_p(:,i)))])
%     end
%     MSE_dual_homotopy_fista = MSE_dual_homotopy_fista/(length(sol_path)-1);
%     disp(' ')
%     disp(['MSE between homotopy sol and FISTA sol (dual): ',...
%         num2str(MSE_dual_homotopy_fista)])
%     disp(' ')
% end