%% Description
% This script compares the performance of my exact homotopy method 
% vs the state-of-the-art GLMNET code.

% Written by Gabriel Provencher Langlois


%% Notes


%% Options for the script
% Specify whether to use FISTA or glmnet or both.
use_fista = true;
use_glmnet = true;

% Option to output the result at each iteration of the exact
% lasso algorithm implementations.
display_iterations = false;

% Option to output the result at each iteration of the FISTA algorithm.
display_output_fista = false;

% Tolerances + minimum number of iterations for the FISTA algorithm
tol_exact = 1e-10;
tol_fista = 1e-04;  min_iters_fista = 100;
RelTol_glmnet = 1e-04;  % Default: 1e-04

% Random seed
rng('default')


%% Data acquisition
% Example 1 -- Synthetic data with noise
% Taken from ``Sparse Regression: Scalable Algorithms and Empirical
% Performance" by Dimitris Bertsimas, Jean Pauphilet and Bart Van Parys
%
% Fix n = 10000, use m = 500 or 2500 (this will vary)
% Matrix are columns from N(0,I)
% k_num = Number of nonzero coefficients
% SNR = 1 (medium noise)
% Values are all set to 1

m = 1000; n = 10000;     % Number of samples and features
k_num = m;              % Number of true nonzero coefficients
val = 1;                % Value of nonzero coefficients
prop = k_num/n;         % Proportion of coefficients that are equal to val.
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


%% Method 1: Exact Lasso algorithm
% Note: We use the standard solver for this.
disp(' ')
disp('----------')
disp('The homotopy method (via gradient inclusions) for the Lasso.')

tic
[sol_exact_x,sol_exact_p,exact_path] = ...
    lasso_homotopy_solver(A,b,tol_exact,display_iterations);
time_exact_total = toc;

disp(['Total time elasped for the exact Lasso algorithm: ',...
    num2str(time_exact_total), ' seconds.'])
disp('----------')

lambda = exact_path; lambda(end) = [];
length_path = length(lambda);


%% Method 2: The FISTA algorithm with (improved) screening rule
if(use_fista)
    disp(' ')
    disp('----------')
    disp('The FISTA method \w improved screening rule.')

    % Define initial quantities and placeholders for solutions
    max_iters = 500000;
    sol_fista_x = zeros(n,length_path);     % Primal solutions
    sol_fista_p = zeros(m,length_path);     % Dual solutions 
    sol_fista_p(:,1) = -b;                  % Dual solution at lambda_est

    % Timings
    tic
    L22 = svds(A,1)^2;
    time_fista = toc;

    % Compute the solution at each kink found by the exact lasso method.
    for i=2:1:length_path
        % Improved screening rule
        ind = lasso_screening_rule(lambda(i),lambda(i-1),...
            sol_fista_x(:,i-1),sol_fista_p(:,i-1),A);

        % Display percentage of zero coefficients
        if(display_output_fista)
            disp(['Iteration ',num2str(i),'/',num2str(length_path)])
            disp(['Percentage of coefficients found to be zero: ',...
                num2str(100-100*sum(ind)/n)])
        end

        % Call the FISTA solver
        tau = 1/L22;
        tic 
        [sol_fista_x(ind,i),sol_fista_p(:,i),num_iters] = ...
            lasso_fista_solver(sol_fista_x(ind,i-1),sol_fista_p(:,i-1),...
            lambda(i),A(:,ind),b,tau,max_iters,tol_fista,min_iters_fista);
        time_iter = toc;
        time_fista = time_fista + time_iter;

        % If enabled, the code below will print info about the iteration
        if(display_output_fista)
            disp(['Solution computed for lambda = ', num2str(lambda(i),'%.4e'), '. Number of primal-dual steps = ', num2str(num_iters), '.'])
            disp(['Time elapsed for solving the Lasso problem = ',...
                num2str(time_iter),' seconds.'])
            disp(' ')
        end
    end
    disp(['Total time elasped for constructing the regularization path with the FISTA algorithm: ',num2str(time_fista), ' seconds.'])
    disp('----------')
end


%% Method 3: The GLMNET software
if(use_glmnet)
    disp(' ')
    disp('----------')
    disp('The GLMNET method')

    tic
    warning('off');
    % Warning 1: The matrix A and response vector b must be rescaled
    % by a factor of sqrt(m) due to how its implemented
    sol_glmnet_x = lasso(sqrt(m)*A,sqrt(m)*b,'lambda',lambda,...
        'Intercept',false,'RelTol',RelTol_glmnet);

    % Warning 2: The lasso function returns the solutions of the
    % regularization path starting from its lowest value.
    sol_glmnet_x = flip(sol_glmnet_x,2);

    % Warning 3: Unlike the exact lasso algorithm, the dual solution is
    % Ax-b
    sol_glmnet_p = (A*sol_glmnet_x - b);
    time_glmnet = toc;

    disp(['Total time elasped for constructing the regularization path with the GLMNET algorithm: ',num2str(time_glmnet), ' seconds.'])
    disp('----------')
end
%% END OF CALCULATION(S) OF THE SOLUTIONS


%% START OF THE DATA ANALYSIS PART
%% Diagnostics I: Properties of the solutions
%%%%%%%%%%%%%%%
% 1) Check how well the optimality condition p = (Ax-b)/t is satisfied at
% each kink (MSE). If accurate, this number should be close
% to machine precision.
%%%%%%%%%%%%%%%
% Homotopy method (via gradient inclusions)
check_opt_cond_1_exact = 0;
for i=1:1:length(exact_path)-1
    check_opt_cond_1_exact = check_opt_cond_1_exact + ...
        norm(sol_exact_p(:,i) - (A*sol_exact_x(:,i)-b)/exact_path(i))^2;
end
check_opt_cond_1_exact = check_opt_cond_1_exact/(length(exact_path)-1);

% FISTA method
if(use_fista)
    check_opt_cond_1_fista = 0;
    for i=1:1:length(exact_path)-1
        check_opt_cond_1_fista = check_opt_cond_1_fista + ...
            norm(sol_fista_p(:,i) - (A*sol_fista_x(:,i) - b))^2;
    end
    check_opt_cond_1_fista = check_opt_cond_1_fista/(length(exact_path)-1);
end

% GLMNET method
if(use_glmnet)
    check_opt_cond_1_glmnet = 0;
    for i=1:1:length(exact_path)-1
        check_opt_cond_1_glmnet = check_opt_cond_1_glmnet + ...
            norm(sol_glmnet_p(:,i) - (A*sol_glmnet_x(:,i) - b))^2;
    end
    check_opt_cond_1_glmnet = ...
        check_opt_cond_1_glmnet/(length(exact_path)-1);
end



%%%%%%%%%%%%%%%
% 2) Compute how far -A^{\top}\bp_{k} deviates from the condition
%    |-A^{\top}\bp_{k}| <= 1. This number should be equal to 1, up to
%    machine precision.
%
%   Note: GLMNET will not satisfy it to a desirable accuracy because
%   it always add a tiny perturbation \eps*normsq{x}/2 in the LASSO
%   problem.
%%%%%%%%%%%%%%%
% Homotopy method (via gradient inclusions)
check_opt_cond_2_exact = 0;
for i=1:1:length(exact_path)-1
    check_opt_cond_2_exact = ...
        max(check_opt_cond_2_exact,norm(A.'*sol_exact_p(:,i),"inf"));
end

% FISTA method
if(use_fista)
    check_opt_cond_2_fista = 0;
    for i=1:1:length(exact_path)-1
        check_opt_cond_2_fista = ...
            max(check_opt_cond_2_fista,norm(A.'*sol_fista_p(:,i),"inf")/exact_path(i));
    end
end

% GLMNET method
if(use_glmnet)
    check_opt_cond_2_glmnet = 0;
    for i=1:1:length(exact_path)-1
        check_opt_cond_2_glmnet = ...
            max(check_opt_cond_2_glmnet,norm(A.'*sol_glmnet_p(:,i),"inf")/exact_path(i));
    end
end



%%%%%%%%%%%%%%%
% 3) Compute the MSE between the primal solutions
%%%%%%%%%%%%%%%
% Homotopy vs FISTA
if(use_fista)
    comp_primal_exact_vs_fista = 0;
    for i=1:1:length(exact_path)-1
        comp_primal_exact_vs_fista = comp_primal_exact_vs_fista + ...
            norm(sol_exact_x(:,i)-sol_fista_x(:,i))^2;
    end
    comp_primal_exact_vs_fista = ...
        comp_primal_exact_vs_fista/(length(exact_path)-1);
end

% Homotopy vs GLMNET
if(use_glmnet)
    comp_primal_exact_vs_glmnet = 0;
    for i=1:1:length(exact_path)-1
        comp_primal_exact_vs_glmnet = comp_primal_exact_vs_glmnet + ...
            norm(sol_exact_x(:,i)-sol_glmnet_x(:,i))^2;
    end
    comp_primal_exact_vs_glmnet = ...
        comp_primal_exact_vs_glmnet/(length(exact_path)-1);
end

% FISTA vs GLMNET
if(use_fista && use_glmnet)
    comp_primal_fista_vs_glmnet = 0;
    for i=1:1:length(exact_path)-1
        comp_primal_fista_vs_glmnet = comp_primal_fista_vs_glmnet + ...
            norm(sol_fista_x(:,i)-sol_glmnet_x(:,i))^2;
    end
    comp_primal_fista_vs_glmnet = ...
        comp_primal_fista_vs_glmnet/(length(exact_path)-1);
end


%%%%%%%%%%%%%%%
% 4) Compute the MSE between the dual solutions
%%%%%%%%%%%%%%%
% Homotopy vs FISTA
if(use_fista)
    comp_dual_exact_vs_fista = 0;
    for i=1:1:length(exact_path)-1
        comp_dual_exact_vs_fista = comp_dual_exact_vs_fista + ...
            norm(exact_path(i)*sol_exact_p(:,i)-sol_fista_p(:,i))^2;
    end
    comp_dual_exact_vs_fista = ...
        comp_dual_exact_vs_fista/(length(exact_path)-1);
end

% Homotopy vs GLMNET
if(use_glmnet)
    comp_dual_exact_vs_glmnet = 0;
    for i=1:1:length(exact_path)-1
        comp_dual_exact_vs_glmnet = comp_dual_exact_vs_glmnet + ...
            norm(exact_path(i)*sol_exact_p(:,i)-sol_glmnet_p(:,i))^2;
    end
    comp_dual_exact_vs_glmnet = ...
        comp_dual_exact_vs_glmnet/(length(exact_path)-1);
end

% FISTA vs GLMNET
if(use_fista && use_glmnet)
    comp_dual_fista_vs_glmnet = 0;
    for i=1:1:length(exact_path)-1
        comp_dual_fista_vs_glmnet = comp_dual_fista_vs_glmnet + ...
            norm(sol_fista_p(:,i)-sol_glmnet_p(:,i))^2;
    end
    comp_dual_fista_vs_glmnet = ...
        comp_dual_fista_vs_glmnet/(length(exact_path)-1);
end



%%%%%%%%%%%%%%%
% 5) Compute 0.5*normsq(Ax-b) + t*normone(x) for each methods
%    and use this to compare them. 
%
%    Also compare the magnitude of the dual solution of glmnet and fista vs
%    the exact homotopy
%%%%%%%%%%%%%%%
% Calculate the value of the primal problem at the solutions for the
% homotopy, fista and glmnet methods.
val_primal_exact = zeros(m,1);
for i=1:1:length(exact_path)-1
    val_primal_exact(i) = 0.5*(norm(A*sol_exact_x(:,i)-b)^2) + ...
        exact_path(i)*norm(sol_exact_x(:,i),1);
end

if(use_fista)
    val_primal_fista = zeros(m,1);
    for i=1:1:length(exact_path)-1
        val_primal_fista(i) = 0.5*norm(A*sol_fista_x(:,i)-b)^2 + ...
            exact_path(i)*norm(sol_fista_x(:,i),1);
    end
end

if(use_glmnet)
    val_primal_glmnet = zeros(m,1);
    for i=1:1:length(exact_path)-1
        val_primal_glmnet(i) = 0.5*norm(A*sol_glmnet_x(:,i)-b)^2 + ...
            exact_path(i)*norm(sol_glmnet_x(:,i),1);
    end
end

% i) Plot the ratio of the values of the primal problems at every kink.
if(use_fista && use_glmnet)
figure(1)
    subplot(1,3,1)
    plot(1:m,val_primal_exact./val_primal_fista)
    title('Ratio of 0.5*normsq(Ax-b)+t*normone(x) vs t (exact/fista)')
    
    subplot(1,3,2)
    plot(1:m,val_primal_exact./val_primal_glmnet)
    title('Ratio of 0.5*normsq(Ax-b)+t*normone(x) vs t (exact/glmnet)')
    
    subplot(1,3,3)
    plot(1:m,val_primal_glmnet./val_primal_fista)
    title('Ratio of 0.5*normsq(Ax-b)+t*normone(x) vs t (glmnet/fista)')
    %
    % Important note (from GPL): In the plots I obtain, I always see
    % exact >= fista and exact >= glmnet. What I also find, however, is that
    % the primal and dual optimality conditions are satisfied more tightly for
    % the exact method than the fista or glmnet methods (specifically,
    % the optimality conditions -A.'*p \in \partial \normone{x}.
    %
    % Since satisfying the optimality conditions <---> solving the problem,
    % it seems likely the FISTA and GLMNET algorithms are unable to resolve
    % the solution of the primal problem correctly.
    
    % ii) Plot the ratio of the norm of the dual solution
    % For the exact solution, rescale p by the hyperparameter t
    norm_exact_axb = exact_path.'.*vecnorm(sol_exact_p,2);
    norm_exact_axb = norm_exact_axb(1:end-1);
    
    norm_fista_axb = vecnorm(sol_fista_p,2);
    norm_glmnet_axb = vecnorm(sol_glmnet_p,2);
    
    figure(2)
    subplot(1,3,1)
    plot(1:m,norm_exact_axb./norm_fista_axb)
    title('Ratio of norm(Ax-b) vs t (exact/fista)')
    
    subplot(1,3,2)
    plot(1:m,norm_exact_axb./norm_glmnet_axb)
    title('Ratio of norm(Ax-b) vs t (exact/glmnet)')
    
    subplot(1,3,3)
    plot(1:m,norm_glmnet_axb./norm_fista_axb)
    title('Ratio of norm(Ax-b) vs t (glmnet/fista)')
    %
    % Important note (from GPL): In the plots above, I get that
    % norm(Ax-b,2) is smaller using the solution obtained from the homotopy
    % method. 
    %
    % This is exactly what we would expect; after all, we picked
    % x precisely so that norm(Ax-b) is minimized at every iteration of the 
    % homotopy algorithm. This suggests that the homotopy method
    % is doing the right thing.
end


%% Diagnostics 2
% WIP
% %%%%%%%%%%%%%%%
% % Check for accuracy and the false discovery rate
% % The higher the accuracy, the better.
% %%%%%%%%%%%%%%%
% ind_nonzero_exact = find(abs(sol_exact_x(:,k_num)) > tol_exact);
% check_acc_exact = 100*length(intersect(ind_nonzero_exact,ind_nonzero_xsol))/k_num;
% check_fdr_exact = 100*length(intersect(ind_nonzero_exact,ind_zero_xsol))...
%     /length(ind_nonzero_exact);
% % 
% % if(use_fista)
% %     check_acc_fista = sum(abs(sol_fista_x(ind_xsol,end)) > tol_fista)/k_num;
% % end
% % 
% if(use_glmnet)
%     ind_nonzero_glmnet = find(abs(sol_glmnet_x(:,k_num)) > RelTol_glmnet);
%     check_acc_glmnet = 100*length(intersect(ind_nonzero_glmnet,ind_nonzero_xsol))/k_num;
%     check_fdr_glmnet = 100*length(intersect(ind_nonzero_glmnet,ind_zero_xsol))...
%         /length(ind_nonzero_glmnet);
% end
% 
% 
% %%%%%%%%%%%%%%%
% % 2) False discovery rate
% %%%%%%%%%%%%%%%