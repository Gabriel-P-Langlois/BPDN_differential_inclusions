%% Description
% This is a script to run and test the exact lasso algorithm developed
% by Gabriel P. Langlois


%% Options for the script
% Option to output the result at each iteration of the exact
% lasso algorithm implementations, if desired
display_iterations = false;

% Tolerance for the exact lasso algorithm
tol = 1e-08;

% Random seed
rng('default')


%% Data acquisition
% Example 1 -- Synthetic data with noise
m = 100; n = 100000;        % Number of samples and features
val = 10;                   % Value of nonzero coefficients
prop = 0.1;                 % Proportion of coefficients that are equal to val.
SNR = 5;                    % Signal to noise ratio

% Design matrix + Normalize it
A = randn(m,n);
A = A./sqrt(sum(A.^2)/m);

xsol = zeros(n,1);
xsol(randsample(n,n*prop)) = val; 
ind_xsol = find(xsol);

sigma = norm(A*xsol)/sqrt(SNR);
b = (A*xsol + sqrt(sigma)*randn(m,1));


%% Method 1: Exact Lasso algorithm (Regular, working code)
% This one solves a least-squares problem from scratch every time.
disp(' ')
disp('----------')
disp('The Lasso homotopy method -- Standard implementation.')

tic
[sol_exact_1_x,sol_exact_1_p,exact_1_path] = ...
    lasso_homotopy_solver(A,b,tol,display_iterations);
time_exact_1_total = toc;


disp(['Total time elasped for the exact Lasso algorithm: ',...
    num2str(time_exact_1_total), ' seconds.'])
disp('----------')


%% Method 2: Exact Lasso algorithm -- QR algorithm with rank updates
disp(' ')
disp('----------')
disp('The Lasso homotopy method -- Implementation with rank-one updates.')

tic
[sol_exact_2_x,sol_exact_2_p,exact_2_path] = ...
    lasso_homotopy_solver_QR_rank_update(A,b,tol,display_iterations);
time_exact_2_total = toc;

disp(['Total time elasped for the exact Lasso algorithm ...' ...
    '(\w QR decomposition): ',num2str(time_exact_2_total), ' seconds.'])
disp('----------')


%% Diagnostics
%%%%%%%%%%%%%%%
% 1) Check how well the optimality condition p = (Ax-b)/t is satisfied at
% each kink and take the average. If accurate, this number should be close
% to machine precision.
%%%%%%%%%%%%%%%

% Standard method
check_opt_cond_1_method_1 = 0;
for i=1:1:length(exact_1_path)-1
    check_opt_cond_1_method_1 = check_opt_cond_1_method_1 + ...
        norm(sol_exact_1_p(:,i) - (A*sol_exact_1_x(:,i)-b)/exact_1_path(i))^2;
end
check_opt_cond_1_method_1 = ...
    check_opt_cond_1_method_1/(length(exact_1_path)-1);

% QR rank-one update method
check_opt_cond_1_method_2 = 0;
for i=1:1:length(exact_2_path)-1
    check_opt_cond_1_method_2 = check_opt_cond_1_method_2 + ...
        norm(sol_exact_2_p(:,i) - (A*sol_exact_2_x(:,i)-b)/exact_2_path(i))^2;
end
check_opt_cond_1_method_2 = ...
    check_opt_cond_1_method_2/(length(exact_2_path)-1);



%%%%%%%%%%%%%%%
% 2) Compute how far -A^{\top}\bp_{k} deviates from the condition
%    |-A^{\top}\bp_{k}| <= 1. This number should be equal to 1, up to
%    machine precision.
%%%%%%%%%%%%%%%

% Standard method
check_opt_cond_2_method_1 = 0;
for i=1:1:length(exact_1_path)-1
    check_opt_cond_2_method_1 = ...
        max(check_opt_cond_2_method_1,norm(A.'*sol_exact_1_p(:,i),"inf"));
end

% QR rank-one update method
check_opt_cond_2_method_2 = 0;
for i=1:1:length(exact_2_path)-1
    check_opt_cond_2_method_2 = ...
        max(check_opt_cond_2_method_2,norm(A.'*sol_exact_2_p(:,i),"inf"));
end



%%%%%%%%%%%%%%%
% 3) Compare the dual solutions of the standard exact lasso implementation
%    vs its QR rank one update implementation
%%%%%%%%%%%%%%%

check_dual_match = 0;
for i=1:1:length(exact_2_path)
    check_dual_match = check_dual_match + ...
        norm(sol_exact_1_p(:,i) - sol_exact_2_p(:,i))^2;
end
check_dual_match = check_dual_match/length(exact_2_path);

