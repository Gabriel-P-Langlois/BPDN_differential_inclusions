%% RUNALL Script
% From the project directory, run
% run ./results/test_new_bp_algorithm/runall.m

% This script solves the Basis Pursuit Problem
%   min_{u \in \Rn} \normone{u}  subject to Au = b
% using the algorithm Ciril et al. (2023) developed.

% It then compares the output with the tentative approach
% that GPL developed, which solves
%   min_{x \in \Rn} \normone{x} subject to A(x+v) = b



%% Example 1 -- Random Gaussian design -- Setup
% Options
rng('default')
tol_bp = 1e-8;
disp_output_bp = true;

% Parameters
m = 100; n = 100;
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



%% Algorithm 1 -- Exact BP from Ciril et al. (2023)
disp(' ')
disp('----------')
disp('Basis Pursuit algorithm (Ciril et al. (2023))')

% Call the BP solver
disp('Starting...')
tic
[sol_x, sol_p] = BP_exact_algorithm(A,b,-b/norm(A.'*b,inf),tol_bp,disp_output_bp);
time_bp = toc;
disp('Done!')



%% Algoritm 2 -- Modified (Silly) BPDN --> BP algorithm
disp(' ')
disp('----------')
disp('Silly BPDN algorithm (homotopy in parameter/data)')

% Call the BP solver
disp('Starting...')
tic
[sol_x_silly, sol_p_silly, sol_t_silly, x_silly, b_silly] = BP_silly_algorithm(A,b,tol_bp,true);
time_bp_silly = toc;
disp('Done!')


% Call the BP solver and use a warm start
disp(' ')
disp('----------')
disp('Basis Pursuit algorithm (Ciril et al. (2023)) with (silly) warm start')
disp('Starting...')
tic
[sol_x_silly_warm, sol_p_silly_warm] = BP_exact_algorithm(A,b,sol_p_silly(:,m),tol_bp,disp_output_bp);
time_bp_silly_warm = toc;
disp(sum(abs(-A.'*sol_p_silly(:,m)) > 1-tol_bp))
disp('Done!')


% %% Algoritm 3 -- Modified (Average) BPDN --> BP algorithm
% disp(' ')
% disp('----------')
% disp('Average BPDN algorithm (homotopy in parameter/data)')
% 
% % Call the BP solver
% disp('Starting...')
% tic
% [sol_x_average, sol_bp_average, sol_t_average, x_average, b_average] = ...
%     BP_average_algorithm(A,b,tol_bp,true);
% time_bp_average = toc;
% disp('Done!')
% 
% 
% % Call the BP solver and use a warm start
% disp(' ')
% disp('----------')
% disp('Basis Pursuit algorithm (Ciril et al. (2023)) with (average) warm start')
% disp('Starting...')
% tic
% [sol_x_average_warm, ~] = BP_exact_algorithm(A,b,sol_bp_average(:,m),tol_bp,disp_output_bp);
% time_bp_average_warm = toc;
% disp('Done!')
% 
% 
% %% Algoritm 4 -- Modified (threshold) BPDN --> BP algorithm
% disp(' ')
% disp('----------')
% disp('Thresholding BPDN algorithm (homotopy in parameter/data)')
% 
% % Call the BP solver
% disp('Starting...')
% tic
% [sol_x_threshold, sol_bp_threshold, sol_t_threshold, x_threshold, b_threshold] = ...
%     BP_thresholding_algorithm(A,b,tol_bp,true);
% time_bp_threshold = toc;
% disp('Done!')
% 
% 
% % Call the BP solver and use a warm start
% disp(' ')
% disp('----------')
% disp('Basis Pursuit algorithm (Ciril et al. (2023)) with (threshold) warm start')
% disp('Starting...')
% tic
% [sol_x_threshold_warm, ~] = BP_exact_algorithm(A,b,sol_bp_threshold(:,m),tol_bp,disp_output_bp);
% time_bp_threshold_warm = toc;
% disp('Done!')



%% Display timings
disp(' ')
disp('- Timings -')

disp(['Total time elasped for the exact BP algorithm: ',...
    num2str(time_bp), ' seconds.'])
disp(' ')
disp(['Total time elasped for the modified (silly) BPDN algorithm: ',...
    num2str(time_bp_silly), ' seconds.'])
disp(['Total time elasped for the exact BP algorithm with (silly) warm start up: ',...
    num2str(time_bp_silly_warm), ' seconds.'])
disp(' ')

% disp(['Total time elasped for the modified (average) BPDN algorithm: ',...
%     num2str(time_bp_average), ' seconds.'])
% disp(['Total time elasped for the exact BP algorithm with (average) warm start up: ',...
%     num2str(time_bp_average_warm), ' seconds.'])
% disp('------------')
% disp(' ')
% 
% disp(['Total time elasped for the modified (threshold) BPDN algorithm: ',...
%     num2str(time_bp_threshold), ' seconds.'])
% disp(['Total time elasped for the exact BP algorithm with (threshold) warm start up: ',...
%     num2str(time_bp_threshold_warm), ' seconds.'])
% disp('------------')
% disp(' ')



%% Print out optimality results
% Display optimality conditions for exact Basis Pursuit problem
disp('- Optimality conditions -')
disp('Optimality conditions for exact Basis Pursuit with cold start -b/norminf(Atop*b)')
disp(['Euclidan norm of Ax-b: ', num2str(norm(A*sol_x-b))])
disp(['l1 Norm of the bp solution: ', num2str(norm(sol_x,1))])
disp(['-<sol_p,b>: ', num2str(-sol_p.'*b)])
disp('----------')
disp(' ')


% Display optimality conditions for modified Basis Pursuit
disp(['Euclidean norm of sol_x_silly(:,end): ', num2str(norm(sol_x_silly(:,end)))])
disp(['Euclidan norm of A*sol_x_silly - b_silly: ', num2str(norm(A*sol_x_silly(:,end) - b_silly(:,end)))])
disp(['Euclidan norm of A(x_silly)-b: ', num2str(norm(A*x_silly(:,end) - b))])
disp(' ')

disp(['Primal obj function at m: ', num2str( (0.5/sol_t_silly(m))*norm(A*sol_x_silly(:,m) - b_silly(:,m))^2 + norm(sol_x_silly(:,m),1))])
disp(['Dual obj function at m: ', num2str(-sol_p_silly(:,m).'*b_silly(:,m) - ...
    0.5*sol_t_silly(m)*norm(sol_p_silly(:,m))^2)])
disp(['l1 Norm of x_silly at m: ', num2str(norm(x_silly(:,m),1))])
disp(' ')

disp(['Primal obj function at end: ', num2str(norm(sol_x_silly(:,end),1))])
disp(['Dual obj function at end: ', num2str(-sol_p_silly(:,end).'*b_silly(:,end) - ...
    0.5*sol_t_silly(end)*norm(sol_p_silly(:,end))^2)])
disp(['l1 Norm of x_silly at end: ', num2str(norm(x_silly(:,end),1))])
disp(' ')

% disp([abs(A.'*sol_p_silly(:,m)), abs(A.'*sol_p_silly(:,end))])
% disp(' ')

disp(['residual of sol_x - (tmp_silly): ', num2str(norm(x_silly(:,end) - sol_x))])
disp('----------')
disp(' ')



% Display optimality conditions for modified (silly) Basis Pursuit --> Warm BP
disp(['Euclidan norm of Ax_average_warm-b: ', num2str(norm(A*sol_x_silly_warm-b))])
disp(['l1 Norm of the bp_warm solution: ', num2str(norm(sol_x_silly_warm,1))])
disp(['residual of sol_x - sol_x_average_warm: ', num2str(norm(sol_x_silly_warm - sol_x))])
disp('----------')
disp(' ')

% % Display optimality conditions for modified (average) Basis Pursuit --> Warm BP
% disp(['Euclidan norm of Ax_average_warm-b: ', num2str(norm(A*sol_x_average_warm-b))])
% disp(['l1 Norm of the bp_warm solution: ', num2str(norm(sol_x_average_warm,1))])
% disp(['residual of sol_x - sol_x_average_warm: ', num2str(norm(sol_x_average_warm - sol_x))])
% disp('----------')
% disp(' ')
% 
% % Display optimality conditions for modified (threshold) Basis Pursuit --> Warm BP
% disp(['Euclidan norm of Ax_threshold_warm-b: ', num2str(norm(A*sol_x_threshold_warm-b))])
% disp(['l1 Norm of the bp_warm solution: ', num2str(norm(sol_x_threshold_warm,1))])
% disp(['residual of sol_x - sol_x_threshold_warm: ', num2str(norm(sol_x_threshold_warm - sol_x))])