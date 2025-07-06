%%  run_timings_incl script
%   This script runs FISTA + selection rule on some l1_testset data 
%   taken from https://wwwopt.mathematik.tu-darmstadt.de/spear/
%   which was used in the following paper:
%   "Solving basis pursuit: Heuristic optimality check 
%   and solver comparison" by Lorenz, Dirk A and Pfetsch, Marc E 
%   and Tillmann, Andreas M.


%% Notes: 
% This script takes about Xs


%% Initialization
% Tolerance levels
tol_fista = 1e-08;

inst_dense = [147,148,274,421,422,548];
inst_sparse = [199,200,473,474];
inst = [inst_dense,inst_sparse];
repeat = 1;
str = './../../../LOCAL_DATA/l1_testset_data/spear_inst_';

% Run the timings for FISTA + Selection rule
disp('Running timings and counts for each instance...')
for i=1:1:length(inst)
    data_file = strcat(str, num2str(inst(i)), '.mat');
    load(data_file)
    [m,n] = size(A);
        
    % Calculate the smallest hyperparameter for which we have the trivial
    % primal solution (zero) and dual solution
    t0 = norm(A.'*b,inf);
    x0 = zeros(n,1);
    p0 = full(-b/t0);
    data_is_sparse = issparse(A);
        
    % Generate desired grid of hyperparameters
    if(data_is_sparse)
        t = t0 * [logspace(0,-4,m/8),0.0];
    else
        t = t0 * [logspace(0,-4,m/2),0.0];
    end
    kmax = length(t);

    % Invoke FISTA
    tic
    L22 = svds(A,1)^2;
    tau = 1/L22;
    time_fista = toc;
    vecA2 = vecnorm(A,2);

    for j=1:1:repeat
        sol_fista_x = zeros(n,kmax);
        sol_fista_p = zeros(m,kmax);
        max_iters = 50000;
        min_iters = 100;
    
        % Run the FISTA solver and rescale the dual solution
        tic

        % k == 1
        [sol_fista_x(:,1),sol_fista_p(:,1),num_iters] = ...
                lasso_fista_solver(x0,p0,t(1),...
                A,b,tau,max_iters,tol_fista,min_iters);
        sol_fista_p(:,1) = sol_fista_p(:,1)/t(1);
    
        % 2 <= k <= kmax-1 (t = 0 at kmax)
        for k=2:1:kmax-1
            % Selection rule
            Atimesp = A.'*sol_fista_p(:,k-1);
            ind = lasso_screening_rule(t(k)/t(k-1),sol_fista_p(:,k-1),vecA2,Atimesp);
            
            % Compute solution
            [sol_fista_x(ind,k),sol_fista_p(:,k),num_iters] = ...
                lasso_fista_solver(sol_fista_x(ind,k-1),sol_fista_p(:,k-1),t(k),...
                A(:,ind),b,tau,max_iters,tol_fista,min_iters);
            sol_fista_p(:,k) = sol_fista_p(:,k)/t(k);
        end
        time_fista = time_fista + toc;
    end

    % Display time averages
    time_fista_avg = time_fista/repeat;

    disp(' ')
    disp(['Average timings over ', num2str(repeat), ...
        ' run(s) and counts for Instance ', num2str(inst(i))])
    disp(['FISTA               ', num2str(time_fista_avg),'s.'])
end