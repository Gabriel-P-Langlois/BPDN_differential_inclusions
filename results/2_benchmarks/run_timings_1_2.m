%%  run_timings_incl script
%   This script runs the BP inclusion solver, the BPDN inclusion solver, 
%   GLMNET, MATLAB's lasso and the greedy solver on some l1_testset data 
%   taken from https://wwwopt.mathematik.tu-darmstadt.de/spear/
%   which was used in the following paper:
%   "Solving basis pursuit: Heuristic optimality check 
%   and solver comparison" by Lorenz, Dirk A and Pfetsch, Marc E 
%   and Tillmann, Andreas M.


%% Notes: 
% This script takes about 32s with 1 repeat to run on GPL's laptop.


%% Initialization
% Tolerance levels
tol = 1e-08;
tol_glmnet = tol;
tol_mlasso = 1e-06;

% Grid of hyperparameters
spacing = -0.005;
maxval = 0.995;
minval = 0.0;

inst_dense = [147,148,274,421,422,548];
inst_sparse = [173,174,447,448,199,200,473,474];
inst = [inst_dense,inst_sparse];
repeat = 1;
str = './../../../LOCAL_DATA/l1_testset_data/spear_inst_';

% Run the timings
disp('Running timings and counts for each instance...')
for i=1:1:length(inst)
    time_incl_BPDN = 0;
    time_incl_BP = 0;
    time_incl_warm = 0;
    time_glmnet = 0;

    data_file = strcat(str, num2str(inst(i)), '.mat');
    load(data_file)
    [m,n] = size(A);

    % Calculate the smallest hyperparameter for which we have the trivial
    % primal solution (zero) and dual solution
    t0 = norm(A.'*b,inf);
    x0 = zeros(n,1);
    p0 = -b/t0;
        
    % Generate desired grid of hyperparameters
    t = t0 * (maxval:spacing:minval);
    kmax = length(t);

    for j=1:1:repeat
        data_is_sparse = issparse(A);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Diff. Inclusions: BPDN
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        if(~data_is_sparse)
            [sol_incl_x,sol_incl_p, bpdn_nnls_count, bpdn_linsolve] = ...
                BPDN_incl_regpath(A,b,p0,t,tol);
        else
            [sol_incl_x,sol_incl_p, bpdn_nnls_count, bpdn_linsolve] = ...
                BPDN_incl_regpath_s(A,b,p0,t,tol);
        end
        time_incl_BPDN = time_incl_BPDN + toc;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Diff. Inclusions: Direct Basis Pursuit Solver
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        if(~data_is_sparse)
            [sol_incl_BP_x, sol_incl_BP_p, bp_nnls_count, bp_linsolve] = ...
                BP_incl_direct(A,b,p0,tol);
        else
            [sol_incl_BP_x, sol_incl_BP_p, bp_nnls_count, bp_linsolve] = ...
                BP_incl_direct_s(A,b,p0,tol);
        end
        time_incl_BP = time_incl_BP + toc;
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Greedy homotopy algorithm via the matroid property
        tic
        if(~data_is_sparse)
            [~,~, warm_nnls_count, warm_linsolve_count] = ...
                BP_incl_greedy(A,b,tol);
        else
            [~,~, warm_nnls_count, warm_linsolve_count] = ...
                BP_incl_greedy_s(A,b,tol);
        end
        time_incl_warm = toc;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % GLMNET
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        warning('off');
        options = glmnetSet;
        options.alpha = 1;
        options.lambda = t;
        options.intr = false;
        options.thresh = tol_glmnet;
        options.maxit = 10^8;
        options.standardize = false;
    
        % Fit GLMNET, trim the x solution and compute the dual solution
        tic
        fit = glmnet(sqrt(m)*A,sqrt(m)*b, 'gaussian', options);
        sol_glmnet_x = fit.beta;
        sol_glmnet_p = zeros(m,kmax); 
        for k=1:1:kmax
            ind = abs(sol_glmnet_x(:,k)) > 1e-05;
            sol_glmnet_x(~ind,k) = 0.0;
            sol_glmnet_p(:,k) = (A*sol_glmnet_x(:,k)-b)/t(k);
        end
        time_glmnet = time_glmnet + toc;
    end
    
    % Timings for MATLAB's lasso method -- on dense data only.
    time_mlasso = 0;
    if(i <= length(inst_dense))
        for j=1:1:repeat
            warning('off');
            % Run MATLAB's native lasso solver, flip it, 
            % and rescale the dual solution.
            tic
            sol_mlasso_x = lasso(sqrt(m)*A,sqrt(m)*b, 'lambda', t, ...
                'Intercept', false, 'RelTol', tol_mlasso);
            sol_mlasso_x = flip(sol_mlasso_x,2);
            sol_mlasso_p = zeros(m,kmax); 
            for k=1:1:kmax
                sol_mlasso_p(:,k) = (A*sol_mlasso_x(:,k)-b)/t(k);
            end
            time_mlasso = time_mlasso + toc;
        end
    end

    % Display time averages
    time_incl_BPDN_avg = time_incl_BPDN/repeat;
    time_incl_BP_avg = time_incl_BP/repeat;
    time_incl_warm_avg = time_incl_warm/repeat;
    time_glmnet_avg = time_glmnet/repeat;
    time_mlasso = time_mlasso/repeat;

    disp(' ')
    disp(['Average timings over ', num2str(repeat), ...
        ' run(s) and counts for Instance ', num2str(inst(i))])
    disp(['BPDN_incl timing:               ', num2str(time_incl_BPDN_avg),'s.'])
    disp(['BP_incl timing:                 ', num2str(time_incl_BP_avg),'s.'])
    disp(['Greedy + warm timing:           ',...
        num2str(time_incl_warm_avg),'s.'])
    disp(['glmnet timing:                  ',...
        num2str(time_glmnet_avg),'s.'])
    if(i <= length(inst_dense))
        disp(['mlasso timing:                  ',...
            num2str(time_mlasso),'s.'])
    end

    disp(' ')
    disp(['BPDN_incl NNLS calls:           ', num2str(bpdn_nnls_count)])
    disp(['BP_incl NNLS calls:             ', num2str(bp_nnls_count)])
    disp(['Greedy + warm NNLS calls:       ',...
        num2str(warm_nnls_count)])

    disp(' ')
    disp(['BPDN_incl linsolve calls:       ', num2str(bpdn_linsolve)])
    disp(['BP_incl linsolve calls:         ', num2str(bp_linsolve)])
    disp(['Greedy + warm linsolve calls:   ',...
        num2str(warm_linsolve_count)])

    disp(' ')
    disp(['l infinity norm between true solution and sol_incl_BP_x:', ...
        num2str(norm(x - sol_incl_BP_x(:),inf))]);
end