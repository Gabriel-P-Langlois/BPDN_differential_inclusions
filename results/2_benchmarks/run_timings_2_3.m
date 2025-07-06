%%  run_timings_incl script
%   This script runs the MATLAB's linprog solver on some l1_testset data 
%   taken from https://wwwopt.mathematik.tu-darmstadt.de/spear/
%   which was used in the following paper:
%   "Solving basis pursuit: Heuristic optimality check 
%   and solver comparison" by Lorenz, Dirk A and Pfetsch, Marc E 
%   and Tillmann, Andreas M.

%% NOTES
% Unused because of how slow it is.


%% Initialization
% Options for the timings
inst_dense = [147,148,274,421,422,548];
inst_sparse = [199,200,473,474];
inst = [inst_dense,inst_sparse];
repeat = 1;

str = './../../../LOCAL_DATA/l1_testset_data/spear_inst_';

% Run the timings
disp('Running timings and counts for each instance...')
for i=1:1:length(inst)
    time_linprog = 0;
    data_file = strcat(str, num2str(inst(i)), '.mat');
    load(data_file)
    [m,n] = size(A);
    data_is_sparse = issparse(A);

    % Calculate the smallest hyperparameter for which we have the trivial
    % primal solution (zero) and dual solution
    t0 = norm(A.'*b,inf);
    x0 = zeros(n,1);
    p0 = -b/t0;
        
    % Generate desired grid of hyperparameters
    if(data_is_sparse)
        t = t0 * [logspace(0,-4,m/8),0.0];
    else
        t = t0 * [logspace(0,-4,m/2),0.0];
    end
    kmax = length(t);clc

    for j=1:1:repeat
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % MATLAB's native lingprog solver for BP
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tic
        sol_linprog_p = linprog(b,[A,-A].',ones(2*n,1));
        if(~isempty(sol_linprog_p))
            ind_linprog = abs(-A.'*sol_linprog_p) > 1-tol;
            sol_linprog_x = zeros(n,1);
            sol_linprog_x(ind_linprog) = A(:,ind_linprog)\b;
        end
        time_linprog = time_linprog + toc;
        disp(' ')
    end

    % Display time averages
    time_lingprog_avg = time_linprog/repeat;

    disp(' ')
    disp(['Average timings over ', num2str(repeat), ...
        ' run(s) and counts for Instance ', num2str(inst(i))])
    disp(['linprog:               ', num2str(time_lingprog_avg),'s.'])

    % Other diagnostics
    disp(['l-infinity norm between true solution and sol_linprog: ', ...
        num2str(norm(x - sol_linprog_x(:),inf))]);
end