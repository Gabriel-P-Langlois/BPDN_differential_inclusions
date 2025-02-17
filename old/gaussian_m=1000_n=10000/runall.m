%% RUNALL script
% From the project directory, run
% run ./results/gaussian_m=1000_n=10000/runall.m

% NOTE: On GPL's laptop, it takes about one minute for the exact BPDN
% algorithm to converge, whereas GLMNET takes about an hour. See the
% accompanying .txt file in the folder. GLMNET will probably run faster
% with a lower tolerance (the default is 1e-04), but performance will
% likely be (even) worse.


%% Parameters
% Dimensions
m = 1000; n = 10000;

% Options for the exact BPDN algorithm
tol_exact = 1e-08;
disp_output_exact = false;
run_opt_cond_checks = false;

% Options for the exact BP algorithm
use_bp = false;
tol_bp = 1e-08;
disp_output_bp = false;

% Options for the FISTA algorithm
use_fista = false;
tol_fista = 1e-08;
disp_output_fista = false;
min_iters_fista = 200;  % Nb of iterations before first 
                        % verifying convergence. There needs to be a small
                        % threshold otherwise FISTA will not work well.

% Options for glmnet
use_glmnet = true;
tol_glmnet = 1e-08;     % Will change the relative tolerance of GLMNET

% Turn on the diagnostics
% This will compare pointwise the BPDN solutions from the methods above
% See the output.
run_diagnostics = true;

% Run the script
script_gaussian_example(m,n,tol_exact,disp_output_exact,...
    run_opt_cond_checks,use_bp,tol_bp,disp_output_bp,...
    use_fista,tol_fista,disp_output_fista,min_iters_fista,...
    use_glmnet,tol_glmnet,...
    run_diagnostics)