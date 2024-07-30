%% RUNALL script
% From the project directory, run
% run ./results/gaussian_m=100_n=500/runall.m


%% Parameters
% Dimensions
m = 100; n = 500;

% Options for the exact BPDN algorithm
tol_exact = 1e-08;
disp_output_exact = true;
run_opt_cond_checks = false;

% Options for the exact BP algorithm
use_bp = true;
tol_bp = 1e-08;
disp_output_bp = false;

% Options for the FISTA algorithm
use_fista = true;
tol_fista = 1e-08;
disp_output_fista = false;
min_iters_fista = 200;  % Nb of iterations before first 
                        % verifying convergence. There needs to be a small
                        % threshold otherwise FISTA will not work well.

% Options for glmnet
use_glmnet = true;
tol_glmnet = 1e-08;

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