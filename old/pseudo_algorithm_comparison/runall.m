%% RUNALL script


%% Parameters
% Dimensions
m = 100; n = 500;

% Options for the exact BPDN algorithm
tol_exact = 1e-10;
disp_output_exact = true;
run_opt_cond_checks = true;

% Options for the exact BP algorithm
use_bp = true;
tol_bp = 1e-08;
disp_output_bp = false;

% Options for the FISTA algorithm
use_pseudo = true;
tol_pseudo = 1e-10;
disp_output_pseudo = true;

% Turn on the diagnostics
% This will compare pointwise the BPDN solutions from the methods above
% See the output.
run_diagnostics = true;

% Run the script
script_pseudo_example(m,n,tol_exact,disp_output_exact,...
    run_opt_cond_checks,use_bp,tol_bp,disp_output_bp,...
    use_pseudo,tol_pseudo,disp_output_pseudo,run_diagnostics)