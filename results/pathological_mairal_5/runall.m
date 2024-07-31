%% RUNALL script
% This runs the script_pathological_mairal example with m=n=5.

%% Parameters

% Options for the exact BPDN algorithm
tol_exact = 1e-10;
disp_output_exact = true;
run_opt_cond_checks = true;

% Options for the exact BP algorithm
use_bp = true;
tol_bp = 1e-08;
disp_output_bp = true;

% Turn on the diagnostics
% This will compare pointwise the BPDN solutions from the methods above
% See the output.
run_diagnostics = true;

% Display some information
load('../../data/mairal_pathological/A_pathological_5.mat', 'A') % saved as A
load('../../data/mairal_pathological/b_pathological_5.mat', 'b') % saved as b
disp('The pathological example of Mairal and Bin (2012) with m = n = 5.')

% Run the script
sol_x_exact = script_pathological_mairal(A,b,tol_exact,disp_output_exact,...
    run_opt_cond_checks,use_bp,tol_bp,disp_output_bp,run_diagnostics);

% Display some results
disp('Solution x obtained at the end')
disp(sol_x_exact(:,end))