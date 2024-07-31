function [sol_x_exact] = script_pathological_mairal(A,b,tol_exact,disp_output_exact,...
    run_opt_cond_checks,use_bp,tol_bp,disp_output_bp,run_diagnostics)
%%  Script for the work on the BPDN solution via Differential Inclusions
%   This script is part of the BPDN project. It performs
%   several numerical experiments to validate a new approach
%   to Basis Pursuit Denoising.
%
%   NOTE: This script uses a pathological example constructed by
%   Mairal, Julien, and Bin Yu. 
%   "Complexity analysis of the lasso regularization path." 
%   arXiv preprint arXiv:1205.0079 (2012).
%
%   This script is called from an appropriate runall.m function.
%
%   Written by Gabriel Provencher Langlois.
%
% Input (TBC.):
% m:
% n:
% tol_exact:
% disp_output_exact:
% run_opt_cond_checks:
% use_bp:
% tol_bp:
% disp_output_bp:
% use_fista:
% tol_fista:
% disp_output_fista:
% min_iters_fista:
% use_glmnet:
% tol_glmnet:
% run_diagnostics:



%% Example
% Synthetic data from Julien Mairal and Yu Bin.
%   Mairal, Julien, and Bin Yu. 
%   "Complexity analysis of the lasso regularization path." 
%   arXiv preprint arXiv:1205.0079 (2012).
%
%   The matrices A and b are loaded here.

% Load the data
disp(['Condition number of the matrix A: ',num2str(cond(A))])


%% Homotopy algorithm
tic
disp(' ')
disp('----------')
disp('Algorithm: Exact solution to BPDN via gradient inclusions')

% Call the solver
[m,n] = size(A);
max_iter = 30000*m;
[sol_x_exact,sol_p_exact,sol_path] = ...
    BPDN_exact_algorithm(A,b,tol_exact,max_iter,disp_output_exact,run_opt_cond_checks);
time_total_exact = toc;

% Calculate the length of the path
length_path = length(sol_path);

% Display some information
disp(['Total time elasped for calculating the solution path to BPDN via gradient inclusions: ',...
    num2str(time_total_exact), ' seconds.'])
disp('----------')
disp(' ')
disp(['Length of the path: ',num2str(length(sol_path))])
disp(['Basis Pursuit solution (residual -- Exact solution to BPDN via gradient inclusions): ||A*sol_x_exact(:,end)-b||_2 = ',num2str(norm(A*sol_x_exact(:,end)-b))])
disp(['Basis Pursuit solution (obj function -- Exact solution to BPDN via gradient inclusions): ||sol_x_exact(:,end)||_{1} = ',num2str(norm(sol_x_exact(:,end),1))])



%% Basis Pursuit algorithm
% This solves the problem
%   min_{x \in \Rn} \norm{x}_{1}   s.t. Ax = b
% and outputs both the primal and dual solutions.
if(use_bp)
    disp(' ')
    disp('----------')
    disp('Basis Pursuit algorithm')
    
    % Call the BP solver
    tic
    [sol_x_bp, sol_p_bp] = BP_exact_algorithm(A,b,tol_bp,disp_output_bp);
    time_bp = toc;

    % Display some information
    disp(['Total time elasped for the exact Lasso algorithm: ',...
        num2str(time_bp), ' seconds.'])
    disp('----------')
    disp(' ')
end



%% Diagnostics -- These are run if the variable run_diagnostics is == true
% Diagnostics involving the Basis Pursuit Algorithm
if(run_diagnostics && use_bp)
    disp(' ')
    disp('--------------------')
    disp('Diagnostics involving the Basis Pursuit Algorithm from Ciril et al. (2021)')
    disp('--------------------')
    disp(' ')

    % Solution obtained from the method from Ciril et al. (2021)
    disp(['Basis Pursuit solution (residual -- method from Ciril et al. (2021)): ||A*sol_x_bp-b||_2 = ',num2str(norm(A*sol_x_bp(:,end) -b))])
    disp(['Basis Pursuit solution (obj function -- method from Ciril et al. (2021)): ||sol_x_bp||_{1} = ',num2str(norm(sol_x_bp,1))])
    disp(['\ell_{2} norm residual between the methods = ',num2str(norm(sol_x_bp-sol_x_exact(:,end)))])
end
end