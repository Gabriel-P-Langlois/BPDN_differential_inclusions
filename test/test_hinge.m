%% runall script for testing the hinge algorithm
% This script tests Meyers's method of hinges and compare 
% its efficiency vs the MATLAB implementation



%% Input
% Nb of samples and features
m = 200;
n = 1000;


%% Generate data
A = randn(m,n); 
y = randn(n,1);
active_set = false(m,1);


%% Run the hinge algorithm
tic
[x,p_hinge] = hinge_algorithm(A, y, active_set);
time_hinge = toc;
disp(['Total time: ', num2str(time_hinge)])