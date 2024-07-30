%% Script for solving the Lasso problem.
% This method is the one where the time steps are allowed to vary.


%% Notes
% As of March 30th, the exact BP reconstruction is about one order of
% magnitude slower than the GLMNET algorithm.

% The bottleneck is actually the mex file!

% Check https://epubs.siam.org/doi/epdf/10.1137/S1064827502410633 
% for possible numerical instabilities (1-exp) term. Rescaling could
% work...


%% Data acquisition
% A = readmatrix('A_d=10.txt');
% b = readmatrix('b_d=10.txt');

A = readmatrix('A_p=8_time=7.598e-02.txt'); 
b = readmatrix('b_p=8_time=7.598e-02.txt');
[m,n] = size(A);
tol = 1e-08;


%% Initialization
% Regularization path
t = [1:-0.05:0.25,0.24:-0.01:0.05,0.045:-0.005:0.01,0];%,0.0095:-0.0005:0.0005,0.00045:-0.00005:0.0001];
tinf = norm(A.'*b,inf);
regpath = tinf*t;
l_path = length(t);

x_bp = zeros(n,l_path);
p_bp = zeros(m,l_path);
x_lpdhg = zeros(n,l_path);
p_lpdhg = zeros(m,l_path);

p_bp(:,1) = -b/tinf; 
p_lpdhg(:,1) = p_bp(:,1);


%% Solution via differential inclusion
disp(' ')
disp('Method 1: Using differential inclusions + 1st-order exponential integrators')

tic
tmp_vec = vecnorm(A,2);
for k=2:1:l_path
    % Variable selection
    lhs = abs(p_bp(:,k-1).'*A);
    rhs = 1-tmp_vec*norm(p_bp(:,k-1))*(regpath(k-1)-regpath(k))/regpath(k-1);
    selection_set = (lhs >= rhs);

    % Compute dual solution -- use differential inclusions
    [x_bp(selection_set,k),p_bp(:,k),~] = cutcells_dual_diff_incl(p_bp(:,k-1),A(:,selection_set),b,regpath(k),tol);
    disp(sum(x_bp(:,k)~=0))
end
time_bp_alg = toc;
disp('Complete!')


%% Method 2: GLMNET
disp(' ')
disp('Method 2: Using GLMNET')   
tic
opts = struct('lambda',regpath,'alpha',1,'intr',false,'standardize',false);
fit=glmnet(sqrt(m)*A,sqrt(m)*b,'gaussian',opts);
lambda_seq = fit.lambda;
coeffs = glmnetPredict(fit,sqrt(m)*A,lambda_seq(end),'coefficients');
coeffs(1) = [];
time_glmnet = toc;
disp('Complete!')


%% Compare quality of solutions between NNLS and exact BP reconstruction
disp(' ')
disp('Timings')
disp(['Direct approach with BP reconstruction and NNLS: ',num2str(time_bp_alg)])
disp(['GLMNET: ',num2str(time_glmnet)])

disp(' ')
disp('Sparsity')
nb_found_bp = sum(x_bp(:,end)~=0);
nb_found_glmnet = sum(coeffs~=0);
disp(['Nb of nonzero components with bp reconstruction: ',num2str(nb_found_bp)])
disp(['Nb of nonzero components with glmnet: ',num2str(nb_found_glmnet)])

disp('Final residual norm (non-squared)')
disp(['x_bp: ',num2str(norm(A*x_bp(:,end)-b))])
disp(['GLMNET: ',num2str(norm(A*coeffs-b))])

% DEBUG