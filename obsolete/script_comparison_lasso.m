%% Script for solving the Lasso problem.
% This method is the one where the time steps are allowed to vary.


%% Notes
% Note: Push variable selection inside differential inclusion solver?
% If doing so, remember that for p =/= pk the rhs will be different.

% Fast version of the code is not robust...


%% Data acquisition
% Example -- Synthetic data with noise
m = 1000; n = 10000;
prop = 0.02;    % Proportion of coefficients that are equal to 10.

% Design matrix + Normalize it
rng('default')
A = randn(m,n);
A = A./sqrt(sum(A.^2)/m);

xsol = zeros(n,1);
xsol(randsample(n,n*prop)) = 10; 
ind_xsol = find(xsol);
b = (A*xsol + randn(m,1));

tol = 1e-8;

%% Initialization for differential inclusion method
% Regularization path
tinf = norm(A.'*b,inf);
regpath = tinf*[1,0.9712];
l_path = length(regpath);

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
% Compute vecnorm. This number is normalized, so...
col_norm = sqrt(m);
for k=2:1:l_path
    % Variable selection
    ratio = (regpath(k-1)-regpath(k))/regpath(k-1);
    selection_set = (abs(p_bp(:,k-1).'*A) >= 1-ratio*col_norm*norm(p_bp(:,k-1)));

    % Compute dual solution -- use differential inclusions
    [x_bp(selection_set,k),p_bp(:,k),~] = lasso_dual_diff_incl(p_bp(:,k-1),A(:,selection_set),b,regpath(k),tol);
end
time_bp_alg = toc;
disp('Complete!')



%% Method 2: GLMNET
disp(' ')
disp('Method 1: Using GLMNET')   
tic
opts = struct('lambda',regpath,'alpha',1,'intr',false,'standardize',false);
fit=glmnet(sqrt(m)*A,sqrt(m)*b,'gaussian',opts);
coeffs = glmnetPredict(fit,sqrt(m)*A,regpath(end),'coefficients');
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
disp(['Nb of nonzero components: ',num2str(n*prop)])
disp(['Nb of nonzero components with bp reconstruction: ',num2str(nb_found_bp)])
disp(['Nb of nonzero components with glmnet: ',num2str(nb_found_glmnet)])

disp(' ')
disp('Nb of true coefficients found')
nb_true_bp = length(intersect(find(x_bp(:,end)~=0),ind_xsol));
nb_true_glmnet = length(intersect(find(coeffs~=0),ind_xsol));
disp(['x_bp: ',num2str(nb_true_bp)])
disp(['glmnet: ',num2str(nb_true_glmnet)])

disp(' ')
nb_false_bp = nb_found_bp-nb_true_bp;
nb_false_glmnet = nb_found_glmnet-nb_true_glmnet;
disp('Nb of false discoveries')
disp(['x_bp: ',num2str(nb_false_bp)])
disp(['glmnet: ',num2str(nb_false_glmnet)])