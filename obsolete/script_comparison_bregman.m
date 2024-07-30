%% Script for solving the Lasso problem using the matrices Andrew gave me
% This is Stan's Bregman method.


%% Data acquisition
% Example -- Synthetic data with noise
m = 2000; n = 10000;
prop = 0.02;    % Proportion of coefficients that are equal to 10.

% Design matrix + Normalize it
rng('default')
A = randn(m,n);
A = A./sqrt(sum(A.^2)/m);

xsol = zeros(n,1);
xsol(randsample(n,n*prop)) = 10; 
ind_xsol = find(xsol);
b = (A*xsol + randn(m,1));
tol = 1e-08;


%% Set up regularization path
num_sols = 8;
x_bp = zeros(n,num_sols+1);
p_bp = zeros(m,num_sols+1);
x_lpdhg = zeros(n,num_sols+1);
p_lpdhg = zeros(m,num_sols+1);

tinf = norm(A.'*b,inf);
p_bp(:,1) = -b/tinf; 
p_lpdhg(:,1) = p_bp(:,1);


%% Method 1: Exact BP reconstruction
disp(' ')
disp('Method 1: Using differential inclusions + 1st-order exponential integrators')

tic
tmp_vec = vecnorm(A,2);
for k=1:1:num_sols
    % Variable selection
    lhs = abs(p_bp(:,k).'*A);
    rhs = 1 - tmp_vec*norm(A*x_bp(:,k) - b)/sqrt(2*tinf);
    selection_set = (lhs >= rhs);
    disp(sum(selection_set))

    % Compute solutions using differential inclusions
    [x_bp(selection_set,k+1),p_bp(:,k+1)] = bregman_dual_diff_incl(p_bp(:,k),A(:,selection_set),b,tinf,tol);
    %[x_bp(:,k+1),p_bp(:,k+1)] = lasso_dual_diff_incl(p_bp(:,k),A,b,tinf,tol);
end
time_bp_alg = toc;
disp('Complete!')


% %% Method 2: nPDHG algorithm
% disp(' ')
% disp('Method 2: The linear PDHG method (regular sequence)')
% tic
% 
% for k=1:1:num_sols
%     L22sq = (svds(A(:,selection_set),1))^2;
%     theta = 0;
%     tau = tinf*0.5/L22sq;
%     sigma = 2.0/tinf;
%     
%     % Call the solver for this problem
%     [x_lpdhg(selection_set,k+1),p_lpdhg(:,k+1)] = ... 
%         lasso_lpdhg(x_lpdhg(selection_set,k),p_lpdhg(:,k),A(:,selection_set),b,tinf,tau,sigma);     
% end
% time_lpdhg_regular = toc;
% disp('Complete!')


%% Compare quality of solutions between NNLS and exact BP reconstruction
disp(' ')
disp('Timings')
disp(['Direct approach with BP reconstruction and NNLS: ',num2str(time_bp_alg)])

disp(' ')
disp(['l2 norm between final and penultimate dual solutions: ',num2str(norm(p_bp(:,end)-p_bp(:,end-1)))])

disp(' ')
disp('Sparsity')
nb_found_bp = sum(x_bp(:,end)~=0);
disp(['Nb of nonzero components: ',num2str(n*prop)])
disp(['Nb of nonzero components with bp reconstruction: ',num2str(nb_found_bp)])

disp(' ')
disp('Nb of true coefficients found')
nb_true_bp = length(intersect(find(x_bp(:,end)~=0),ind_xsol));
disp(['x_bp: ',num2str(nb_true_bp)])

disp(' ')
nb_false_bp = nb_found_bp-nb_true_bp;
disp('Nb of false discoveries')
disp(['x_bp: ',num2str(nb_false_bp)])





%% Helper function I: Dual solver via differential inclusion
function [x,p] = lasso_dual_diff_incl(p,A,b,t,tol)
n = size(A,2);
A = [A,-A];

% Placeholders and parameters
lambda = p; % Initial value

shall_continue = true;
tol_minus = 1-tol;
nb_iterations = 0;

while (shall_continue)
    %%% 1. Compute active set
    v_active = lambda.'*A;
    active_set = (v_active.' >= tol_minus);

    %%% 2. Compute direction
    K = A(:,active_set);
    [ME, NU] = size(K);

    direction = -b - t*(lambda-p);
    x_nnls = nnls_for_paths(K,direction,ME,NU);
    direction = direction - K*x_nnls;
    
    %%% 3. Compute kick time
    % Compute S_Cˆ+(lambda) set
    v_direction = A.'*(direction);
    active_set_plus = (v_direction > tol);
    asp_check = any(active_set_plus);

    % Verify if active_set_plus is empty. Stop if it is, else continue.
    if (asp_check)
        vec = (1-v_active)./v_direction.';
        timestep = min(vec(active_set_plus));
        % Safety check: t must be non negative
        if (timestep < 0) 
            timestep = 0;
        end
    else
        timestep = realmax;
    end

    %%% 4. Check termination condition
    if(timestep==realmax || timestep == 0)
        shall_continue = false;
    else
        lambda_plus = lambda + (1/t)*(1-exp(-t*timestep))*direction; 
        if (norm(lambda_plus-lambda) < tol)
            shall_continue = false;
        end
    lambda = lambda_plus;
    nb_iterations=nb_iterations+1;
    end
end

%%% 5. Return primal and dual solution
tmp = zeros(2*n,1); tmp(active_set) = x_nnls;
x = -tmp(1:end/2) + tmp(end/2+1:end);
p = lambda;
end

%% Helper function III: Lasso solver
function [x,p] = lasso_lpdhg(x,p,A,b,t,tau,sigma)

% Placeholders and parameters    
theta = 0;
xminus = x;

max_iter = 250000;
iter_now = 0;
tol = 1e-10;

pk = p;

for k=1:1:max_iter
    % Update the variables
    tmp = x + theta*(x-xminus);
    pplus = A*tmp - b + t*pk;
    pplus = (p + sigma*pplus)/(1+(t*sigma));

    xplus = x - tau*(A'*pplus);
    xplus = sign(xplus).*max(0,abs(xplus) - tau);

    % Check for convergence
    if(mod(k,50) == 0)
        error_norm_p = norm(pplus-p);
        if((error_norm_p < tol*norm(pplus)))
           break; 
        end
    end

    % Increment
    iter_now = iter_now + 1;
    xminus = x;
    x = xplus;
    p = pplus;
    theta = 1/sqrt(1 + t*sigma); tau = tau/theta;  sigma = theta*sigma;
end
end