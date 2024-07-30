%% Script for solving the cutcells problem efficiently


%% Notes
% Cutcell data: matrices have their first row padded by ones,
% and b is strictly positive.


%% Load cutcell data
A = readmatrix('A_p=8_time=7.598e-02.txt');
b = readmatrix('b_p=8_time=7.598e-02.txt');

%A = readmatrix('A_p=4_time=2.136e-04.txt');
%b = readmatrix('b_p=4_time=2.136e-04.txt');

[m,n] = size(A);

% Select tolerance level
tol = 1e-10;
tol_minus = 1-tol;


%% Initialization for differential inclusion method
disp(' ')
disp('Method: Lasso via differential inclusions')
tic

% Augment matrix and take transpose
A = [A,-A];
At = A.';

% Compute -b, -b.', norminf(A.'*b) and (A.'*b).'/norminf(A.'*b).
minusb = -b;
minusb_t = minusb.';
mv_term1 = minusb_t*A;
tinf = norm(mv_term1,inf);
mv_term1 = mv_term1/tinf;

% Compute initial active set
v_active = mv_term1;
active_set = (v_active >= tol_minus);

% Compute initial matrix
K = A(:,active_set);
[Q,R] = qr(K,'econ');
x_nnls = R\((minusb_t*Q).');

% Compute initial time and dual solution
t = tinf;
p = minusb/t;

% Placeholders
direction = zeros(m);
tmp1 = zeros(m,1);
mv_term2 = zeros(2*n,1);

% Dumb approach to storing the regularization path
max_iteration = 20001;
regpath = zeros(max_iteration,1);
regpath(1) = tinf;

% Parameters for the algorithms
shall_continue = true;
nb_iteration = 0;


%% Solution via differential inclusion
while (shall_continue)
    %%% 1. Compute direction
    tmp1 = K*x_nnls;
    direction = minusb - tmp1;

    %%% 2. Compute kick time
    % Compute S_Cˆ+(lambda) set
    mv_term2 = A.'*direction;
    v_direction = mv_term2;
    active_set_plus = (v_direction > tol);
    
    % Verify if active_set_plus is empty. Stop if it is, else continue.
    if (any(active_set_plus))
        vec = (1-v_active)./v_direction.';
        timestep = min(vec(active_set_plus));
        % Safety check: t must be non-negative.
        if (timestep < 0) 
            t = 0; nb_iteration=nb_iteration+1;
            break;
        end
    else % Stop; termination condition satisfied.
        t = 0; nb_iteration=nb_iteration+1;
        break
    end

    %%% 3. Update variables and proceed to the next iteration
    % Update dual solution + time parameter
    p = p + timestep*direction;
    t = t/(1+t*timestep); 
    nb_iteration=nb_iteration+1;        
    regpath(nb_iteration+1) = t;

    % Compute active set
    mv_term1 = mv_term1 + timestep*mv_term2.';
    v_active = mv_term1;
    active_set = (v_active.' >= tol_minus);

    % Update active matrix
    K = A(:,active_set);

    % Compute QR
    [Q,R] = qr(K,'econ');
    x_nnls = R\((minusb_t*Q).');

    %%% 6. Print diagnostics
    %disp(['Number of active elements: ',num2str(sum(active_set))])
    %disp(t/tinf)
end

time_bp_alg = toc;
disp('Complete!')
disp(time_bp_alg)
tmp = zeros(2*n,1); tmp(active_set) = x_nnls;
x = -tmp(1:end/2) + tmp(end/2+1:end);



%A = readmatrix('A_p=8_time=7.598e-02.txt'); 
A = readmatrix('A_p=4_time=2.136e-04.txt');
disp(['Residual norm (non-squared): ',num2str(norm(A*x-b))])

%%% TESTING
new_active_set = (abs(-A.'*p) >= 1-tol);
[ME,NU] = size(A(:,new_active_set));
x_NNLS = nnls_cutcells(A(:,new_active_set),b,ME,NU);

% Why not try to implement Meyer's algorithm with efficient QR
% decomposition?


%% Recuperate the path...
regpath(nb_iteration+2:end) = [];


%% GLMNET
% disp(' ')
% disp('Method 2: Using GLMNET')   
% tic
% opts = struct('lambda',regpath,'alpha',1,'intr',false,'standardize',false);
% fit=glmnet(sqrt(m)*A,sqrt(m)*b,'gaussian',opts);
% coeffs = glmnetPredict(fit,sqrt(m)*A,regpath(end),'coefficients');
% coeffs(1) = [];
% time_glmnet = toc;
% disp('Complete!')
% disp(time_glmnet)
% 
% %% Diagnostic
% disp(['Residual norm (non-squared) with differential inclusions: ',num2str(norm(A*x-b))])
% disp(['Residual norm (non-squared): ',num2str(norm(A*coeffs-b))])