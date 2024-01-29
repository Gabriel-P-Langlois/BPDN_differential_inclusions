%% Script for solving the Lasso problem with random data.
% Exact procedure as outlined in GPL's notes + Tendero et al. (2021)

%% NOTES
% disp(norm(A(:,set_active_new(1:n)).'*direction)) will be zero.
% So A_{\eps}^{\top}*direction = 0 identically.

% Update QR decomposition separately for matrices [A,B]?
% How about updating A.'*K on the fly too? Would that work?

% 05/29/2023
% Remove [A,-A]?

%% Data acquisition
%m = 1000; n = 25000;
%m = 500; n = 12500;
m = 250; n = 5000;
prop = 0.05;    % Proportion of coefficients that are equal to 10.
SNR = 5;

% Design matrix + Normalize it
rng('default')
A = randn(m,n);
A = A./sqrt(sum(A.^2)/m);

xsol = zeros(n,1);
xsol(randsample(n,n*prop)) = 10; 
ind_xsol = find(xsol);
sigma = norm(A*xsol)/sqrt(SNR);

b = (A*xsol + sqrt(sigma)*randn(m,1));
tol = 1e-8;
tol_minus = 1-tol;


%% Initialization for differential inclusion method
disp(' ')
disp('Method: Lasso via differential inclusions')
tic

% Vector term used in the algorithm and largest hyperparameter tinf
minusb = -b; 
minusb_t = minusb.';
mv_term1 = (minusb_t*[A,-A]).';
tinf = norm(mv_term1,inf); 
mv_term1 = mv_term1/tinf;

tmp1 = zeros(m,1);
mv_term2 = zeros(2*n,1);

% Active set
new_active_set = (mv_term1 >= tol_minus);
current_indices = find(new_active_set);
n_active_old = sum(new_active_set);

%% Active matrix
% Cannot use 'econ' for the rank-1 update
K = [A(:,new_active_set(1:n)),-A(:,new_active_set(n+1:end))];
[Q,R] = qr(K);
x_nnls = R(1:n_active_old,:)\((minusb_t*Q(:,1:n_active_old)).');

%[Q,R] = qr(K,'econ');
%x_nnls = R\((minusb_t*Q).');

%% Initial dual solution and hyperparameter
p = minusb/tinf;
t = tinf;

% Placeholders
direction = zeros(m);

% Dumb approach to storing the regularization path
max_iteration = 20001;
regpath = zeros(max_iteration,1);
regpath(1) = tinf;

% Parameters for the algorithms
shall_continue = true;
nb_iteration = 0;
max_iter = 1;


%% Solution via differential inclusion
while ((shall_continue && (nb_iteration < max_iter)))
    %%% 1. Compute direction
    tmp1 = K*x_nnls;
    direction = minusb - tmp1;

    % Compute S_Cˆ+(lambda) set
    mv_term2(1:n) = A.'*direction;
    mv_term2(n+1:end) = -mv_term2(1:n);
    active_set_plus = (mv_term2 > tol);
    
    % Verify if active_set_plus is empty. Stop if it is, else continue.
    if (any(active_set_plus))
        vec = (1-mv_term1)./mv_term2;
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
    test = (t*timestep^2)/(1+t*timestep);
    %p = p + timestep*direction; % This is NOT the dual solution?
    p = p + test*direction;
    t = t/(1+t*timestep); 

    nb_iteration=nb_iteration+1;        
    regpath(nb_iteration+1) = t;

    % DIAGNOSIS
    % tmp1 = -A*x
    disp(norm((-tmp1-b)/t))
    disp(norm(p))

    mv_term1 = mv_term1 + timestep*mv_term2;

    % Compute new active set and indices
    new_active_set = (mv_term1 >= tol_minus);
    new_indices = find(new_active_set);
    n_active_new = sum(new_active_set);

    % Update active matrix
    K = [A(:,new_active_set(1:n)),-A(:,new_active_set(n+1:end))]; % SLOW
    
    % Update QR matrix if only one active element has changed.
    if(n_active_new-n_active_old == 1)
        % Append a row
        [~,pos] = setdiff(new_indices,current_indices);
        
        R(:,pos+1:n_active_old+1) = R(:,pos:n_active_old);
        R(:,pos) = (K(:,pos).'*Q).';
        [Q,R] = matlab.internal.math.insertCol(Q,R,pos); % SLOW
        x_nnls = R(1:n_active_new,:)\((minusb_t*Q(:,1:n_active_new)).'); % SLOW
    else
        % More than one active vector is added or removed,
        % so perform the QR decomposition in full.
        [Q,R] = qr(K);
        x_nnls = R(1:n_active_new,1:n_active_new)\((minusb_t*Q(:,1:n_active_new)).');
    end

    % Update indices for the next iteration
    n_active_old = n_active_new;
    current_indices = new_indices;

    %%% 6. Print diagnostics
    %disp(['Number of active elements: ',num2str(sum(active_set))])
    %disp(t/tinf)
end

time_bp_alg = toc;
disp('Complete!')
disp(time_bp_alg)
tmp = zeros(2*n,1); tmp(new_active_set) = x_nnls;
x = -tmp(1:end/2) + tmp(end/2+1:end);
disp(['Residual norm (non-squared): ',num2str(norm(A*x-b))])


%% Recuperate the path...
regpath(nb_iteration+2:end) = [];


%% GLMNET
disp(' ')
disp('Method 2: Using GLMNET')   
tic
opts = struct('lambda',regpath,'alpha',1,'intr',false,'standardize',false);
fit=glmnet(sqrt(m)*A,sqrt(m)*b,'gaussian',opts);
coeffs = glmnetPredict(fit,sqrt(m)*A,regpath(end),'coefficients');
coeffs(1) = [];
time_glmnet = toc;
disp('Complete!')
disp(time_glmnet)

%% Diagnostic
disp(['Residual norm (non-squared) with differential inclusions: ',num2str(norm(A*x-b))])
disp(['Residual norm (non-squared) with GLMNET: ',num2str(norm(A*coeffs-b))])
