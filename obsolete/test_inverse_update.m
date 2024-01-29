%% Data acquisition
% Example 1 -- Synthetic data with noise
m = 400; n = 20000;      % Number of samples and features
val = 10;               % Value of nonzero coefficients
prop = 0.1;            % Proportion of coefficients that are equal to val.
SNR = 5;                % Signal to noise ratio

% Design matrix + Normalize it
rng('default')
A = randn(m,n);
A = A./sqrt(sum(A.^2)/m);

xsol = zeros(n,1);
xsol(randsample(n,n*prop)) = val; 
ind_xsol = find(xsol);

sigma = norm(A*xsol)/sqrt(SNR);
b = (A*xsol + sqrt(sigma)*randn(m,1));

%% Extract one column of the matrix A then solve the reduced system
ind = 3000;
K = A(:,ind);
M = 1/(K.'*K);

%% Update
ind2 = 2000;
v = A(:,ind2);

new_M = rank_one_LS_update(K,M,v,1);
K = A(:,[ind2,ind]);
new_x = new_M*(K.'*(-b));
test3 = K\(-b);