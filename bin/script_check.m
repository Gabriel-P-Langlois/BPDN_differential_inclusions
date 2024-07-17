%%  Description
%   Written by Gabriel Provencher Langlois.


%%
% Update Jul 16: The solution computed by the homotopy method 
% pick different values of x at the time points where the NNLS
% returns a strictly positive solution.

% NOTE: Interesting, in the example with m=40 and n=50,
% the FISTA solution picks a zero components that is not picked up
% by the homotopy method.

% It seems the exact method is not picking up all entries for which -A.*'p
% is equal to +/- 1.

% CLUE: At iteration 25, the NNLS finds a zero component. So far,
% so good. It seems that the FISTA solution ``corrects" the anomaly and
% returns the previous nonzero coefficient to that zero component.

% Index of the 0 component of the NNLS solution at 
% iteration 25 *is* component 149, as suspected.

% TODO: Could we solve the least squares problem and interpolate only 
% UNTIL we reach the zero component, and then restart from there? That
% would be crazy, but maybe that's what we need to do? Ivnestigate!

%% Options for the script
display_iterations = false;
run_diagnostics = true;
use_fista = true;
use_glmnet = false;

% Options for the FISTA algorithm
if(use_fista)
    display_output_fista = true;
    tol_fista = 1e-10;
    min_iters_fista = 150;
end

% Tolerance for the homotopy algorithm
tol_exact = 1e-12;

% Random seed
rng('default')


%% Examples
% Synthetic data with noise
% Taken from ``Sparse Regression: Scalable Algorithms and Empirical
% Performance" by Dimitris Bertsimas, Jean Pauphilet and Bart Van Parys
%
% Fix n = 10000, use m = 500 or 2500 (this will vary)
% Matrix are columns from N(0,I)
% k_num = Number of nonzero coefficients
% SNR = 1 (medium noise)
% Values are all set to 1

m = 40; n = 200;        % Number of samples and features
k_num = floor(n/50);     % Number of true nonzero coefficients
val = 1;                % Value of nonzero coefficients
SNR = 1;                % Signal to noise ratio

% Design matrix + Normalize it
A = randn(m,n);
A = A./sqrt(sum(A.^2)/m);

xsol = zeros(n,1);
xsol(randsample(n,k_num)) = val; 
ind_nonzero_xsol = find(xsol);
ind_zero_xsol = find(~xsol);

sigma = norm(A*xsol)/sqrt(SNR);
b = (A*xsol + sqrt(sigma)*randn(m,1));


%% Homotopy algorithm
disp(' ')
disp('----------')
disp('The homotopy method (via gradient inclusions) for the Lasso.')

tic
[sol_exact_x,sol_exact_p,exact_path] = ...
    lasso_solver_homotopy(A,b,m,n,tol_exact,display_iterations,run_diagnostics);
time_exact_total = toc;

disp(['Total time elasped for the exact Lasso algorithm: ',...
    num2str(time_exact_total), ' seconds.'])
disp('----------')
disp(' ')

% Remove superfluous components of the path at the end
length_path = length(find(exact_path));
sol_path = exact_path(1:length_path); 
sol_exact_x(:,length_path+1:end) = [];
sol_exact_p(:,length_path+1:end) = [];

% Display a few details
disp(['Length of the path: ',num2str(length(sol_path))])
disp(length(sol_exact_x(1,:)))


%% FISTA algorithm
% Solve min_{x} 0.5*normsq{Ax-b} + t*normone(x)
if(use_fista)
    disp(' ')
    disp('----------')
    disp('The FISTA method \w improved screening rule.')

    % Define initial quantities and placeholders for solutions
    max_iters = 500000;
    sol_fista_x = zeros(n,length_path);     % Primal solutions
    sol_fista_p = zeros(m,length_path);     % Dual solutions 
    sol_fista_p(:,1) = -b;

    % Timings
    tic
    L22 = svds(A,1)^2;
    time_fista = toc;

    % Compute the solution at each kink found by the exact lasso method.
    for i=2:1:length_path-1
        % Improved screening rule
        ind = lasso_screening_rule(sol_path(i),sol_path(i-1),...
            sol_fista_x(:,i-1),sol_fista_p(:,i-1),A);

        % Display percentage of zero coefficients
        if(display_output_fista)
            disp(['Iteration ',num2str(i),'/',num2str(length_path-1),': Percentage of coefficients found to be zero: ',...
                num2str(100-100*sum(ind)/n)])
        end

        % Call the FISTA solver
        tau = 1/L22;
        tic 
        [sol_fista_x(ind,i),sol_fista_p(:,i),num_iters] = ...
            lasso_fista_solver(sol_fista_x(ind,i-1),sol_fista_p(:,i-1),...
            sol_path(i),A(:,ind),b,tau,max_iters,tol_fista,min_iters_fista);
        time_iter = toc;
        time_fista = time_fista + time_iter;

        % If enabled, the code below will print info about the iteration
        if(display_iterations)
            disp(['Solution computed for t/t0 = ', num2str(sol_path(i)/sol_path(1),'%.4e'), '. Number of primal-dual steps = ', num2str(num_iters), '.'])
            disp(['Time elapsed for solving the Lasso problem = ',...
                num2str(time_iter),' seconds.'])
            disp(' ')
        end
    end
    disp(['Total time elasped for constructing the regularization path with the FISTA algorithm: ',num2str(time_fista), ' seconds.'])
    disp(' ')
    disp('----------')
    disp(' ')

    % Rescale the dual solution
    for(i=1:1:length(sol_path)-1)
        sol_fista_p(:,i) = sol_fista_p(:,i)/sol_path(i);
    end
end


%% RUN GLMNET


%% Run diagnostics if enabled

% Further diagnostics if the FISTA method was used
if(use_fista && run_diagnostics)
    % Compute the MSE between homotopy sol - FISTA sol -- Primal 
    MSE_primal_homotopy_fista = 0;
    for i=2:1:length(sol_path)-1
        var = norm(sol_exact_x(:,i)-sol_fista_x(:,i))^2;
        MSE_primal_homotopy_fista = MSE_primal_homotopy_fista + var;
        disp(['Iteration: ',num2str(i),': (Norm of homotopy sol - FISTA sol)/norm(FISTA sol) (primal): ',...
            num2str(sqrt(var)/norm(sol_fista_x(:,i)))])
    end
    MSE_primal_homotopy_fista = MSE_primal_homotopy_fista/...
        (length(sol_path)-1);
    disp(' ')
    disp(['MSE between homotopy sol and FISTA sol (primal): ',...
        num2str(MSE_primal_homotopy_fista)])
    disp(' ')


    % Compute the MSE between homotopy sol - FISTA sol -- Dual
    MSE_dual_homotopy_fista = 0;
    for i=2:1:length(sol_path)-1
        var = norm(sol_exact_p(:,i)-sol_fista_p(:,i))^2;
        MSE_dual_homotopy_fista = MSE_dual_homotopy_fista + var;
        disp(['Iteration: ',num2str(i),': (Norm of homotopy sol - FISTA sol)/norm(FISTA sol) (dual): ',...
            num2str(sqrt(var)/norm(sol_fista_p(:,i)))])
    end
    MSE_dual_homotopy_fista = MSE_dual_homotopy_fista/(length(sol_path)-1);
    disp(' ')
    disp(['MSE between homotopy sol and FISTA sol (dual): ',...
        num2str(MSE_dual_homotopy_fista)])
    disp(' ')
end

disp('Index 25')
ind25_exact = abs(sol_exact_x(:,25)) > tol_exact; check25_exact = -A.'*sol_exact_p(:,25);
ind25_fista = abs(sol_fista_x(:,25)) > tol_fista; check25_fista = -A.'*sol_fista_p(:,25);

disp(sol_exact_x(abs(sol_exact_x(:,25)) > 0,25))
disp(check25_exact(ind25_exact))
disp(find(ind25_exact))

disp(sol_fista_x(abs(sol_fista_x(:,25)) > 0,25))
disp(check25_fista(ind25_fista))
disp(find(ind25_fista))

% Index 179
%disp(check25_exact(179))

disp(' ')
disp('Index 26')
disp(' ')

ind26_exact = abs(sol_exact_x(:,26)) > tol_exact; check26_exact = -A.'*sol_exact_p(:,26);
ind26_fista = abs(sol_fista_x(:,26)) > tol_fista; check26_fista = -A.'*sol_fista_p(:,26);

disp(sol_exact_x(abs(sol_exact_x(:,26)) > 0,26))
disp(check26_exact(ind26_exact))
disp(find(ind26_exact))

disp(sol_fista_x(abs(sol_fista_x(:,26)) > 0,26))
disp(check26_fista(ind26_fista))
disp(find(ind26_fista))

% Index 149
disp('Index 149')
disp(sol_exact_x(149,25))
disp(check25_exact(149))
disp(sol_exact_x(149,26))
disp(check26_exact(149))

disp(sol_fista_x(149,25))
disp(check25_fista(149))
disp(sol_fista_x(149,26))
disp(check26_fista(149))



% Refine the FISTA solution inbetween the difficult points
t1 = sol_path(25);
t2 = sol_path(26);
h = (t1-t2)/50;
tmp_t = t1:-h:t2;
tmp_vec_x = zeros(n,length(t1:-h:t2));
tmp_vec_x(:,1) = sol_fista_x(:,25);

tmp_vec_p = zeros(m,length(t1:-h:t2));
tmp_vec_p(:,1) = sol_fista_p(:,25);


for i=1:1:length(t1:-h:t2)-1
    [tmp_vec_x(:,i+1),tmp_vec_p(:,i+1),num_iters] = ...
            lasso_fista_solver(tmp_vec_x(:,i),tmp_vec_p(:,i),...
            tmp_t(i+1),A,b,tau,max_iters,tol_fista,min_iters_fista);
end

plot(sol_path,sol_exact_x(149,:))
plot(sol_path,sol_fista_x(149,:))
plot(tmp_t,tmp_vec_x(149,:),'-x')
legend('Test solution','Homotopy path','Lasso solution','Refined Lasso solution')



