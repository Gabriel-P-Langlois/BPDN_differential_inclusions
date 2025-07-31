%% runall script
%  Part 1/5.


%% Initialization
tol_fista = 1e-8;


%% FISTA + Selection
x0 = zeros(n,1);
sol_fista_x = zeros(n,kmax); 
sol_fista_p = zeros(m,kmax);
min_iters = 100; max_iters = 50000;

% Compute the L22 norm of the matrix A and tau parameter for FISTA
disp('Running the fista algorithm...')
tic
L22 = svds(A,1)^2;
time_fista_L22 = toc;
vecA2 = vecnorm(A,2);

% Computer some parameters and options for FISTA
tau = 1/L22;

% Run the FISTA solver and rescale the dual solution
tic

% k == 1
[sol_fista_x(:,1),sol_fista_p(:,1),num_iters] = ...
        lasso_fista_solver(x0,p0,t(1),...
        A,b,tau,max_iters,tol_fista,min_iters);
sol_fista_p(:,1) = sol_fista_p(:,1)/t(1);

% 2 <= k <= kmax-1
for k=2:1:kmax
    % Selection rule
    Atimesp = A.'*sol_fista_p(:,k-1);
    ind = lasso_screening_rule(t(k)/t(k-1),sol_fista_p(:,k-1),vecA2,Atimesp);

    % Compute solution
    [sol_fista_x(ind,k),sol_fista_p(:,k),num_iters] = ...
        lasso_fista_solver(sol_fista_x(ind,k-1),sol_fista_p(:,k-1),t(k),...
        A(:,ind),b,tau,max_iters,tol_fista,min_iters);
    sol_fista_p(:,k) = sol_fista_p(:,k)/t(k);
end
time_fista_alg = toc;
time_fista_total = time_fista_alg + time_fista_L22;
disp(['Done. Total time = ', num2str(time_fista_total), ' seconds.'])
disp(' ')


%% Plot some results
nnz_fista = zeros(kmax,1);
rel_err_dual_fista_alg1 = zeros(kmax,1);
for k=1:1:kmax
    nnz_fista(k) = nnz(sol_fista_x(:,k));
    rel_err_dual_fista_alg1(k) = ...
                norm(sol_fista_p(:,k)-sol_incl_p(:,k),2)/...
                norm(sol_incl_p(:,k),2);
end

figure(f1)
semilogx(t,nnz_fista,'^','DisplayName','fista',...
        'MarkerFaceColor',RGB(3,:),'MarkerEdgeColor','k')

figure(f2)
loglog(t,rel_err_dual_fista_alg1,...
        '^','DisplayName','fista',...
        'MarkerFaceColor',RGB(3,:),'MarkerEdgeColor','k')
