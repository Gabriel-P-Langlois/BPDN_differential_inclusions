%% runall script
%  Part 4/5.


%% Run glmnet with harsh tolereance
warning('off');
options = glmnetSet;
options.alpha = 1;
options.intr = false;
options.standardize = false;
options.thresh = 1e-13;
options.lambda = t;

% Run the external GLMNET package
disp('Running GLMNET for the BPDN problem with harsh tolerance...')
tic
fit = glmnet(sqrt(m)*A,sqrt(m)*b, 'gaussian', options);

% Trim x solution and compute the dual solution
sol_glmnet_harsh_x = fit.beta;
sol_glmnet_harsh_p = zeros(m,kmax); 
for k=1:1:kmax
    sol_glmnet_harsh_p(:,k) = (A*sol_glmnet_harsh_x(:,k)-b)/(t(k));
end
time_glmnet_alg_1 = toc;
disp(['Done. Total time = ', num2str(time_glmnet_alg_1), ' seconds.'])
disp(' ')

%% Plot the number of nonzero components
nnz_glmnet_harsh = zeros(kmax,1);
for k=1:1:kmax
    nnz_glmnet_harsh(k) = nnz(sol_glmnet_harsh_x(:,k));
end

rel_err_dual_glmnet_harsh_alg1 = zeros(kmax,1);

for k=1:1:kmax
    rel_err_dual_glmnet_harsh_alg1(k) = ...
        norm(sol_glmnet_harsh_p(:,k)-sol_incl_p(:,k),2)/...
        norm(sol_incl_p(:,k),2);
end

% Figure
figure(f1)
semilogx(t,nnz_glmnet_harsh,'diamond','DisplayName','glmnet (harsh tolerance 1e-13)',...
        'MarkerFaceColor',RGB(4,:),'MarkerEdgeColor','k')

figure(f2)
    loglog(t,rel_err_dual_glmnet_harsh_alg1,...
        'diamond','DisplayName','glmnet (harsh tolerance 1e-13)',...
        'MarkerFaceColor',RGB(4,:),'MarkerEdgeColor','k')
    hold(gca,'on')