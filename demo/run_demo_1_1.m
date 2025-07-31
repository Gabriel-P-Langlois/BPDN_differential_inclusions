%% runall script
%  Part 1/5.


%% Close current figures
close all


%% Load the data
load './../../LOCAL_DATA/l1_testset_data/spear_inst_474.mat'
[m,n] = size(A);
t0 = norm(A.'*b,inf);
t = t0 * [logspace(0,-4,500)];


%% Run glmnet with default options (except standardize).
warning('off');
options = glmnetSet;
options.alpha = 1;
options.intr = false;
options.standardize = false;
options.lambda = t;

% Run the external GLMNET package
disp('Running GLMNET for the BPDN problem with default tolerance...')
tic
fit = glmnet(sqrt(m)*A,sqrt(m)*b, 'gaussian', options);
kmax = length(t);

% Trim x solution and compute the dual solution
sol_glmnet_x = fit.beta;
sol_glmnet_p = zeros(m,kmax); 
for k=1:1:kmax
    sol_glmnet_p(:,k) = (A*sol_glmnet_x(:,k)-b)/(t(k));
end
time_glmnet_alg_1 = toc;
disp(['Done. Total time = ', num2str(time_glmnet_alg_1), ' seconds.'])
disp(' ')


%% Plot the number of nonzero components
nnz_glmnet = zeros(kmax,1);
for k=1:1:kmax
    nnz_glmnet(k) = nnz(sol_glmnet_x(:,k));
end

% Color palette
RGB = orderedcolors("dye");

% Figure
f1 = figure(1);
semilogx(t,nnz_glmnet,'square','DisplayName','glmnet (default tol = 1e-04)',...
        'MarkerFaceColor',RGB(1,:),'MarkerEdgeColor','k') 
title('Number of nonzero components of the solution $w_{\mathrm{X}}^{s}(t,y)$',...
        'interpreter','latex','fontsize',19)
xlabel('t','interpreter','latex','fontsize',19)
ylabel('Number of nonzero components','fontsize',19,'interpreter','latex')
ylim('padded')
lgd = legend('Location','best');
fontsize(lgd,19,'points')
grid on
set(gca, 'xdir', 'reverse')
set(gca,'xminorgrid','on','yminorgrid','off')