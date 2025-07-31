%% runall script
%  Part 2/5.


%% Initialization
tol = 1e-08;
t0 = norm(A.'*b,inf);
p0 = -b/t0;


%% Run the differential inclusions algorithm
disp('Running the differential inclusions BPDN algorithm...')
tic
[sol_incl_x,sol_incl_p, bpdn_count, bpdn_linsolve] = ...
    BPDN_incl_regpath_s(A,b,p0,t,tol);
time_incl_alg = toc;
disp(['Done. Total time = ', num2str(time_incl_alg), ' seconds.'])
disp(' ')


%% Plot the number of nonzero components
nnz_alg1 = zeros(kmax,1);
for k=1:1:kmax
    nnz_alg1(k) = nnz(sol_incl_x(:,k));
end

hold(gca,'on')
semilogx(t,nnz_alg1,'o','DisplayName','differential inclusions',...
        'MarkerFaceColor',RGB(2,:),'MarkerEdgeColor','k')
shg


% % Figure
% f1 = figure(1);
% semilogx(t/t(1),nnz_glmnet,'square','DisplayName','glmnet',...
%         'MarkerFaceColor',RGB(1,:),'MarkerEdgeColor','k') 
% title('Number of nonzero components of the primal solution $x_{\mathrm{X}}^{s}(t,b)$',...
%         'interpreter','latex','fontsize',14)
% xlabel('t/$||A^{\top}b||_{\infty}$ (scaled to $[0,1]$).','interpreter','latex','fontsize',14)
% ylabel('Number of nonzero components','fontsize',14,'interpreter','latex')
% ylim('padded')
% lgd = legend('Location','best');
% fontsize(lgd,14,'points')
% grid on
% set(gca, 'xdir', 'reverse')
% set(gca,'xminorgrid','on','yminorgrid','off')