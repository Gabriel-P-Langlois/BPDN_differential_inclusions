%% runall script
%  Part 3/5.

%% Plot the relative error between glmnet (default) vs diff inclusions
% objd_alg1 = dual_obj_fun(sol_incl_p,t,b);
% objd_glmnet = dual_obj_fun(sol_glmnet_p,t,b);
% 
% rel_err_objd_glmnet_alg1 = abs(objd_glmnet-objd_alg1)./abs(objd_alg1);
rel_err_dual_glmnet_alg1 = zeros(kmax,1);

for k=1:1:kmax
    rel_err_dual_glmnet_alg1(k) = ...
        norm(sol_glmnet_p(:,k)-sol_incl_p(:,k),2)/...
        norm(sol_incl_p(:,k),2);
end

f2 = figure(2);
    loglog(t,rel_err_dual_glmnet_alg1,...
        'square','DisplayName','glmnet',...
        'MarkerFaceColor',RGB(1,:),'MarkerEdgeColor','k')
    hold(gca,'on')
% Customize the figure
    title('Relative error of the dual solution obtained from differential inclusions vs glmnet', ...
        'interpreter','latex','fontsize', 14)
    xlabel('t','interpreter','latex','fontsize',18)
    ylabel('$||p^{s}_{\mathrm{X}}(t,y) - p^{s}_{\mathrm{alg1}}(t,y)||_{\infty}/||p^{s}_{\mathrm{alg1}}(t,y)||_{\infty}$',...
        'interpreter','latex','fontsize',18)
    lgd = legend('Location','best');
    fontsize(lgd,18,'points')
    grid on
    set(gca, 'xdir', 'reverse')
    set(gca,'xminorgrid','on','yminorgrid','off')