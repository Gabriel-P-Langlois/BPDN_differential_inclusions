%%  summarize script
%   This script summarizes the results from the run_dense_example.m file 

if(summarize_1_sparse)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 0. Perform calculations
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(~summarize_1_calculations)
        objd_homotopy = dual_obj_fun(sol_hBPDN_p,t,b);
        objd_alg1 = dual_obj_fun(sol_incl_p,t,b);
        objd_glmnet = dual_obj_fun(sol_glmnet_p,t,b);
        objd_fista = dual_obj_fun(sol_fista_p,t,b);

        % Relative error of the primal objective function w.r.t. alg1
        rel_err_objd_homotopy_alg1 = ...
            abs(objd_homotopy-objd_alg1)./abs(objd_alg1);
        rel_err_objd_glmnet_alg1 = ...
            abs(objd_glmnet-objd_alg1)./abs(objd_alg1);
        rel_err_objd_fista_alg1 = ...
            abs(objd_fista-objd_alg1)./abs(objd_alg1);

    
        % Relative error of the norm of the dual solution w.r.t. homotopy
        rel_err_dual_glmnet_alg1 = zeros(kmax,1);
        rel_err_dual_homotopy_alg1 = zeros(kmax,1);
        rel_err_dual_fista_alg1 = zeros(kmax,1);
        for k=1:1:kmax
            rel_err_dual_glmnet_alg1(k) = ...
                norm(sol_glmnet_p(:,k)-sol_incl_p(:,k),2)/...
                norm(sol_incl_p(:,k),2);
            rel_err_dual_homotopy_alg1(k) = ...
                norm(sol_incl_p(:,k)-sol_hBPDN_p(:,k),2)/...
                norm(sol_incl_p(:,k),2);
            rel_err_dual_fista_alg1(k) = ...
                norm(sol_fista_p(:,k)-sol_incl_p(:,k),2)/...
                norm(sol_incl_p(:,k),2);
        end
    
        % Count nonzero components of primal solutions obtained by
        % Algorithm 1 and the glmnet algorithm
        nnz_alg1 = zeros(kmax,1);
        nnz_glmnet = zeros(kmax,1);
        nnz_fista = zeros(kmax,1);

        for k=1:1:kmax
            nnz_alg1(k) = nnz(sol_incl_x(:,k));
            nnz_glmnet(k) = nnz(sol_glmnet_x(:,k));
            nnz_fista(k) = nnz(sol_fista_x(:,k));
        end
        % These methods don't work at t=0, so set their output to 0.
        nnz_glmnet(kmax) = 0;
        nnz_fista(kmax) = 0;
    
        % Optimality checks
        check1_incl = zeros(kmax,1); check2_incl = zeros(kmax,1);
        check2_glmnet = zeros(kmax,1);
        check1_fista = zeros(kmax,1); check2_fista = zeros(kmax,1);
        for k=1:1:kmax
            check1_incl(k) = norm(t(k).*sol_incl_p(:,k) - ...
                        (A*sol_incl_x(:,k) - b),inf);
            check1_fista(k) = norm(t(k).*sol_fista_p(:,k) - ...
                    (A*sol_fista_x(:,k) - b),inf);

            check2_incl(k) = norm(-A.'*sol_incl_p(:,k),inf);
            check2_glmnet(k) = norm(-A.'*sol_glmnet_p(:,k),inf);
            check2_fista(k) = norm(-A.'*sol_fista_p(:,k),inf);
        end
        % Color palette
        RGB = orderedcolors("gem");

        % Avoid recalculating the second this function is called.
        summarize_1_calculations = true;
    end


    % %%%%%%%%%%
    % % 1. Plot the dual function V(p^{s}(t,b)) as a function of t
    % %%%%%%%%%%
    f1 = figure(1);
    loglog(t/t0,rel_err_objd_homotopy_alg1,'o',...
        'DisplayName','Algorithm 2 (homotopy)',...
        'MarkerFaceColor',RGB(2,:),'MarkerEdgeColor','k')
    hold(gca,'on')
    loglog(t(1:end-1)/t0,rel_err_objd_glmnet_alg1(1:end-1),...
        'square','DisplayName','glmnet',...
        'MarkerFaceColor',RGB(1,:),'MarkerEdgeColor','k')     
    loglog(t(1:end-1)/t0,rel_err_objd_fista_alg1(1:end-1),...
        '^','DisplayName','fista',...
        'MarkerFaceColor',RGB(3,:),'MarkerEdgeColor','k')

    title('Relative error of the dual objective function w.r.t. Algorithm 1',...
        'interpreter','latex','fontsize',14)
    xlabel('t/$||A^{\top}b||_{\infty}$','interpreter','latex','fontsize',14)
    ylabel(['$(V(p_{\mathrm{X}}^{s})(t,b) - V(p_{\mathrm{alg1}}^{s}(t,b))' ...
        '/V(p_{\mathrm{alg1}}^{s}(t,b))$'],'interpreter','latex','fontsize',14)
    xlim([10^-3 1])
    ylim([10^-19 10^0])
    lgd = legend('Location','southeast');
    fontsize(lgd,14,'points')
    grid on
    set(gca, 'xdir', 'reverse')
    set(gca,'xminorgrid','on','yminorgrid','off')    

    % Check if (V(p_{alg1}) - V(p_{X}))/V(p_{X}) < 0.
    % Note: This is a measure of the optimality of the solution obtained
    % from algorithm 1. We want something negative.
    disp(' ')
    disp('min(relative error w.r.t. V(p_{primal}^{s}(t,b)))')
    disp(['X = Algorithm 2 (homotopy): ', ...
        num2str(min(rel_err_objd_glmnet_alg1(1:end-1)))])
    disp(['X = glmnet: ', ...
        num2str(min(rel_err_objd_homotopy_alg1(1:end-1)))])
    disp(['X = fista: ', ...
        num2str(min(rel_err_objd_fista_alg1(1:end-1)))])
    disp(' ')
    
    %%%%%%%%%%
    % 2. Plot the relative error between dual solutions
    %%%%%%%%%%
    f2 = figure(2);
    loglog(t(1:end-1)/t0,rel_err_dual_homotopy_alg1(1:end-1),...
        'o','DisplayName','Algorithm 2 (homotopy)',...
        'MarkerFaceColor',RGB(2,:),'MarkerEdgeColor','k')
    hold(gca,'on')
    loglog(t(1:end-1)/t0,rel_err_dual_glmnet_alg1(1:end-1),...
        'square','DisplayName','glmnet',...
        'MarkerFaceColor',RGB(1,:),'MarkerEdgeColor','k')
    loglog(t(1:end-1)/t0,rel_err_dual_fista_alg1(1:end-1),...
        '^','DisplayName','fista',...
        'MarkerFaceColor',RGB(3,:),'MarkerEdgeColor','k')

    % Customize the figure
    title('Relative error of the dual solution obtained from Algorithm 1 vs X', ...
        'interpreter','latex','fontsize', 14)
    xlabel('t/$||A^{\top}b||_{\infty}$','interpreter','latex','fontsize',14)
    ylabel('$||p^{s}_{\mathrm{X}}(t,b) - p^{s}_{\mathrm{alg1}}(t,b)||_{\infty}/||p^{s}_{\mathrm{alg1}}(t,b)||_{\infty}$',...
        'interpreter','latex','fontsize',14)
    xlim([10^-3 1])
    ylim([10^-18 10^0])
    lgd = legend('Location','southeast');
    fontsize(lgd,14,'points')
    grid on
    set(gca, 'xdir', 'reverse')
    set(gca,'xminorgrid','on','yminorgrid','off')


    %%%%%%%%%%
    % 3. Plot number of nonzero components
    %%%%%%%%%%
    f3 = figure(3);
    semilogx(t/t0,nnz_alg1,'o','DisplayName','Algorithm 2 (homotopy)',...
        'MarkerFaceColor',RGB(2,:),'MarkerEdgeColor','k')
    hold(gca,'on')
    semilogx(t(1:end-1)/t0,nnz_glmnet(1:end-1),'square','DisplayName','glmnet',...
        'MarkerFaceColor',RGB(1,:),'MarkerEdgeColor','k') 
    semilogx(t/t0,nnz_fista,'^','DisplayName','fista',...
        'MarkerFaceColor',RGB(3,:),'MarkerEdgeColor','k')
    title('Number of nonzero components of the primal solution $x_{\mathrm{X}}^{s}(t,b)$',...
        'interpreter','latex','fontsize',14)
    xlabel('t/$||A^{\top}b||_{\infty}$','interpreter','latex','fontsize',14)
    ylabel('Number of nonzero components','fontsize',14,'interpreter','latex')
    xlim([10^-3 10^-1])
    ylim('padded')
    lgd = legend('Location','best');
    fontsize(lgd,14,'points')
    grid on
    set(gca, 'xdir', 'reverse')
    set(gca,'xminorgrid','on','yminorgrid','off')
    
    
    
    %%%%%%%%%%
    % 4. Check how close the optimality conditions are satisfied
    %%%%%%%%%%
    % First condition
    % NOTE: GLMNET and mlasso are ignored because they does not
    % provide the dual solutions directly.
    disp('Opt. cond. tp-Ax-b for incl/fista')
    disp(['    bpdn diff incl: ',num2str(max(check1_incl))])
    
    % Second condition
    disp(' ')
    disp('Opt. cond. (||-At*p||_{inf} - 1) = 0 for incl/glmnet/fista/mlasso')
    disp(['    bpdn diff incl: ', num2str(max(check2_incl)-1)])
    disp(['    glmnet: ', num2str(max(check2_glmnet(1:end-1))-1)])

    %%%%%%%%%%
    % 5 Save files
    %%%%%%%%%%
    exportgraphics(f1,"fig1_" + name + ".pdf")
    exportgraphics(f2,"fig2_" + name + ".pdf")
    exportgraphics(f3,"fig3_" + name + ".pdf")
end