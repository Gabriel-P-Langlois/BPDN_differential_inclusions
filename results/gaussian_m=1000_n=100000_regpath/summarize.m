%%  summarize script
%   This script summarizes the results from the runall.m file located for
%   the gaussian_m=1000_n=5000_regpath runall script.

if(summarize_1000_5000_regpath)
    norm_primal_glmnet_incl = zeros(kmax,1);
    norm_primal_fista_incl = zeros(kmax,1);
    norm_primal_fista_glmnet = zeros(kmax,1);

    norm_dual_glmnet_incl = zeros(kmax,1);
    norm_dual_fista_incl = zeros(kmax,1);
    norm_dual_fista_glmnet = zeros(kmax,1);

    dual_obj_glmnet = zeros(kmax,1);
    dual_obj_incl = zeros(kmax,1);
    norm_dual_obj_glmnet_incl = zeros(kmax,1);

    for k=1:1:kmax
        dual_obj_glmnet(k) = 0.5*t(k)*norm(sol_glmnet_p(:,k),2)^2 + ...
            sol_glmnet_p(:,k).'*b;
        dual_obj_incl(k) = 0.5*t(k)*norm(sol_incl_p(:,k),2)^2 + ...
            sol_incl_p(:,k).'*b;
        norm_dual_obj_glmnet_incl(k) = dual_obj_incl(k) - dual_obj_glmnet(k);

        norm_dual_glmnet_incl(k) = norm(sol_glmnet_p(:,k)-sol_incl_p(:,k),inf)/...
            norm(sol_glmnet_p(:,k),inf);
        norm_primal_glmnet_incl(k) = norm(sol_glmnet_x(:,k)-sol_incl_x(:,k),inf)/...
            norm(sol_glmnet_x(:,k),inf);
    end


    if(use_fista)
        for k=1:1:kmax
            norm_dual_fista_incl(k) = norm(sol_fista_p(:,k)-sol_incl_p(:,k),inf);
            norm_dual_fista_glmnet(k) = norm(sol_fista_p(:,k)-sol_glmnet_p(:,k),inf);
        end
    end

    disp('dual_obj_incl; dual_obj_glmnet; incl-glmnet')
    disp([dual_obj_incl, dual_obj_glmnet, norm_dual_obj_glmnet_incl])

    disp('Rel. error in linf norm of the differences between primal solutions (glmnet/incl first).')
    disp([norm_dual_glmnet_incl, norm_dual_fista_incl,norm_dual_fista_glmnet])

    disp('Rel. error in linf norm of the differences between dual solutions (glmnet/incl first).')
    disp([norm_dual_glmnet_incl, norm_dual_fista_incl,norm_dual_fista_glmnet])
end