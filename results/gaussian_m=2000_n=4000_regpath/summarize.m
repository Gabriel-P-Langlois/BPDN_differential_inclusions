%%  summarize script
%   This script summarizes the results from the runall.m file located for
%   the gaussian_m=2000_n=4000_regpath runall script.

if(summarize_2000_4000_regpath)
    % Placeholders for comparing the primal solutions
    norm_primal_glmnet_incl = zeros(kmax,1);
    norm_primal_fista_incl = zeros(kmax,1);
    norm_primal_fista_glmnet = zeros(kmax,1);

    % Placeholders for comparing the dual solutions
    norm_dual_glmnet_incl = zeros(kmax,1);
    norm_dual_fista_incl = zeros(kmax,1);
    norm_dual_fista_glmnet = zeros(kmax,1);

    % Placeholders for comparing the primal objective functions
    primal_obj_glmnet = zeros(kmax,1);
    primal_obj_incl = zeros(kmax,1);
    norm_primal_obj_glmnet_incl = zeros(kmax,1);

    % Placeholders for comparing the dual objective functions
    dual_obj_glmnet = zeros(kmax,1);
    dual_obj_incl = zeros(kmax,1);
    norm_dual_obj_glmnet_incl = zeros(kmax,1);

    for k=1:1:kmax
        % Primal objective function
        if(k ~= kmax)
            primal_obj_glmnet(k) = (0.5/t(k))*norm(A*sol_glmnet_x(:,k)-b,2)^2 + ...
                norm(sol_glmnet_x(:,k),1);
            primal_obj_incl(k) = (0.5/t(k))*norm(A*sol_incl_x(:,k) - b,2)^2 + ...
                norm(sol_incl_x(:,k),1);
        else
            primal_obj_glmnet(k) = norm(sol_glmnet_x(:,k),1);
            primal_obj_incl(k) = norm(sol_incl_x(:,k),1);
        end
        norm_primal_obj_glmnet_incl(k) = ...
            (primal_obj_incl(k) - primal_obj_glmnet(k))/primal_obj_incl(k);

        % Dual objective function
        dual_obj_glmnet(k) = 0.5*t(k)*norm(sol_glmnet_p(:,k),2)^2 + ...
            sol_glmnet_p(:,k).'*b;
        dual_obj_incl(k) = 0.5*t(k)*norm(sol_incl_p(:,k),2)^2 + ...
            sol_incl_p(:,k).'*b;
        norm_dual_obj_glmnet_incl(k) = ...
            (dual_obj_incl(k) - dual_obj_glmnet(k))/dual_obj_incl(k);

        % Dual and primal solutions
        norm_dual_glmnet_incl(k) = norm(sol_glmnet_p(:,k)-sol_incl_p(:,k),inf)/...
            norm(sol_incl_p(:,k),inf);
        norm_primal_glmnet_incl(k) = norm(sol_glmnet_x(:,k)-sol_incl_x(:,k),inf)/...
            norm(sol_incl_x(:,k),inf);
    end


    if(use_fista)
        for k=1:1:kmax
            norm_dual_fista_incl(k) = norm(sol_fista_p(:,k)-sol_incl_p(:,k),inf);
            norm_dual_fista_glmnet(k) = norm(sol_fista_p(:,k)-sol_glmnet_p(:,k),inf);
        end
    end

    % Primal vs dual for the BPDN method
    primal_dual_incl_difference = primal_obj_incl + dual_obj_incl;
    disp('primal_obj_incl; dual_obj_incl; primal_dual_obj_diff')
    disp([primal_obj_incl, -dual_obj_incl, primal_dual_incl_difference])

    % Diff inclusions vs GLMNET: Primal objective function
    disp('primal_obj_incl; primal_obj_glmnet; incl-glmnet')
    disp([primal_obj_incl, primal_obj_glmnet, norm_primal_obj_glmnet_incl])

    % Diff inclusions vs GLMNET: Dual objective function
    disp('dual_obj_incl; dual_obj_glmnet; incl-glmnet')
    disp([dual_obj_incl, dual_obj_glmnet, norm_dual_obj_glmnet_incl])

    disp('Rel. error in linf norm of the differences between primal solutions (glmnet/incl first).')
    disp([norm_dual_glmnet_incl, norm_dual_fista_incl,norm_dual_fista_glmnet])

    disp('Rel. error in linf norm of the differences between dual solutions (glmnet/incl first).')
    disp([norm_dual_glmnet_incl, norm_dual_fista_incl,norm_dual_fista_glmnet])
end