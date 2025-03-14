%%  summarize script
%   This script summarizes the results from the runall.m file located for
%   the gaussian_m=1000_n=2500_homotopy runall script.

if(summarize_1000_2500_homotopy)
    kmax = length(t);
    norm_primal_homotopy_incl = zeros(kmax,1);
    norm_dual_homotopy_incl = zeros(kmax,1);

    dual_obj_homotopy = zeros(kmax,1);
    dual_obj_incl = zeros(kmax,1);
    norm_dual_obj_homotopy_incl = zeros(kmax,1);

    for k=1:1:kmax
        dual_obj_homotopy(k) = 0.5*t(k)*norm(sol_homotopy_p(:,k),2)^2 + ...
            sol_homotopy_p(:,k).'*b;
        dual_obj_incl(k) = 0.5*t(k)*norm(sol_incl_p(:,k),2)^2 + ...
            sol_incl_p(:,k).'*b;
        norm_dual_obj_homotopy_incl(k) = dual_obj_incl(k) - dual_obj_homotopy(k);

        norm_dual_homotopy_incl(k) = norm(sol_homotopy_p(:,k)-sol_incl_p(:,k),inf)/...
            norm(sol_incl_p(:,k),inf);
        norm_primal_homotopy_incl(k) = norm(sol_homotopy_x(:,k)-sol_incl_x(:,k),inf)/...
            norm(sol_incl_x(:,k),inf);
    end

    disp('dual_obj_incl; dual_obj_glmnet; incl-homotopy')
    disp([dual_obj_incl, dual_obj_homotopy, norm_dual_obj_homotopy_incl])

    disp('Rel. error in linf norm of the differences between primal solutions (homotopy/incl first).')
    disp([norm_primal_homotopy_incl])

    disp('Rel. error in linf norm of the differences between dual solutions (homotopy/incl first).')
    disp([norm_dual_homotopy_incl])

end