%%  summarize script
if(summarize_greedy)
    % Compute the norms ||Ax - b||_{2} and l1 norm ||x||_{1} obtained from 
    % the BP inclusion algorithm and the greedy method.
    norm_residual_inclBP = norm(A*sol_inclBP_x-b);
    norm_l1_inclBP = norm(sol_inclBP_x,1);

    norm_residual_greedy_b = norm(A*sol_g_xf - b);
    norm_l1_feasible = norm(sol_g_xf,1);
    
    
    % Compute the l1 norm ||x||_{1} obtained from the QR decomposition
    % and Gaussian elimination, for good measure
    x_QR = A\b;
    [L,U,P] = lu(A); y = L\(P*b); x_LU = U\y;

    norm_l1_linsolve = norm(x_QR,1);
    norm_l1_LUdecomp = norm(x_LU,1);

    
    %%%%%%%%%%%%%%%%%%%%
    % Print the results
    %%%%%%%%%%%%%%%%%%%%
    disp('1. Norm of the residual Ax - b.')
    disp(['   ||A*sol_inclBP_x - b||_{2}^{2} = ', ...
        num2str(norm_residual_inclBP)])
    disp(['   ||A*(x_g_feasible) - b||_{2}^{2} = ', ...
        num2str(norm_residual_greedy_b)])
    disp(' ')


    disp('2. l1 norm of x satisfying Ax=b')
    disp([' ||sol_inclBP_x||_{1} = ', num2str(norm_l1_inclBP)])
    disp([' ||x_g_feasible||_{1} = ', num2str(norm_l1_feasible)])
    disp([' ||x_QR||_{1} = ', num2str(norm_l1_linsolve)])
    disp([' ||x_LU||_{1} = ', num2str(norm_l1_LUdecomp)])
    disp(' ')

    disp('3. Norm of sol_g_x and sol_g_xf - sol_g_x')
    disp([' ||sol_g_x||_{1} = ', num2str(norm(sol_g_x,1))])
    disp([' ||sol_g_xf - sol_g_x||_{1} = ', ...
        num2str(norm(sol_g_xf - sol_g_x,1))])
    disp(' ')
    

    disp(['5. Analyzing what happens when we use wmin: ' ...
        'counting nonzero components...'])
    disp('-Atop*sol_BP_p, sol_BP_x, -Atop*sol_g_p, sol_g_xf')
    disp([-A.'*sol_inclBP_p,sol_inclBP_x,-A.'*sol_g_p, sol_g_xf])

    

    % disp('6. Ax = bk = bk-1 + [...] analysis...')
    % residual = sol_g_b(:,end) - sol_g_b(:,m);
    % t_check = norm(A.'*residual,inf);
    % residual_2 = residual/t_check;
    % A_times_residual_2 = A.'*residual_2;


    %disp(norm(A*sol_g_x(:,end) - sol_g_b(:,m) - t_check*residual_2))
    %disp(abs(A.'*residual_2))

    % % Placeholders for comparing the primal solutions
    % norm_primal_glmnet_incl = zeros(kmax,1);
    % norm_primal_fista_incl = zeros(kmax,1);
    % norm_primal_fista_glmnet = zeros(kmax,1);
    % 
    % % Placeholders for comparing the dual solutions
    % norm_dual_glmnet_incl = zeros(kmax,1);
    % norm_dual_fista_incl = zeros(kmax,1);
    % norm_dual_fista_glmnet = zeros(kmax,1);
    % 
    % % Placeholders for comparing the primal objective functions
    % primal_obj_glmnet = zeros(kmax,1);
    % primal_obj_incl = zeros(kmax,1);
    % norm_primal_obj_glmnet_incl = zeros(kmax,1);
    % 
    % % Placeholders for comparing the dual objective functions
    % dual_obj_glmnet = zeros(kmax,1);
    % dual_obj_incl = zeros(kmax,1);
    % norm_dual_obj_glmnet_incl = zeros(kmax,1);
    % 
    % for k=1:1:kmax
    %     % Primal objective function
    %     if(k ~= kmax)
    %         primal_obj_glmnet(k) = (0.5/t(k))*norm(A*sol_glmnet_x(:,k)-b,2)^2 + ...
    %             norm(sol_glmnet_x(:,k),1);
    %         primal_obj_incl(k) = (0.5/t(k))*norm(A*sol_incl_x(:,k) - b,2)^2 + ...
    %             norm(sol_incl_x(:,k),1);
    %     else
    %         primal_obj_glmnet(k) = norm(sol_glmnet_x(:,k),1);
    %         primal_obj_incl(k) = norm(sol_incl_x(:,k),1);
    %     end
    %     norm_primal_obj_glmnet_incl(k) = ...
    %         (primal_obj_incl(k) - primal_obj_glmnet(k))/primal_obj_incl(k);
    % 
    %     % Dual objective function
    %     dual_obj_glmnet(k) = 0.5*t(k)*norm(sol_glmnet_p(:,k),2)^2 + ...
    %         sol_glmnet_p(:,k).'*b;
    %     dual_obj_incl(k) = 0.5*t(k)*norm(sol_incl_p(:,k),2)^2 + ...
    %         sol_incl_p(:,k).'*b;
    %     norm_dual_obj_glmnet_incl(k) = ...
    %         (dual_obj_incl(k) - dual_obj_glmnet(k))/dual_obj_incl(k);
    % 
    %     % Dual and primal solutions
    %     norm_dual_glmnet_incl(k) = norm(sol_glmnet_p(:,k)-sol_incl_p(:,k),inf)/...
    %         norm(sol_incl_p(:,k),inf);
    %     norm_primal_glmnet_incl(k) = norm(sol_glmnet_x(:,k)-sol_incl_x(:,k),inf)/...
    %         norm(sol_incl_x(:,k),inf);
    % end
    % 
    % 
    % if(use_fista)
    %     for k=1:1:kmax
    %         norm_dual_fista_incl(k) = norm(sol_fista_p(:,k)-sol_incl_p(:,k),inf);
    %         norm_dual_fista_glmnet(k) = norm(sol_fista_p(:,k)-sol_glmnet_p(:,k),inf);
    %     end
    % end
    % 
    % % Primal vs dual for the BPDN method
    % primal_dual_incl_difference = primal_obj_incl + dual_obj_incl;
    % disp('primal_obj_incl; dual_obj_incl; primal_dual_obj_diff')
    % disp([primal_obj_incl, -dual_obj_incl, primal_dual_incl_difference])
    % 
    % % Diff inclusions vs GLMNET: Primal objective function
    % disp('primal_obj_incl; primal_obj_glmnet; incl-glmnet')
    % disp([primal_obj_incl, primal_obj_glmnet, norm_primal_obj_glmnet_incl])
    % 
    % % Diff inclusions vs GLMNET: Dual objective function
    % disp('dual_obj_incl; dual_obj_glmnet; incl-glmnet')
    % disp([dual_obj_incl, dual_obj_glmnet, norm_dual_obj_glmnet_incl])
    % 
    % disp('Rel. error in linf norm of the differences between primal solutions (glmnet/incl first).')
    % disp([norm_dual_glmnet_incl, norm_dual_fista_incl,norm_dual_fista_glmnet])
    % 
    % disp('Rel. error in linf norm of the differences between dual solutions (glmnet/incl first).')
    % disp([norm_dual_glmnet_incl, norm_dual_fista_incl,norm_dual_fista_glmnet])
end