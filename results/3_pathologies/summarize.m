%%  summarize script
%   This script summarizes the results from the runall.m file 
%   of the ./results/1_calibration experiment

%%%%%%%%%%
% 1. Plot the objective function V_{Primal}(x^{s}(t)) as a function of t
%%%%%%%%%%
Vprimal_incl = primal_obj_fun(sol_incl_x,t,A,b);
Vprimal_glmnet = primal_obj_fun(sol_glmnet_x,t,A,b);

figure(1)
plot(t(1:end-1)/t0,Vprimal_incl(1:end-1),'-o','DisplayName','diff incl')
hold
plot(t(1:end-1)/t0,Vprimal_glmnet(1:end-1),'x','DisplayName','glmnet')

if(use_mlasso)
    Vprimal_mlasso = primal_obj_fun(sol_mlasso_x,t,A,b);
    plot(t(1:end-1)/t0,Vprimal_mlasso(1:end-1),'*','DisplayName','matlab lasso')
end
if(use_fista)
    Vprimal_fista = primal_obj_fun(sol_fista_x,t,A,b);
    plot(t(1:end-1)/t0,Vprimal_fista(1:end-1),'+','DisplayName','fista')
end
if(use_homotopy)
    Vprimal_homotopy = primal_obj_fun(sol_hBPDN_x,t,A,b);
    plot(t(1:end-1)/t0,Vprimal_homotopy(1:end-1),'d','DisplayName','homotopy')
end
title('Primal objective function V_{primal}(x(t)) vs t')
xlabel('t/t0')
ylabel('V_{primal}(x^{s}(t))')
set(gca, 'xdir', 'reverse')
lgd = legend('Location','northwest');
lgd.NumColumns = 2;


%%%%%%%%%%
% 2. Plot the objective function V_{dual}(p^{s}(t)) as a function of t
%%%%%%%%%%
Vdual_incl = dual_obj_fun(sol_incl_p,t,b);
Vdual_glmnet = dual_obj_fun(sol_glmnet_p,t,b);

figure(2)
plot(t(1:end-1)/t0,Vdual_incl(1:end-1),'-o','DisplayName','diff incl')
hold
plot(t(1:end-1)/t0,Vdual_glmnet(1:end-1),'x','DisplayName','glmnet')

if(use_mlasso)
    Vdual_mlasso = dual_obj_fun(sol_mlasso_p,t,b);
    plot(t(1:end-1)/t0,Vdual_mlasso(1:end-1),'*','DisplayName','matlab lasso')
end
if(use_fista)
    Vdual_fista = dual_obj_fun(sol_fista_p,t,b);
    plot(t(1:end-1)/t0,Vdual_fista(1:end-1),'+','DisplayName','fista')
end
if(use_homotopy)
    Vdual_homotopy = dual_obj_fun(sol_hBPDN_p,t,b);
    plot(t(1:end-1)/t0,Vdual_homotopy(1:end-1),'d','DisplayName','homotopy')
end
title('Dual objective function V_{dual}(p(t)) vs t')
xlabel('t/t0')
ylabel('V_{dual}(p^{s}(t))')
set(gca, 'xdir', 'reverse')
lgd = legend('Location','southwest');
lgd.NumColumns = 2;


%%%%%%%%%%
% 3. Compute the relative error between dual solutions
%%%%%%%%%%
norm_dual_glmnet_incl = zeros(kmax,1);
norm_dual_fista_incl = zeros(kmax,1);
norm_dual_mlasso_incl = zeros(kmax,1);
for k=1:1:kmax
    norm_dual_glmnet_incl(k) = ...
        norm(sol_glmnet_p(:,k)-sol_incl_p(:,k),2)/...
        norm(sol_incl_p(:,k),2);
end

figure(3)
plot(t(1:end-1)/t0,norm_dual_glmnet_incl(1:end-1),'-o','DisplayName','GLMNET')
title('Relative error between glmnet dual solution vs diff incl')
xlabel('t/t0')
ylabel('||p_{glmnet} - p_{incl}||_{\infty}/||p_{incl}||_{\infty}')
lgd = legend('Location','northeast');
set(gca, 'xdir', 'reverse')

if(use_fista)
    for k=1:1:kmax
        norm_dual_fista_incl(k) = ...
            norm(sol_fista_p(:,k)-sol_incl_p(:,k),2)/...
            norm(sol_incl_p(:,k),2);
    end
figure(4)
plot(t(1:end-1)/to,norm_dual_fista_incl(1:end-1),'-o','DisplayName','FISTA')
title('Relative error between fista dual solution vs diff incl')
xlabel('t/t0')
ylabel('||p_{fista} - p_{incl}||_{\infty}/||p_{incl}||_{\infty}')
lgd = legend('Location','northeast');
set(gca, 'xdir', 'reverse')
end

if(use_mlasso)
    for k=1:1:kmax
        norm_dual_mlasso_incl(k) = ...
            norm(sol_mlasso_p(:,k)-sol_incl_p(:,k),2)/...
            norm(sol_incl_p(:,k),2);
    end
figure(5)
plot(t(1:end-1)/t0,norm_dual_mlasso_incl(1:end-1),'-o','DisplayName','mlasso')
title('Relative error between mlasso dual solution vs diff incl')
xlabel('t/t0')
ylabel('||p_{mlasso} - p_{incl}||_{\infty}/||p_{incl}||_{\infty}')
lgd = legend('Location','northeast');
set(gca, 'xdir', 'reverse')
end


%%%%%%%%%%
% 4. Check for number of nonzero components
%%%%%%%%%%
nnz_glmnet = zeros(kmax,1);
nnz_incl = zeros(kmax,1);

for k=1:1:kmax
    nnz_glmnet(k) = nnz(sol_glmnet_x(:,k));
    nnz_incl(k) = nnz(sol_incl_x(:,k));
end
nnz_glmnet(kmax) = 0;
figure(6)
plot(t/t0,nnz_incl,'-o','DisplayName','diff incl')
hold
plot(t/t0,nnz_glmnet,'x','DisplayName','glmnet')
title('Number of nonzero coefficients found along numerical solutions')
xlabel('t/t0')
ylabel('nnz')
lgd = legend('Location','northwest');
set(gca, 'xdir', 'reverse')


%%%%%%%%%%
% 5. Check how close the optimality conditions are satisfied
%%%%%%%%%%
% First condition
% NOTE: GLMNET and mlasso are ignored because they does not
% provide the dual solutions directly.
disp('Opt. cond. tp-Ax-b for incl/fista')
check1_incl = zeros(kmax,1); check2_incl = zeros(kmax,1);
check2_glmnet = zeros(kmax,1);
for k=1:1:kmax
    check1_incl(k) = norm(t(k).*sol_incl_p(:,k) - ...
                (A*sol_incl_x(:,k) - b),inf);

    check2_incl(k) = norm(-A.'*sol_incl_p(:,k),inf);
    check2_glmnet(k) = norm(-A.'*sol_glmnet_p(:,k),inf);
end
disp(['    bpdn diff incl: ',num2str(max(check1_incl))])

if(use_fista)
    check1_fista = zeros(kmax,1); check2_fista = zeros(kmax,1);
    for k=1:1:kmax
        check1_fista(k) = norm(t(k).*sol_fista_p(:,k) - ...
                (A*sol_fista_x(:,k) - b),inf);
        check2_fista(k) = norm(-A.'*sol_fista_p(:,k),inf);
    end
    disp(['    fista: ', num2str(max(check1_fista(1:end-1)))])
end
if(use_mlasso)     
    check2_mlasso = zeros(kmax,1);
    for k=1:1:kmax
        check2_mlasso(k) = norm(-A.'*sol_mlasso_p(:,k),inf);
    end
end

% Second condition
disp(' ')
disp('Opt. cond. (||-At*p||_{inf} - 1) = 0 for incl/glmnet/fista/mlasso')
disp(['    bpdn diff incl: ', num2str(max(check2_incl)-1)])
disp(['    glmnet: ', num2str(max(check2_glmnet(1:end-1))-1)])
if(use_fista)
    disp(['    fista: ', num2str(max(check2_fista(1:end-1))-1)])
end
if(use_mlasso)
    disp(['    mlasso: ', num2str(max(check2_mlasso(1:end-1))-1)])
end
disp(' ')


%%%%%%%%%%
% 6. Check if the BP and linprog solutions match the true solution
%    from the spear datasets, if used.
%%%%%%%%%%
if(using_spear)
    norm_true_bp = norm(x-sol_incl_BP_x,inf);
    disp(['linf norm between true solution and diff incl: ',num2str(norm_true_bp)])
    if(use_mlinprog && ~isempty(sol_linprog_p))
        norm_true_linprog = norm(x-sol_linprog_x); 
    end
elseif(use_mlinprog && ~isempty(sol_linprog_p))
    norm_bp_mlinprog = norm(sol_linprog_x-sol_incl_BP_x,inf);
    disp(['linf norm between mlinprog and diff incl: ', ...
        num2str(norm_bp_mlinprog)])
end


%% Auxiliary function
function Vprimal = primal_obj_fun(x,t,A,b)
    kmax = length(t);
    Vprimal = zeros(kmax,1);
    for i=1:1:kmax-1
        Vprimal(i) = (0.5/t(i))*norm(A*x(:,i) - b,2)^2 + ...
            norm(x(:,i),1);
    end
    Vprimal(kmax) = norm(x(:,end),1);
end

function Vdual = dual_obj_fun(p,t,b)
    kmax = length(t);
    Vdual = zeros(kmax,1);
    for i=1:1:kmax
        Vdual(i) = 0.5*t(i).*(norm(p(:,i),2).^2) + b.'*p(:,i);
    end
end