%% Description
% FISTA for solving the classic lasso problem
%
% Note: Convergence criterion require an extra matrix multiplication...
%       This likely can also be optimized away...
%
% min_{x} 0.5*normsq(Ax-b) + t*normone(x)
%

function [xplus,pplus,num_iters] = lasso_fista_solver(x,p,lambda,A,b,...
    tau,max_iters,tol,min_iters_fista)

% Auxiliary variables
xminus = x; pminus = p;
tminus = 0; beta = 0;

% Counter for the iterations
num_iters = 0;

% Iterations
for k=1:1:max_iters
    num_iters = num_iters + 1;
    
    % FISTA Variable update
    tmp1 = x + beta*(x - xminus);
    tmp2 = p + beta*(p-pminus);
    tmp1 = tmp1 - ((tau*tmp2).'*A).';
    
    % Proximal calculation
    xplus = l1_prox_operator(tmp1,tau*lambda);
    pplus = A*xplus-b;
    
    % Check for convergence
    if(num_iters >= min_iters_fista && lasso_conv_criterion(num_iters,max_iters,...
                lambda,(-pplus.'*A).',tol))
        break;
    end
    
    % Increment
    t = 0.5*(1 + sqrt(1 + 4*(tminus^2)));
    beta = (tminus - 1)/t;
    xminus = x; x = xplus;
    pminus = p; p = pplus;
    tminus = t;
end
end