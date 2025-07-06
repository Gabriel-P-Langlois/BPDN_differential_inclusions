function [xplus,pplus,num_iters] = lasso_fista_solver(x,p,t,A,b,...
    tau,max_iters,tol,min_iters)
% lasso_fista_solver    Computes the primal and dual solution of the
%                       LASSO problem
%                       min_{x \in \Rn} \frac{1}{2}\normsq{Ax-b} + t||x||_1
%                       within a certain tolerance level using the
%                       accelerated FISTA algorithm.
%
%   Input
%       x           -   n dimensional col vector. This is used as a warm
%                       start for the fista algorithm, typically using a
%                       primal solution at a different parameter t.
%
%       p           -   m dimensional col vector. This is used as a warm
%                       start for the fista algorithm, typically using a
%                       dual solution at a different parameter t.
%
%       t           -   positive hyperparameter of the LASSO problem.
%       A           -   m by n design matrix of the LASSO problem.
%       b           -   m dimensional col data vector of the LASSO problem.
%       tau         -   positive number used for the acceleration.
%       max_iters   -   positive number
%       tol         -   small positive number (e.g., 1e-08)
%       min_iters   -   positive number
%
%   Output
%       xplus       -   Primal solution to the LASSO problem at 
%                       parameter t with data A and b.
%       pplus       -   Dual solution to the LASSO problem at
%                       parameter t with data A and b.
%       num_iters   -   Number of iterations taken to calculate the
%                       primal and dual solutions.


%% Algorithm
% Auxiliary variables
xminus = x; pminus = p;
rminus = 0; beta = 0;

% Counter for the iterations
num_iters = 0;

% Iterations
for k=1:1:max_iters
    num_iters = num_iters + 1;
    
    % FISTA Variable update
    tmp1 = x + beta*(x - xminus);
    tmp2 = tau*(p + beta*(p-pminus));
    tmp1 = tmp1 - A.'*tmp2;
    
    % Proximal calculation
    xplus = l1_prox_operator(tmp1,tau*t);
    tmp3 = A*xplus;
    pplus = tmp3-b;
    
    % Check for convergence
    if(num_iters >= min_iters)
        %stop = norm((-pplus.'*A).',inf) <= t*(1 + tol);
        stop = norm(pplus-p) <= tol*norm(p);
        if(stop || (num_iters >= max_iters))
            break;
        end
    end
    
    % Increment
    r = 0.5*(1 + sqrt(1 + 4*(rminus^2)));
    beta = (rminus - 1)/r;
    xminus = x; x = xplus;
    pminus = p; p = pplus;
    rminus = r;
end
end