function [sol_x, sol_p, sol_t, k] = BPDN_homotopy_solver(A,b,tol)
% BP_homotopy_solver    Computes the primal and dual solutions of the 
%                           BPND problem \{min_{x \in \Rn} ||x||_1 
%                               + \frac{1}{2t}normsq{Ax-b} \},
%                           up to the tolerance level tol, via homotopy
%                           using the minimal selection principle
%
%   Input
%       A       -   m by n design matrix of the BPDN problem
%       b       -   m dimensional col data vector of the BPDN problem
%       tol     -   small positive number (e.g., 1e-08)
%
%   Output
%       sol_x   -   (n,k) primal solution to BP at hyperparameter t and data
%                   (A,b) within tolerance level tol.
%       sol_p   -   (m,k) dual solutions to BPDN at hyperparameter t and 
%                   data (A,b) within tolerance level tol.
%       sol_t   -   (k,1)-dimensional vector of positive numbers t,
%                   corresponding to the hyperparameters found by homotopy.
%       k       -   Number of nonnegative least-squares solved performed.


%% Initialization
% Options, placeholders and initial conditions
[m,n] = size(A);
tol_minus = 1-tol;
kmax = 200*n;   % Upper bound on the max number of hyperparameters.

sol_t = zeros(1,kmax); sol_t(1) = norm(A.'*b,inf);
sol_x = zeros(n,kmax);
sol_p = zeros(m,kmax); sol_p(:,1) = -b/sol_t(1);
Atop_times_p = (sol_p(:,1).'*A).';

% Initialize the equicorrelation set
eq_set = (abs(Atop_times_p) >= tol_minus);

% Compute sign(-A.'*p) and assemble the effective matrix
vec_of_signs = sign(-Atop_times_p);


%% Compute the trajectory of the slow system in a piecewise fashion
k = 1;
while(k <= kmax)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute the NNLS problem 
        % min_{uj>=0 if j \in eq_set and xj = 0} ||K*u + t*sol_p||_{2}^{2}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rhs = -sol_t(k)*sol_p(:,k);
        K = A(:,eq_set).*(vec_of_signs(eq_set).');
        ind = find(sol_x(eq_set,k) == 0);
        [v,xi] = hinge_mod_lsqnonneg(K,rhs,ind,tol);

    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute tplus and tminus
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        % Compute the maximal descent direction and tplus
        Atop_times_d = (xi.'*A).';
        pos_set = abs(Atop_times_d) > tol;
        timestep = inf;

        if (any(pos_set))
            term1 = vec_of_signs.*Atop_times_d;
            term2 = vec_of_signs.*Atop_times_p;
            term3 = sign(term1);
            term4 = term3 - term2; 
    
            vec = term4(pos_set)./term1(pos_set);
            timestep = min(vec);
        end
        tplus = sol_t(k)/(1 + sol_t(k)*timestep);

        % Compute tminus
        tmp_v = abs(sol_x(eq_set,k));
        ind_tminus = find(v < -tmp_v);
        if(~isempty(ind_tminus))
            tminus = sol_t(k)*...
                (1-min(tmp_v(ind_tminus)./abs(v(ind_tminus))));
        else
            tminus = -inf;
        end

        % Compute next time
        sol_t(k+1) = max(tminus,tplus);

    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check for convergence
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(sol_t(k+1) == 0)
            sol_x(eq_set,k+1) = sol_x(eq_set,k) + ...
                (vec_of_signs(eq_set).*v);
            sol_p(:,k+1) = sol_p(:,k);
            break;
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update dual solution and the equicorrelation set
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Update the primal solution x
        sol_x(eq_set,k+1) = sol_x(eq_set,k) + ...
            (1-sol_t(k+1)/sol_t(k))*(vec_of_signs(eq_set).*v);

        % Update the dual solution p, Atop_times_p and vector of signs
        sol_p(:,k+1) = sol_p(:,k) + (1/sol_t(k+1) - 1/sol_t(k))*xi;
        Atop_times_p = Atop_times_p + ...
            (1/sol_t(k+1) - 1/sol_t(k))*Atop_times_d;
        vec_of_signs = sign(-Atop_times_p);

        % Update the equicorrelation set
        if(sol_t(k+1) == tminus)
            [~,J] = min(abs(sol_x(eq_set,k+1)));
            ind = find(eq_set);
            I = ind(J);
            eq_set(I) = 0;
        else
            eq_set = (abs(Atop_times_p) >= tol_minus);   
        end
        k = k + 1;
end
% Error if there was no convergence. Else, clip the output.
if(k == kmax)
    disp('Possible error: convergence not achieved')
else
    sol_x(:,k+2:end) = [];
    sol_p(:,k+2:end) = [];
    sol_t(:,k+2:end) = [];
end
end





function [x,d] = hinge_mod_lsqnonneg(A,b,ind,tol)
% HINGE_MOD_LSQNONNEG   This function computes the nonnegative LSQ problem
%                       min_{x \in \Rn} ||A*x-b||_{2}^{2}
%                       subject to xj >= 0 if j \in ind 
%                       using the ``Method of Hinges" presented in
%                       ``A Simple New Algorithm for Quadratic Programming 
%                       with Applications in Statistics" by Mary C. Meyers.
%
%
%   Input
%           A           -   (m x l)-dimensional design matrix A
%           b           -   m-dimensional col data vector.
%           ind         -   ind \coloneqq \subset {1,...l}
%           tol         -   small number specifying the tolerance
%                           (e.g., 1e-08).
%
%   Output
%           x   -   l-dimensional col solution vector of nonzero
%                   coefficients to the NNLS problem.
%           d   -   m-dimensional col residual vector d = A*x - b


%% Algorithm: Method of Hinges
% Invoke the method of hinges with full active set
[~, l] = size(A);
eqset = true(l,1);
[u,~] = linsolve(A,b);

% Check if the least-squares solution is feasible. If so, stop.
if(min(u(ind)) >= tol)
    d = A*u - b;
    x = zeros(l,1);
    x(eqset) = u;
    return;
end

% Compute the solution via the method of hinges
while(true)
    if(min(u(ind)) < -tol)       % Check if the primal constraint is violated.
        x = zeros(l,1);
        x(eqset) = u;
        [~,J] = min(x(ind));
        I = ind(J);
        eqset(I) = false;
    else                    % Check if the dual constraint is violated.
        theta = A(:,eqset)*u;
        rho = b - theta;
        tmp = rho.'*A;
        [val,J] = max(tmp(ind));
        I = ind(J);

        if(val > tol)
            eqset(I) = true;
        else
            break;          % STOP; all constraints are satisfied.
        end
    end

    % Compute the solution to A(:,eqset)*u = b and continue.
    u = A(:,eqset)\b;
end

% Return the solution vector x and the residual d = A*x-b
x = zeros(l,1);
x(eqset) = u;
d = -rho;
end