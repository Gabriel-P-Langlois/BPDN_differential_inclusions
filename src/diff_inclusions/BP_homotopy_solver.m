function [sol_x, sol_p, count] = BP_homotopy_solver(A,b,tol)
% BP_homotopy_solver    Computes the primal and dual solutions of the 
%                           BP problem \{min_{x \in \Rn} ||x||_1 
%                               subject to Ax = b \},
%                           up to the tolerance level tol, via homotopy
%                           using the minimal selection principle
%
%   Input
%       A       -   m by n design matrix of the BPDN problem
%       b       -   m dimensional col data vector of the BPDN problem
%       tol     -   small positive number (e.g., 1e-08)
%
%   Output
%       sol_x   -   Primal solution to BP at hyperparameter t and data
%                   (A,b) within tolerance level tol. (n,1) array
%       sol_p   -   Dual solution to BPDN at hyperparamter t and data
%                   (A,b) within tolerance level tol. (m,1) array
%       count   -   Number of nonnegative least-squares solved performed.


%% Initialization
% Options, placeholders and initial conditions
[~,n] = size(A);
tol_minus = 1-tol;
t = norm(A.'*b,inf);
sol_x = zeros(n,1);
sol_p = -b/t;
Atop_times_p = (sol_p(:,1).'*A).';

% Initialize the equicorrelation set
eq_set = (abs(Atop_times_p) >= tol_minus);

% Compute sign(-A.'*p) and assemble the effective matrix
vec_of_signs = sign(-Atop_times_p);


%% Compute the trajectory of the slow system in a piecewise fashion
count = 0;
kmax = 100*n; 
while(count <= kmax)
        count = count + 1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute the NNLS problem 
        % min_{uj>=0 if j \in eq_set and xj = 0} ||K*u + t*sol_p||_{2}^{2}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rhs = -t*sol_p;
        K = A(:,eq_set).*(vec_of_signs(eq_set).');
        ind = find(sol_x(eq_set) == 0);
        [v,xi] = hinge_mod_lsqnonneg(K,rhs,ind,tol);

        % Issue after... either there is an issue in the function
        % or in the calculation below.
    
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
        tplus = t/(1 + t*timestep);

        % Compute tminus
        tmp_v = abs(sol_x(eq_set));
        ind_tminus = find(v < -tmp_v);
        if(~isempty(ind_tminus))
            tminus = t*(1-min(tmp_v(ind_tminus)./abs(v(ind_tminus))));
        else
            tminus = -inf;
        end
        % Compute next time
        told = t;
        t = max(tminus,tplus);
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check for convergence
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(t == 0 || count == kmax)
            sol_x(eq_set) = sol_x(eq_set) + (vec_of_signs(eq_set).*v);
            break;
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update dual solution and the equicorrelation set
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Update the primal solution x
        sol_x(eq_set) = sol_x(eq_set) + ...
            (1-t/told)*(vec_of_signs(eq_set).*v);

        % Update the dual solution p, Atop_times_p and vector of signs
        sol_p = sol_p + (1/t - 1/told)*xi;
        Atop_times_p = Atop_times_p + (1/t - 1/told)*Atop_times_d;
        vec_of_signs = sign(-Atop_times_p);

        % Update the equicorrelation set
        if(t == tminus)
            [~,J] = min(abs(sol_x(eq_set)));
            ind = find(eq_set);
            I = ind(J);
            eq_set(I) = 0;
        else
            eq_set = (abs(Atop_times_p) >= tol_minus);   
        end
end
if(count == kmax)
    disp('Possible error: convergence not achieved')
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
u = A\b;

% Check if the least-squares solution is feasible
if(min(u(ind)) >= tol)
    d = A*u - b;
    x = zeros(l,1);
    x(eqset) = u;
    return;    % STOP; the least-squares solution is valid.
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