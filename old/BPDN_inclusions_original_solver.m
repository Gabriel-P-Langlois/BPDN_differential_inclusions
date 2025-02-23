function [sol_x, sol_p] = BPDN_inclusions_solver(A,b,p0,t,tol)
% BPDN_inclusions_solver    Computes the primal and dual solutions of the 
%                           BPDN problem \{min_{x \in \Rn} 
%                               \frac{1}{2t}\normsq{Ax-b} + ||x||_1 \},
%                           up to the tolerance level tol, using
%                           the minimal selection principle and solving
%                           the corresponding slow system.
%
%   Input
%       A       -   m by n design matrix of the BPDN problem
%       b       -   m dimensional col data vector of the BPDN problem
%       p0      -   m dimensional col initial value of the slow system
%       t       -   nonnegative hyperparameter of the BPDN problem
%       tol     -   small positive number (e.g., 1e-08)
%
%   Output
%       sol_x   -   Primal solution to BPDN at hyperparameter t and data
%                   (A,b) within tolerance level tol.
%       sol_p   -   Dual solution to BPDN at hyperparamter t and data
%                   (A,b) within tolerance level tol.


%% Notes
%   1)  This differential inclusion solver uses the QR decomposition 
%       to improve effiency.


%% Initialization
[~,n] = size(A);
tol_minus = 1-tol;

% Placeholders
sol_x = zeros(n,1);
sol_p = p0;
Atop_times_p = (sol_p.'*A).';

% Initialize the equicorrelation set
eq_set = (abs(Atop_times_p) >= tol_minus); 

% Compute sign(-A.'*p) and assemble the effective matrix
vec_of_signs = sign(-Atop_times_p);
K = A(:,eq_set).*(vec_of_signs(eq_set).');

% Perform the QR decomposition
[Q,R] = qr(K,"econ","vector");


%% Compute the trajectory of the slow system in a piecewise fashion
while(true)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the NNLS problem 
    % min_{u>=0} ||K*u - b + t*sol_p||_{2}^{2}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Compute first the least-squares solution.
    rhs = b + t*sol_p;
    tmp = (rhs.'*Q).';
    u = R\tmp;

    % Check if the least squares solution is positive. 
    % If not, invoke the method of hinges 
    % with full active set to solve it.

    if(min(u) > -tol)
        d = Q*(R*u) - rhs;
    else
        [u,d,eq_set] = hinge_qr_lsqnonneg(K,Q,R,u,rhs,eq_set,tol);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the maximum descent time 
    % over the set where abs(A.'*d) >= 0
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Atop_times_d = (d.'*A).';
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check for convergence
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if(timestep == inf)
        sol_x(eq_set) = vec_of_signs(eq_set).*u;
        if(t > 0)
            sol_p = sol_p + d/t;
        end
        break;
    elseif(t > 0 && timestep*t > tol_minus)
        sol_x(eq_set) = vec_of_signs(eq_set).*u;
        sol_p = sol_p + d/t;
        break;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update quantities and reassemble the effective matrix
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Dual solution p and the vector Atop_times_p = A.'*p
    sol_p = sol_p + timestep*d;
    Atop_times_p = Atop_times_p + timestep*Atop_times_d;

    % Equicorrelation set and the vector of signs
    eq_set = (abs(Atop_times_p) >= tol_minus); 
    vec_of_signs = sign(-Atop_times_p);
    
    % Assemble the effective matrix
    K = A(:,eq_set).*(vec_of_signs(eq_set).');

    % Update the QR decomposition
    [Q,R] = qr(K,"econ","vector");  % BOTTLENECK
end
end