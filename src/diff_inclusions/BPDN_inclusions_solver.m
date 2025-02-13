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


%% Algorithm
% Initialization
[~,n] = size(A);
tol_minus = 1-tol;
sol_x = zeros(n,1);
sol_p = p0;


% Compute the trajector of the slow system in a piecewise fashion
while(true)

    % Precompute -A.'*p and create placeholder vector u
    Atop_times_p = (sol_p.'*A).';
    u = zeros(n,1);

    % Compute the matrix of signs and the equicorrelation set
    D = diag(sign(-Atop_times_p));
    eq_set = (abs(Atop_times_p) >= tol_minus); 

    % Solve the nonnegative least squares problem
    % min_{u_{eq_set} >= 0} ||A(eq_set)*D(eq_set)*u_{eq_set} - b - t*p||^2
    u(eq_set) = lsqnonneg(A(:,eq_set)*D(eq_set,eq_set),b + t*sol_p);

    % Compute the descent direction
    d = A(:,eq_set)*D(eq_set,eq_set)*u(eq_set) - b - t*sol_p;

    % Compute the maximum descent time
    Atop_times_d = (d.'*A).';
    ac_set = abs(Atop_times_d) > tol;
    timestep = inf;

    if (any(ac_set))
        term1 = D*Atop_times_d;
        term2 = D*Atop_times_p;
        term3 = sign(term1);

        vec = (term3(ac_set) - term2(ac_set))./term1(ac_set);
        timestep = min(vec);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check for convergence and  update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(t > 0)
        if(timestep == inf)
            sol_x = D*u;
            break;
        elseif(timestep*t > tol_minus)
            sol_x = D*u;
            sol_p = sol_p + d/t;
            break;
        else
            sol_p = sol_p + timestep*d;
        end
    else
        if(timestep == inf)
            sol_x = D*u;
            break;
        else
            sol_p = sol_p + timestep*d;
        end
    end
end
end