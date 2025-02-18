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
%
%
%% TODO
% 1)    Initialize the matrix of signs and equicorrelation set OUTSIDE the
%       loop, and add the extra postprocessing step after.
%
% 2)    Look for small economies; initialize u once before the loop, replace
%       u(eq_set) as v instead, and only at the end have u(active_set) = v.
%
% 3)    Once that's done, create a copy of this code + hinge lsqnnoneg
%       and create fast_* methods instead. The fast methods will precompute
%       the QR decomposition OUTSIDE of the hinge_lsqnnonneg and update
%       it appropriately. This is the one remaining bottleneck.

%% Initialization
[~,n] = size(A);
tol_minus = 1-tol;

sol_x = zeros(n,1);
sol_p = p0;
Atop_times_p = (sol_p.'*A).';

% Compute the trajectory of the slow system in a piecewise fashion
while(true)
    % Compute the matrix of signs and the equicorrelation set
    vec_of_signs = sign(-Atop_times_p);
    eq_set = (abs(Atop_times_p) >= tol_minus); 


    % Solve the NNLS problem
    % min_{u_{eq_set} >= 0} ||A(eq_set)*D(eq_set)*u_{eq_set} - b - t*p||^2
    % and compute the descent direction.
    u = zeros(n,1);
    K = A(:,eq_set).*(vec_of_signs(eq_set).');
    [u(eq_set),d] = hinge_lsqnonneg(K,b + t*sol_p,tol);


    % Compute the maximum descent time over the set where abs(A.'*d) >= 0
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


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check for convergence and update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(t > 0)
        if(timestep == inf)
            sol_x = vec_of_signs.*u;
            break;
        elseif(timestep*t > tol_minus)
            sol_x = vec_of_signs.*u;
            sol_p = sol_p + d/t;
            break;
        else
            sol_p = sol_p + timestep*d;
            Atop_times_p = Atop_times_p + timestep*Atop_times_d;
            % Continue here...
        end
    else
        if(timestep == inf)
            sol_x = vec_of_signs.*u;
            break;
        else
            sol_p = sol_p + timestep*d;
            Atop_times_p = Atop_times_p + timestep*Atop_times_d;
            % Continue here...
        end
    end
end
end