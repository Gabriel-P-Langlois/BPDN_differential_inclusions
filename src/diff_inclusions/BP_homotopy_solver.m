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
Atop_times_p = (sol_p.'*A).';

% Initialize the equicorrelation set
eq_set = (abs(Atop_times_p) >= tol_minus);

% Compute sign(-A.'*p) and assemble the effective matrix
vec_of_signs = sign(-Atop_times_p);


%% Compute the trajectory of the slow system in a piecewise fashion
count = 0;
while(true)
        count = count + 1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute the NNLS problem 
        % min_{uj>=0 if j \in eq_set and xj = 0} ||K*u + t*sol_p||_{2}^{2}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rhs = -t*sol_p;
        K = A(:,eq_set).*(vec_of_signs(eq_set).');
        ind = find(abs(sol_x(eq_set)) < tol);

        if(isempty(ind))
            v = K\rhs;
            xi = K*v - rhs;
        else
            [v,xi] = hinge_for_homotopy(K,rhs,ind,tol);
        end

    
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
        if(t == 0)
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
        eq_set = (abs(Atop_times_p) >= tol_minus);   
end
end