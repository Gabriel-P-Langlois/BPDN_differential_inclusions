function [sol_x, sol_p, sol_t, count_NNLS, count_LSQ] = ...
    BPDN_incl_homotopy_s(A,b,kmax,tol)
% BP_incl_homotopy_s    Computes the primal and dual solutions of the 
%                           BPND problem \{min_{x \in \Rn} ||x||_1 
%                               + \frac{1}{2t}normsq{Ax-b} \},
%                           up to the tolerance level tol, via homotopy
%                           using the minimal selection principle
%
%                       1. The data (A,b) is assumed to be sparse.
%
%   Input
%       A       -   m by n design matrix of the BPDN problem
%       b       -   m dimensional col data vector of the BPDN problem
%       kmax    -   Max Nb of points to include in the homotopy algorithm
%       tol     -   small positive number (e.g., 1e-08)
%
%   Output
%       sol_x   -   (n,k) primal solution to BP at hyperparameter t and data
%                   (A,b) within tolerance level tol.
%       sol_p   -   (m,k) dual solutions to BPDN at hyperparameter t and 
%                   data (A,b) within tolerance level tol.
%       sol_t   -   (k,1)-dimensional vector of positive numbers t,
%                   corresponding to the hyperparameters found by homotopy.
%       count_NNLS  -   Number of NNLS calls.
%
%       count_LSQ   -   Total number of LSQ solves


%% Initialization
% Options, placeholders and initial conditions
[m,n] = size(A);
tol_minus = 1-tol;

sol_t = zeros(1,kmax); sol_t(1) = norm(A.'*b,inf);
sol_x = sparse(n,kmax);
sol_p = zeros(m,kmax); sol_p(:,1) = -b/sol_t(1);
Atop_times_p = (sol_p(:,1).'*A).';

% Initialize the equicorrelation set
eq_set = (abs(Atop_times_p) >= tol_minus);

% Compute sign(-A.'*p) and assemble the effective matrix
vec_of_signs = sign(-Atop_times_p);
K = A(:,eq_set).*(vec_of_signs(eq_set).');


%% Compute the trajectory of the slow system in a piecewise fashion
k = 1;
count_NNLS = 0;
count_LSQ = 0;
while(k <= kmax)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute the NNLS problem 
        % min_{uj>=0 if j \in eq_set and xj = 0} ||K*u + t*sol_p||_{2}^{2}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        rhs = -sol_t(k)*sol_p(:,k);
        ind = find(abs(sol_x(eq_set,k)) < tol);

        [v,xi,num_linsolve] = hinge_homotopy_s(K,rhs,ind,tol);
        count_NNLS = count_NNLS + 1;
        count_LSQ = count_LSQ + num_linsolve;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute tplus and tminus
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        % Compute the maximal descent direction and tplus
        Atop_times_d = A.'*xi;
        pos_set = find(abs(Atop_times_d) > tol);
        timestep = inf;

        if(~isempty(pos_set))
            term1 = vec_of_signs.*Atop_times_d;
            term2 = vec_of_signs.*Atop_times_p;
            term3 = sign(term1);
            term4 = term3 - term2; 
            vec = term4./term1;    
            timestep = min(vec(pos_set));
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

        % Update the equicorrelation set and assemble the effective matrix
        eq_set = (abs(Atop_times_p) >= tol_minus);   
        K = A(:,eq_set).*(vec_of_signs(eq_set).');

        % Increment
        k = k + 1;
end

% Error if there was no convergence. Else, clip the output.
if(k == kmax)
    warning('Incomplete regularization path -- increase kmax!')
else
    sol_x(:,k+2:end) = [];
    sol_p(:,k+2:end) = [];
    sol_t(:,k+2:end) = [];
end
end