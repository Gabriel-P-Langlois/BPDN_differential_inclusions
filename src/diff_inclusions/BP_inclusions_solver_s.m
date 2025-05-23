function [sol_x, sol_p, count] = BP_inclusions_solver_s(A,b,p0,tol)
% BP_inclusions_solver_s    Computes the primal and dual solutions of the 
%                           BPDN problem \{min_{x \in \Rn} 
%                               ||x||_1 s.t. A*x = b\},
%                           up to the tolerance level tol, using
%                           the minimal selection principle and solving
%                           the corresponding slow system.
%
%   Input
%       A       -   m by n * sparse * design matrix of the BPDN problem
%       b       -   m dimensional col data vector of the BPDN problem
%       p0      -   m dimensional col initial value of the slow system
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
sol_x = zeros(n,1);
sol_p = p0;
Atop_times_p = (sol_p.'*A).';

% Initialize the equicorrelation set
eq_set = (abs(Atop_times_p) >= tol_minus);

% Compute sign(-A.'*p) and assemble the effective matrix
vec_of_signs = sign(-Atop_times_p);
K = A(:,eq_set).*(vec_of_signs(eq_set).');


%% Compute the trajectory of the slow system in a piecewise fashion
count = 0;

%% Algorithm
while(true && count < 20000)
    count = count + 1;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the NNLS problem 
    % min_{u>=0} ||K*u - b||_{2}^{2}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [v,d] = hinge_lsqnonneg_s(K,b,tol);   

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the maximum admissible descent time over the
    % indices j \in {1,...,n} where abs(<A.'*d,ej>) >= 0.
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
        sol_x(eq_set) = vec_of_signs(eq_set).*v(v~=0);
        break;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Updates
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Update sol_p, Atop_times_p, and sign(-Atop_times_p).
    sol_p = sol_p + timestep*d;
    Atop_times_p = Atop_times_p + timestep*Atop_times_d;
    vec_of_signs = sign(-Atop_times_p);

    % Update the equicorrelation set and assemble the effective matrix.
    new_eq_set = (abs(Atop_times_p) >= tol_minus); 
    eq_set = new_eq_set;
    K = A(:,eq_set).*(vec_of_signs(eq_set).');
end
end