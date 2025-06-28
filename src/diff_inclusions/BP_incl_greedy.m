function [sol_x, sol_p, count_NNLS, count_LSQ] = BP_incl_greedy(A,b,tol)
% BP_INCL_GREEDY            Computes the primal and dual solutions to BP 
%                           \{min_{x \in \Rn} ||x||_1 s.t. A*x = b\},
%                           up to the tolerance level tol, using
%                           differential inclusions and the minimal
%                           selection principle.
%
%                           1) This code uses the QR decomposition and
%                           column updates to speed up the calculations.
%
%                           2) This code first calculates a feasible point
%                              of BP before solving the problem exactly.
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
%       count_NNLS  -   Total number of NNLS calls
%
%       count_LSQ   -   Total number of LSQ solves                    


%% Initialization
[m,n] = size(A);
tol_minus = 1-tol;

% Placeholders and initial condition
sol_x = zeros(n,1);
sol_t = norm(-A.'*b,inf);
sol_p = -b/sol_t;
Atop_times_p = (sol_p.'*A).';

% Initialize the equicorrelation set
eq_set = (abs(Atop_times_p) >= tol_minus);

% Compute sign(-A.'*p) and assemble the effective matrix
vec_of_signs = sign(-Atop_times_p);
K = A(:,eq_set).*(vec_of_signs(eq_set).');

% Perform the initial QR decomposition and set the opts field
[Q,R] = qr(K);
opts.UT = true;
count_LSQ = 0;
count_NNLS = 0;


%% Greedy algorithm
for k=1:1:m
    % Compute the LSQ problem min_{u} ||K*v + t*sol_p||_{2}^{2}
    rhs = -sol_t*sol_p;
    tmp = Q.'*rhs;
    v = linsolve(R,tmp,opts);
    count_LSQ = count_LSQ + 1;

    % Compute the descent direction.
    xi = K*v - rhs;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the maximum admissible descent time over the
    % indices j \in {1,...,n} where abs(<A.'*xi,ej>) >= 0.
    % Update the time variable accordingly.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Atop_times_xi = A.'*xi;
    pos_set = abs(Atop_times_xi) > tol;
    Ck = inf;

    if (any(pos_set))
        term1 = vec_of_signs.*Atop_times_xi;
        term2 = vec_of_signs.*Atop_times_p;
        term3 = sign(term1);
        term4 = term3 - term2; 
        vec = term4./term1;
        Ck = min(vec(pos_set));
    end

    fac = sol_t/(1+(sol_t*Ck));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check for convergence/update the sol_p variable
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Convergence check
    if(Ck == inf)
        break;
    end

    % Update sol_p, Atop_times_p, and sign(-Atop_times_p).
    timestep = (1/fac - 1/sol_t);
    sol_t = fac;
    sol_p = sol_p + timestep*xi;
    Atop_times_p = Atop_times_p + timestep*Atop_times_xi;
    vec_of_signs = sign(-Atop_times_p);

    % Update the equicorrelation set and assemble the effective matrix.
    new_eq_set = (abs(Atop_times_p) >= tol_minus);
    ind = setxor(find(eq_set),find(new_eq_set));

    eq_set = new_eq_set;
    K = A(:,eq_set).*(vec_of_signs(eq_set).');

    % Update the QR decomposition if one column is added.
    % Else, recompute the QR decomposition from scratch.
    if(~isscalar(ind))
        [Q,R] = qr(K);
    else
        % Extract new element added to eq_set and its column.
        col = A(:,ind).*(vec_of_signs(ind).');
        loc = find(find(eq_set) == ind);

        % Call MATLAB's [Q,R] = qrinsert(Q,R,loc,col) without overhead
        [~,nr] = size(R);
        R(:,loc+1:nr+1) = R(:,loc:nr);
        R(:,loc) = Q'*col;
        [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
    end
end


%% Invoke the BP_incl_direct.m solver using the warm start.
while(true)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the NNLS problem 
    % min_{u>=0} ||K*u - b||_{2}^{2}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [v,d,new_eq_set,Q,R,num_linsolve] = ...
        hinge_qr_lsqnonneg(K,Q,R,b,eq_set,opts,tol);    

    count_NNLS = count_NNLS + 1;
    count_LSQ = count_LSQ + num_linsolve;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute the maximum admissible descent time over the
    % indices j \in {1,...,n} where abs(<A.'*d,ej>) >= 0.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Atop_times_d = A.'*d;
    pos_set = abs(Atop_times_d) > tol;
    timestep = inf;

    if(any(pos_set))
        term1 = vec_of_signs.*Atop_times_d;
        term2 = vec_of_signs.*Atop_times_p;
        term3 = sign(term1);
        term4 = term3 - term2; 
        vec = term4./term1;    
        timestep = min(vec(pos_set));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check for convergence
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if(timestep == inf)
        sol_x(eq_set) = vec_of_signs(eq_set).*v;
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
    new_eq_set_2 = (abs(Atop_times_p) >= tol_minus); 
    ind = setxor(find(new_eq_set),find(new_eq_set_2));
    eq_set = new_eq_set_2;
    K = A(:,eq_set).*(vec_of_signs(eq_set).');

    % Update the QR decomposition if only one column is added.
    % Else, recompute the QR decomposition from scratch.
    if(isscalar(ind))
        % Extract new element added to eq_set and its column.
        col = A(:,ind).*(vec_of_signs(ind).');
        loc = find(find(eq_set) == ind);

        % Call MATLAB's [Q,R] = qrinsert(Q,R,loc,col) without overhead
        [~,nr] = size(R);
        R(:,loc+1:nr+1) = R(:,loc:nr);
        R(:,loc) = Q'*col;
        [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
    else
        [Q,R] = qr(K); 
    end
end
end