function [sol_x, sol_p, sol_b, sol_xf, sol_eqset] = greedy_homotopy_threshold(A,b,tol)
% GREEDY_HOMOTOPY_THRESHOLD Computes a feasible point of the linear system
%                           Ax=b using a greedy algorithm inspired by
%                           the basis pursuit problem.
%
%                           This code selects w = wmin (thresholding rule)
%                           as the selection criterion.
%
%   Input
%       A       -   m by n design matrix of the BPDN problem
%       b       -   m dimensional col data vector of the BPDN problem
%       tol     -   small positive number (e.g., 1e-08)
%
%   Output
%       sol_x   -   An (n,1)-dimensional solutions to the modified basis
%                   pursuit problem at data sol_b. 
%               
%       sol_p   -   An (m,1)-dimensional solution to the modified dual
%                   basis pursuit problem.
% 
%       sol_b   -   An (m,1)-dimensional modified data to a modified basis
%                   pursuit denoising problem.
%
%       sol_xf      -   An (n,1)-dimensional vector satisfying A*xf = b;
%
%       sol_eqset   -   An (n,1)-dimensional boolean vector corresponding
%                       to the active set of sol_xf. So, sol_eqset(j) = 1
%                       means sol_xf(j) =/= 0.                       


%% Initialization
[m,n] = size(A);
tol_minus = 1-tol;

% Placeholders and initial condition
sol_x = zeros(n,1);
sol_y = zeros(n,1);
sol_b = zeros(m,1); sol_b(:,1) = b;
sol_t = zeros(m+1,1); 

sol_t(1) = norm(-A.'*b,inf);
sol_p = -b/sol_t(1);
Atop_times_p = (sol_p.'*A).';

% Initialize the equicorrelation set
eq_set = (abs(Atop_times_p) >= tol_minus);

% Compute sign(-A.'*p) and assemble the effective matrix
vec_of_signs = sign(-Atop_times_p);
K = A(:,eq_set).*(vec_of_signs(eq_set).');

% Perform the initial QR decomposition and set the opts field
[Q,R] = qr(K);
opts.UT = true;
data_is_sparse = issparse(A);


%% Greedy algorithm
for k=1:1:m
    % Compute the LSQ problem min_{u} ||K*v + t*sol_p||_{2}^{2}
    rhs = -sol_t(k)*sol_p;
    tmp = Q.'*rhs;
    if(~data_is_sparse)
        v = linsolve(R,tmp,opts);
    else
        v = R\tmp;
    end

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

    sol_t(k+1) = sol_t(k)/(1+(sol_t(k)*Ck));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Readjust the data term with a linear perturbation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Apply thresholding and readjust the linear solution
    u = vec_of_signs.*sol_x(k);
    sol_w = zeros(n,1);
    sol_w(eq_set) = max(0, -u(eq_set) - v);

    % Posteriori update of the LSQ solution
    v = v + sol_w(eq_set);

    % Posteriori update of the drift q in the data
    sol_q = K*(-sol_w(eq_set)/sol_t(k));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update the sol_b and sol_x variables.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    sol_b = sol_b + (sol_t(k+1) - sol_t(k))*sol_q;
    sol_x(eq_set) = sol_x(eq_set) + ...
        (1 - sol_t(k+1)/sol_t(k))*vec_of_signs(eq_set).*v;

    % Update sol_y so that sol_b = A*sol_y.
    sol_y = sol_y + (1-sol_t(k+1)/sol_t(k))*...
        vec_of_signs.*sol_w;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check for convergence and update the sol_p variable
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if(Ck == inf)
        break;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update steps
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Update sol_p, Atop_times_p, and sign(-Atop_times_p).
    timestep = (1/sol_t(k+1) - 1/sol_t(k));
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
    if(data_is_sparse)
        [Q,R] = qr(K);
    elseif(isscalar(ind))
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

% Compute the feasible point and its active set
sol_xf = sol_x - sol_y;
sol_eqset = find(abs(sol_xf) > tol);
end