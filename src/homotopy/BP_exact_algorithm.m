function [sol_x,sol_p] = BP_exact_algorithm(A,b,tol,disp_output_bp)
%% Description -- bp_algorithm
% This function computes a solution to the Basis Pursuit problem
%   min_{x \in \Rn} \norm{x}_{1}   s.t. Ax = b
% using the methodology described in
%
% Tendero, Yohann, Igor Ciril, Jérôme Darbon, and Susana Serna. 
% "An Algorithm Solving Compressive Sensing Problem 
% Based on Maximal Monotone Operators." 
% SIAM Journal on Scientific Computing 43, no. 6 (2021): A4067-A4094.
%
% Input:
% - A: An m x n matrix where m <= n.
% - b: An m x 1 vector b where b \in \span{A}.
% - tol: A positive scalar that is used for the error tolerance.
%        E.g., tol = 1e-10.
% - disp_output_bp: A boolean variable (true/false). Setting this to true
%                    will print some additional information.
%
% Output:
% - sol_x: An n x 1 solution to the Basis Pursuit problem
% - sol_p: An m x 1 solution to the dual of the Basis Pursuit problem
% Written by Gabriel Provencher Langlois


%% Initialization
tol_minus = 1-tol;

% Initialize the auxiliary quantities
[m,n] = size(A);
bminus = -b;
Atop_times_bminus = (bminus.'*A).';
t = norm(Atop_times_bminus,inf);

% Initialize the main quantities
sol_x = zeros(n,1);
sol_p = bminus/t;
Atop_times_pk = Atop_times_bminus/t; % Used to speed up the computations

% Compute initial equicorrelation set
equi_set = (abs(Atop_times_pk) >= tol_minus); 


%% Algorithm
k = 1;
max_iter = 50*m;
while(true && k < max_iter)
    %%%%% 1. Compute the relevant vectors and matrices
    K = A(:,equi_set);
    vector_of_signs = sign(-Atop_times_pk);
    D = diag(vector_of_signs(equi_set));
    

    %%%%% 2. Solve the LSQ problem
    u = lsqnonneg(K*D,b);
    v = zeros(n,1); v(equi_set) = D*u;


    %%%%% 3. Compute the descent direction
    d = K*(D*u) + bminus;

    % Compute A.'*d and the set of indices abs(A.'*d) > 0
    Atop_times_dk = (d.'*A).'; 
    active_set_plus = abs(Atop_times_dk) > tol;
    

    %%%%% 4. Compute the kick time
    %%% Compute the `smallest' amount of time we can descent over
    % the set of admissible faces. If the set of indices abs(A.'*d) > 0
    % is empty, then timestep = +\infty

    if (any(active_set_plus))
        term1 = vector_of_signs.*Atop_times_dk;
        sign_coeffs = sign(term1);
        term2 = -vector_of_signs.*Atop_times_pk;

        vec = (1+sign_coeffs(active_set_plus).*term2(active_set_plus))./...
            abs(term1(active_set_plus));

        timestep = min(vec);
    else
        sol_x = v; 
        break;
    end


    %%%%% 5. Update all solutions and variables
    alpha = 1/(1+t*timestep);
    sol_x = alpha*sol_x + (1-alpha)*v; 
    sol_p = sol_p + timestep*d;
    t = alpha*t;


    % Compute next equicorrelation set
    Atop_times_pk = Atop_times_pk + timestep*Atop_times_dk;
    equi_set = (abs(Atop_times_pk) >= tol_minus); 


    %%%%% 6. Optionals
    % If enabled, display the current iteration and the timestep
    % used in the calculation.
    if(disp_output_bp)
        disp(['Iteration: ', num2str(k),' has timestep = ',num2str(timestep)])
    end

    %%%%% 7. Increment
    k=k+1;
end

% Display warning if max_iter has been reached
% Note: May possibly be reached for some contorted matrix A.
if(k == max_iter)
    disp('Warning: k = 100*m iterations reached!')
end
end