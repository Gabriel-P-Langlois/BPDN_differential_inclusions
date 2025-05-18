function [sol_x, sol_p, count] = ...
    BPDN_inclusions_regpath_solver(A,b,p0,t,tol)
% BPDN_inclusions_solver    Computes the primal and dual solutions of the 
%                           BPDN problem \{min_{x \in \Rn} 
%                               \frac{1}{2t}\normsq{Ax-b} + ||x||_1 \},
%                           up to the tolerance level tol, using
%                           the minimal selection principle and solving
%                           the corresponding slow system.
%
%
%                           This code computes the solution to the problem 
%                           above along a regularization path, starting 
%                           from t(1) until t(length(t)). In particular, 
%                           it reuses a previous computed dual solution
%                           sol_p(k) as the initial starting point of 
%                           computation of the next iterate k + 1.
%
%
%                           This code is optimized and uses the QR
%                           decomposition and column updates to speed up
%                           the calculations.
%
%   Input
%       A       -   m by n design matrix of the BPDN problem
%       b       -   m dimensional col data vector of the BPDN problem
%       p0      -   m dimensional col initial value of the slow system
%       t       -   k-dimensional array of nonnegative hyperparameters 
%                   of the BPDN problem
%       tol     -   small positive number (e.g., 1e-08)
%
%   Output
%       sol_x   -   (n,k) array containing the solutions at different
%                   hyperparameters. It contains the primal solutions to 
%                   BPDN at hyperparameter t and data b within 
%                   tolerance level tol.
%       sol_p   -   (m,k) array containing the solutions at different
%                   hyperparameters. It contains the dual solutions to 
%                   BPDN at hyperparamter t and data b within 
%                   tolerance level tol.



%% Initialization
% Options, placeholders and initial conditions.
[m,n] = size(A);
tol_minus = 1-tol;

kmax = length(t);
sol_x = zeros(n,kmax);
sol_p = zeros(m,kmax);  
sol_p(:,1) = p0;
Atop_times_p = (sol_p(:,1).'*A).';

% Initialize the equicorrelation set.
eq_set = (abs(Atop_times_p) >= tol_minus);

% Compute sign(-A.'*p) and assemble the effective matrix.
vec_of_signs = sign(-Atop_times_p);
K = A(:,eq_set).*(vec_of_signs(eq_set).');

% Perform the initial QR decomposition and set the opts field.
[Q,R] = qr(K);
opts.UT = true;


%% Regularization path
count = 0;
for k=1:1:kmax  
    %% Compute the trajectory of the slow system in a piecewise fashion
    while(true)
        count = count + 1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute the NNLS problem 
        % min_{u>=0} ||K*u - (b + t*sol_p)||_{2}^{2}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        rhs = b + t(k)*sol_p(:,k);
        [u,d,eq_set,Q,R] = hinge_qr_lsqnonneg(K,Q,R,rhs,eq_set,opts,tol);
 
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
        % Check for convergence.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if(timestep == inf)
            if(t(k) > 0)
                sol_x(eq_set,k) = vec_of_signs(eq_set).*u(u~=0);
                sol_p(:,k) = sol_p(:,k) + d/t(k);
            else
                %u = K\rhs;
                sol_x(eq_set,k) = vec_of_signs(eq_set).*u(u~=0);
            end
            break;
            
        elseif(t(k) > 0 && timestep*t(k) > tol_minus)
            sol_x(eq_set,k) = vec_of_signs(eq_set).*u(u~=0);
            sol_p(:,k) = sol_p(:,k) + d/t(k);
            break;
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Updates
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Update sol_p, Atop_times_p, and sign(-Atop_times_p).
        sol_p(:,k) = sol_p(:,k) + timestep*d;
        Atop_times_p = Atop_times_p + timestep*Atop_times_d;
        vec_of_signs = sign(-Atop_times_p);
    
        % Update the equicorrelation set and assemble the effective matrix.
        new_eq_set = (abs(Atop_times_p) >= tol_minus); 
        ind = setxor(find(eq_set),find(new_eq_set));
        eq_set = new_eq_set;        
        
        K = A(:,eq_set).*(vec_of_signs(eq_set).');
    
        % Update the QR decomposition if one column is added.
        % Else, recompute the QR decomposition from scratch.
        if(isscalar(ind))
            % Extract new element added to eq_set and its column.
            col = A(:,ind).*(vec_of_signs(ind).');
            loc = find(find(eq_set) == ind);
    
            % Execute [Q,R] = qrinsert(Q,R,loc,col) without overhead.
            [~,nr] = size(R);
            R(:,loc+1:nr+1) = R(:,loc:nr);
            R(:,loc) = (col.'*Q).';
            [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
        else
            [Q,R] = qr(K);
        end
    end


    %% Prepare for the next step in the regularization path
    if(k < kmax)
        % Update the dual solution p, Atop_times_p and vector of signs
        sol_p(:,k+1) = sol_p(:,k);
        Atop_times_p = (sol_p(:,k+1).'*A).';
        vec_of_signs = sign(-Atop_times_p);
    
        % Update the equicorrelation set and assemble the effective matrix.
        new_eq_set = (abs(Atop_times_p) >= tol_minus); 
        ind = setxor(find(eq_set),find(new_eq_set));
        eq_set = new_eq_set;

        K = A(:,eq_set).*(vec_of_signs(eq_set).');
    
        % Update the QR decomposition if one column is added.
        % Else, recompute the QR decomposition from scratch.
        if(isscalar(ind))
            % Extract new element added to eq_set and its column.
            col = A(:,ind).*(vec_of_signs(ind).');
            loc = find(find(eq_set) == ind);
    
            % Execute [Q,R] = qrinsert(Q,R,loc,col) without overhead.
            [~,nr] = size(R);
            R(:,loc+1:nr+1) = R(:,loc:nr);
            R(:,loc) = (col.'*Q).';
            [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
        else
            [Q,R] = qr(K);  % Multiple column updates are not supported.
        end
    end
end

end