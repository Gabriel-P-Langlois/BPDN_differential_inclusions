function [sol_x, sol_p] = BPDN_inclusions_regpath_solver(A,b,p0,t,tol)
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
%       t       -   k-dimensional array of nonnegative hyperparameters 
%                   of the BPDN problem
%       tol     -   small positive number (e.g., 1e-08)
%
%   Output
%       sol_x   -   Primal solution to BPDN at hyperparameter t and data
%                   (A,b) within tolerance level tol.
%                   (n,k) array containing the solutions at different
%                   hyperparameters.
%       sol_p   -   Dual solution to BPDN at hyperparamter t and data
%                   (A,b) within tolerance level tol.
%                   (m,k) array containing the solutions at different
%                   hyperparameters.


%% Initialization
[m,n] = size(A);
kmax = length(t);
tol_minus = 1-tol;
opts.UT = true;     % Option for the linsolve internal MATLAB function.
                    % We use the QR decomposition \w R upper triangular.

% Placeholders and initial condition
sol_x = zeros(n,kmax);
sol_p = zeros(m,kmax);  
sol_p(:,1) = p0;
Atop_times_p = (sol_p(:,1).'*A).';

% Initialize the equicorrelation set
eq_set = (abs(Atop_times_p) >= tol_minus);

% Compute sign(-A.'*p) and assemble the effective matrix
vec_of_signs = sign(-Atop_times_p);
K = A(:,eq_set).*(vec_of_signs(eq_set).');

% Perform the initial QR decomposition
[Q,R] = qr(K);


%% Regularization path
for k=1:1:kmax  
    %% Compute the trajectory of the slow system in a piecewise fashion
    while(true)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute the NNLS problem 
        % min_{u>=0} ||K*u - b + t*sol_p||_{2}^{2}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        % Compute the least-squares solution.
        % Note: We use the economical forms of the matrices Q and R for this.
        rhs = b + t(k)*sol_p(:,k);
        neff = sum(eq_set);
        tmp = (rhs.'*Q(:,1:neff)).';
        u = linsolve(R(1:neff,:),tmp,opts);
    
        % Check if the least squares solution is positive. 
        % If not, invoke the Method of Hinges
        % with full active set and compute its solution.
        if(min(u) > -tol)
            tmp2 = R*u;
            d = Q*tmp2;
            d = d - rhs;
        else
            % Invoke the Method of Hinges to compute the solution to the
            % NNLS problem.
            % Note: The code below is equivalent to the call
            %
            % [u,d,eq_set,Q,R] = ...
            %     hinge_qr_lsqnonneg(K,Q,R,u,rhs,eq_set,opts,tol);
            %
            % We invoke it here to avoid overhead and improve efficiency.

            active_set = true(neff,1);
            while(true)
                remove_update = false;
                insert_update = false;

                if(min(u) <= -tol)   % Check primal constraint
                    x = zeros(neff,1);
                    x(active_set) = u;
                    [~,I] = min(x);
                    active_set(I) = false;
                    remove_update = isscalar(I);
                else
                    tmp2 = R*u;
                    d = Q*tmp2;
                    d = rhs - d;
                    Ktop_times_rho = (d.'*K);
                    [val,I] = max(Ktop_times_rho);
            
                    if(val >= tol)  % Check dual constraint
                        active_set(I) = true;
                        insert_update = isscalar(I);
                    else
                        break;      % Exit; All constraints are satisfied.
                    end
                end

                % Update the QR decomposition. 
                if(remove_update)
                    [~, J] = min(u);
                    R(:,J) = [];

                    [Q,R] = matlab.internal.math.deleteCol(Q,R,J);
                elseif(insert_update)
                    [~, J] = max(Ktop_times_rho);
                    col = K(:,J);
                    [~,nr] = size(R);
                    R(:,J+1:nr+1) = R(:,J:nr);
                    R(:,J) = (col.'*Q).';

                    [Q,R] = matlab.internal.math.insertCol(Q,R,J);
                else
                    [Q,R] = qr(K(:,active_set));
                end

                % Compute LSQ solution to Q*R = b.
                [~,neff2] = size(R);
                tmp = (rhs.'*Q(:,1:neff2)).';
                u = linsolve(R(1:neff2,:),tmp,opts);
            end

            % Update the equicorrelation set
            ind = find(eq_set);
            eq_set(ind(~active_set)) = 0;
            
            % Compute the residual d = A*x-b \equiv -d
            d = -d;
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute the maximum descent time 
        % over the set where abs(A.'*d) >= 0
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        % Compute the 
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
            if(t(k) > 0)
                sol_p(:,k) = sol_p(:,k) + d/t(k);
                sol_x(eq_set,k) = vec_of_signs(eq_set).*u;
            else
                u = K\rhs;
                sol_x(eq_set,k) = vec_of_signs(eq_set).*u;
            end
            break;
            
        elseif(t(k) > 0 && timestep*t(k) > tol_minus)
            sol_x(eq_set,k) = vec_of_signs(eq_set).*u;
            sol_p(:,k) = sol_p(:,k) + d/t(k);
            break
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update dual solution and the equicorrelation set
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Update the dual solution p, Atop_times_p and vector of signs
        sol_p(:,k) = sol_p(:,k) + timestep*d;
        Atop_times_p = Atop_times_p + timestep*Atop_times_d;
        vec_of_signs = sign(-Atop_times_p);
    
        % Update the equicorrelation set
        new_eq_set = (abs(Atop_times_p) >= tol_minus); 
        
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update the QR decomposition
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        % Extract new element added to the equicorrelation set
        ind = setxor(find(eq_set),find(new_eq_set));
    
        % Update the equicorrelation set and assemble the effective matrix
        eq_set = new_eq_set;
        K = A(:,eq_set).*(vec_of_signs(eq_set).');
    
        % Check that only one element is added. If so, use qrinsert.
        % Note: Multiple columns updates are not supported.
        if(isscalar(ind))
            % Extract the column to insert
            col = A(:,ind);
    
            % Locate where to insert the column
            loc = find(find(eq_set) == ind);

            % Catch any errors due to innaccuracies -- recompute the QR
            % decomposition from scratch, then.
            if(isempty(loc))
                [Q,R] = qr(K);
                break;
            end
    
            % The following is equivalent to [Q,R] = qrinser(Q,R,loc,col);
            % Some overhead has been removed to optimize for speed.
            [~,nr] = size(R);
            R(:,loc+1:nr+1) = R(:,loc:nr);
            R(:,loc) = (col.'*Q).';
            [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
        else 
            [Q,R] = qr(K);
        end
    end


    %% Warm start + updates for the next step in the regularization path
    if(k < kmax)
        % Update the dual solution p, Atop_times_p and vector of signs
        sol_p(:,k+1) = sol_p(:,k);
        Atop_times_p = (sol_p(:,k+1).'*A).';
        vec_of_signs = sign(-Atop_times_p);
    
        % Compute the new equicorrelation set
        new_eq_set = (abs(Atop_times_p) >= tol_minus); 

        % Extract new element added to the equicorrelation set
        ind = setxor(find(eq_set),find(new_eq_set));
    
        % Update the equicorrelation set and assemble the effective matrix
        eq_set = new_eq_set;
        K = A(:,eq_set).*(vec_of_signs(eq_set).');
    
        % Check if an element is added. If so, use qrinsert.
        % Note: Multiple columns updates are not supported.
        if(isscalar(ind))
            % Extract the column to insert
            col = A(:,ind);
    
            % Locate where to insert the column
            loc = find(find(eq_set) == ind);

            % Catch any errors due to innaccuracies -- recompute the QR
            % decomposition from scratch, then.
            if(~isempty(loc))
                % The following is equivalent to [Q,R] = qrinser(Q,R,loc,col);
                % Some overhead has been removed to optimize for speed.
                [~,nr] = size(R);
                R(:,loc+1:nr+1) = R(:,loc:nr);
                R(:,loc) = (col.'*Q).';
                [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
            else
                [Q,R] = qr(K);
            end
        elseif(~isempty(ind))   % Multiple updates not supported
            [Q,R] = qr(K);
        end
    end
end

end