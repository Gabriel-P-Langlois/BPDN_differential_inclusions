function [sol_x, sol_p, t] = BPDN_homotopy_solver(A,t0,p0,tol)
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
%       t       -   k-dimensional array of nonnegative hyperparameters 
%                   of the BPDN problem

%% NOTES
% Debug the code... 
% 1) Error in the estimation of x at k = 60 (OK for p).


%% Initialization
[m,n] = size(A);
tol_minus = 1-tol;
opts.UT = true;     % Option for the linsolve internal MATLAB function.
                    % We use the QR decomposition \w R upper triangular.

% Initial length of t.
% Note: It's impossible to know a priori the length of t; it is often
% observed to be close to m, but there are contorted examples where the
% length is O(e^{m}).
kmax = 5*m; 
t = length(kmax); t(1) = t0;

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


%% Compute the trajectory of the slow system in a piecewise fashion
kmax = 61;
for k=1:1:kmax
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute the NNLS problem 
        % min_{uj>=0 if j \in eq_set and xj = 0} ||K*u + t*sol_p||_{2}^{2}
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Initialize the active set
        neff = sum(eq_set);
        active_set = true(neff,1);

        % Identify indices in eq_set for which sol_x(eq_set) = 0.
        ind_nnls = find(~sol_x(eq_set,k));

        % Compute the least-squares solution.
        % Note: We use the economical forms of the matrices Q and R for this.
        rhs = -t(k)*sol_p(:,k);
        tmp = (rhs.'*Q(:,1:neff)).';
        u = linsolve(R(1:neff,:),tmp,opts);
    
        % Check if solution is admissible. 
        % If not, invoke the Method of Hinges
        % with full active set and compute its solution. 
        if(min(u(ind_nnls)) > -tol)
            tmp2 = R*u;
            xi = Q*tmp2;
            xi = xi - rhs;
        else
            % Invoke the Method of Hinges to solve the NNLS problem.
            while(true)
                remove_update = false;
                insert_update = false;
                x = zeros(neff,1);
                x(active_set) = u;

                if(min(x(ind_nnls)) <= -tol)   % Check primal constraint
                    % Find the indices in the active set to throw out.
                    tmp_zero = zeros(neff,1);
                    tmp_zero(ind_nnls) = x(ind_nnls);
                    [~,I] = min(tmp_zero);
                    active_set(I) = false;
                    remove_update = isscalar(I);
                else
                    tmp2 = R*u;
                    rho = Q*tmp2;
                    rho = rhs - rho;
                    Ktop_times_rho = (rho.'*K);
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
                    tmp_zero = zeros(neff,1);
                    tmp_zero(ind_nnls) = x(ind_nnls);
                    [~, J] = min(tmp_zero);
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
        end
        tmp2 = R*u;
        xi = Q*tmp2;
        xi = xi - rhs;
        
        x = zeros(neff,1);
        x(active_set) = u;
    
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
        tplus = t(k)/(1 + t(k)*timestep);

        % Compute tminus
        tmp_x = abs(sol_x(eq_set,k));
        ind_tminus = find(x < -tmp_x);
        if(~isempty(ind_tminus))
            tminus = t(k)*(1-min(tmp_x(ind_tminus)./abs(x(ind_tminus))));
        else
            tminus = -inf;
        end
        % Compute next time
        t(k+1) = max(tminus,tplus);
        disp([tminus,tplus,t(k),t(k+1),k])
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Check for convergence
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if(t(k+1) == 0 || k == kmax)
            sol_x(:,k+1) = sol_x(:,k);
            sol_x(eq_set,k+1) = sol_x(eq_set,k+1) + ...
                (1-t(k+1)/t(k))*(vec_of_signs(eq_set).*x);
            sol_p(:,k+1) = sol_p(:,k) + (1/t(k+1) - 1/t(k))*xi;

            t(k+2:end) = [];
            sol_x(:,k+2:end) = [];
            sol_p(:,k+2:end) = [];

            t(1) = [];
            sol_x(:,1) = [];
            sol_p(:,1) = [];
            break;
        end
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update dual solution and the equicorrelation set
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Update the primal solution x
        sol_x(:,k+1) = sol_x(:,k);
        sol_x(eq_set,k+1) = sol_x(eq_set,k+1) + ...
            (1-t(k+1)/t(k))*(vec_of_signs(eq_set).*x);

        % Update the dual solution p, Atop_times_p and vector of signs
        sol_p(:,k+1) = sol_p(:,k) + (1/t(k+1) - 1/t(k))*xi;
        Atop_times_p = Atop_times_p + (1/t(k+1) - 1/t(k))*Atop_times_d;
        vec_of_signs = sign(-Atop_times_p);

        % Update the equicorrelation set
        new_eq_set = (abs(Atop_times_p) >= tol_minus); 
        
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update the QR decomposition
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Check if we add an element      
        if(tplus > tminus)
            % Extract the element to add to the equicorrelation set
            ind = setxor(find(eq_set),find(new_eq_set));
    
            % Update the equicorrelation set and assemble the effective matrix
            eq_set = new_eq_set;
            K = A(:,eq_set).*(vec_of_signs(eq_set).');
    
            % Check that only one element is added. If so, use qrinsert.
            if(isscalar(ind))
                % Extract the column to insert
                col = A(:,ind);
            
                % Locate where to insert the column
                loc = find(find(eq_set) == ind);
        
                % Catch any errors due to innaccuracies.
                if(~isempty(loc)) % Ok -- perform column update
                    % The following is equivalent to [Q,R] = qrinser(Q,R,loc,col);
                    % Some overhead has been removed to optimize for speed.
                    [~,nr] = size(R);
                    R(:,loc+1:nr+1) = R(:,loc:nr);
                    R(:,loc) = (col.'*Q).';
                    [Q,R] = matlab.internal.math.insertCol(Q,R,loc);
                else % Not ok -- recompute the QR decomposition
                    [Q,R] = qr(K);
                end
            else % Not ok -- recompute the QR decomposition
                [Q,R] = qr(K);
            end
        
        % Remove an element
        else
            [~, J] = min(tmp_x(ind_tminus)./abs(x(ind_tminus)));
            tmp30 = find(eq_set);
            idx = tmp30(ind_tminus(J));
            
            eq_set(idx) = 0;
            K = A(:,eq_set).*(vec_of_signs(eq_set).');

            R(:,ind_tminus(J)) = [];
            [Q,R] = matlab.internal.math.deleteCol(Q,R,ind_tminus(J));
        end        
end
end