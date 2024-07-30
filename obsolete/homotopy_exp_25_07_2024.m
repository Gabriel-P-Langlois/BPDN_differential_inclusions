%% Description -- lasso_homotopy_solver_opt
% Exact homotopy algorithm -- Written by Gabriel Provencher Langlois

% Error in timestep_1 was found.
% 
% It seems that we have -D*A.'*pk = 1 and D*A.'*d =\= 0.
% Print the index that is wrong and then see what can be done.
%
% Maybe there is some zero x for which Atop*p is equal to + or - 1?


function [sol_x,sol_p,sol_t] = ...
    homotopy_exp_25_07_2024(A,b,m,n,tol,display_iterations,run_diagnostics)


%% Initialization
rng('default')
tol_minus = 1-tol;

% Compute t0 and the variable moving_term_1 (= -A.'*p(t0))
bminus = -b;
Atop_times_pk = (bminus.'*A).';
t0 = norm(Atop_times_pk,inf);
Atop_times_pk = Atop_times_pk/t0;

% Compute initial equicorrelation set
equi_set = (abs(Atop_times_pk) >= tol_minus); 

% Create solution paths + initialize
buffer_path = 10*m;
sol_p = zeros(m,buffer_path); 
sol_t = zeros(1,buffer_path);
sol_x = zeros(n,buffer_path);

sol_t(1) = t0;
sol_p(:,1) = bminus/t0;


%% Algorithm
shall_continue = true;
k = 1;
while(shall_continue && k <= buffer_path)
    %%%%% 1. Compute the relevant vectors and matrices
    vector_of_signs = sign(-Atop_times_pk);
    D = diag(vector_of_signs(equi_set));
    K = A(:,equi_set);
    

    %%%%% 2. Solve the LSQ problem
    flag = true;

    while(flag)
        u = (K*D)\b;

        % Retrieve components of the lsq solution that are negative
        ind_u_neg = find(u<-tol); uhat = u(ind_u_neg);

        % Retrieve indices j \in \eps(pk) where uhat_j < 0.
        ind_equi_set = find(equi_set); my_ind = ind_equi_set(ind_u_neg);

        % Retrieve corresponding utilde
        utilde = vector_of_signs(my_ind).*sol_x(my_ind,k);

        % Remove indices violating the other condition
        if(any(abs(utilde) < tol))
            equi_set(my_ind(find(abs(utilde) < tol))) = 0;
            D = diag(vector_of_signs(equi_set));
            K = A(:,equi_set);
        else
            flag = false;
        end
    end
    % u = (K*D)\b;
    % ind_u_neg = find(u<-tol); uhat = u(ind_u_neg);
    % ind_equi_set = find(equi_set); my_ind = ind_equi_set(ind_u_neg);
    % utilde = vector_of_signs(my_ind).*sol_x(my_ind,k);


    %%%%% 3. Compute the descent direction
    v = zeros(n,1); v(equi_set) = D*u;
    d = K*(D*u) - b;

    % Compute the variable moving_term_2 (= A.'*d)
    Atop_times_dk = (d.'*A).'; 

    % Compute the set of descent directions (abs(<d,Aej>) > 0)
    active_set_plus = abs(Atop_times_dk) > tol;
    

    %%%%% 4. Compute the kick time
    %%% This is the `smallest' amount of time we can slide down
    % over the possible set of directions before changing the active set.

    % Check if the least square problem returned a solution
    % with negative coefficient.
    if(any(u<-tol))        
        % Compute smallest t for which the convex combination of
        % quantities are zero. Here, tc = factor*tk
        factor = (-uhat./(utilde - uhat));
        tc = max(factor)*sol_t(k);
        timestep_1 = (1/tc - 1/sol_t(k));
    end


    % Check how far we have to go before adding a new index
    if (any(active_set_plus))

        term1 = diag(vector_of_signs)*Atop_times_dk;
        sign_coeffs = sign(term1);
        term2 = -diag(vector_of_signs)*Atop_times_pk;

        vec = (1+sign_coeffs(active_set_plus).*term2(active_set_plus))./...
            abs(term1(active_set_plus));

        timestep_2 = min(vec);

        % if((k == 40) || (k == 44))
        %     disp(term2(active_set_plus)) % Compute indices for which 1 is achieved
        %     disp(u)
        %     disp(D*K.'*d)
        %     disp(abs(D*K.'*d) - tol)
        %     disp(vec)
        % end
    end

    if(any(active_set_plus) && abs(timestep_2) < tol)
        equi_set = (abs(Atop_times_pk) >= tol_minus); 
    else

        % Decide on the timestep to use and update the equicorrelation set
        if(any(u<-tol) && any(active_set_plus))
            timestep = min(timestep_1,timestep_2);
        elseif(any(u<-tol))
            timestep = timestep_1;
        elseif(any(active_set_plus))
            timestep = timestep_2;
        else
            % Compute final solutions
            sol_x(:,k+1) = v; 
            sol_p(:,k+1) = sol_p(:,k);
    
            % Truncate the path and exit
            sol_x(:,k+2:end) = [];
            sol_p(:,k+2:end) = [];
            sol_t(:,k+2:end) = [];
            break;
        end
    
    
        %%%%% 5. Update all solutions and variables
        % Update the dual solution + time parameter
        sol_p(:,k+1) = sol_p(:,k) + timestep*d;
        sol_t(k+1) = sol_t(k)/(1+sol_t(k)*timestep);
    
        % Update the primal solution
        alpha = sol_t(k+1)/sol_t(k);
        sol_x(:,k+1) = alpha*sol_x(:,k) + (1-alpha)*v; 
    
    
        % Compute next equicorrelation set
        Atop_times_pk = Atop_times_pk + timestep*Atop_times_dk;
        if(any(u<-tol) && any(active_set_plus))
            if(timestep == timestep_1)
                equi_set(my_ind) = 0;
            else
                equi_set = (abs(Atop_times_pk) >= tol_minus); 
            end
        elseif(any(u<-tol))
            equi_set(my_ind) = 0;
        else
            equi_set = (abs(Atop_times_pk) >= tol_minus); 
        end

    

        %%%%% 6. Optionals
        % If enabled, display current iteration
        if(display_iterations)
            disp(['Iteration ',num2str(k),' is complete.'])
        end
    
        % If enable, run diagnostics
        if(run_diagnostics)
            disp(['Iteration: ', num2str(k),' has timestep = ',num2str(timestep)])
            disp(' ')
            % Compute the residual norm of the NNLS problem
            % min_{u >= 0} ||K*D*u - (b+tk*pk)||^2
            [~,quantity_1,~] = lsqnonneg(K*D,b + sol_t(k)*sol_p(:,k));
            disp(['Diagnostic I: Computing min_{u >= 0} ||K*D*u - (b+tk*pk)||^2: ',num2str(quantity_1)]);
    
    
            if(quantity_1 > tol)
                %disp(u)
            end
    
            % Compute dual vector between pk and pk+1 at different points
            % disp(' ')
            % disp('Diagnostics II: ')
            % for h = 0.05:0.1:0.95
            %     tmp_p = sol_p(:,k) + (h*timestep)*d;
            %     tmp_t = exact_path(k)/(1 + exact_path(k)*(h*timestep));
            %     [~,tmp2,~] = lsqnonneg(K*D,b + tmp_t*tmp_p);
            %     disp(num2str(tmp2))
            % end
            % disp(' ')
    
            % Display when the solution is strictly positive
            if(any(u < -tol))
                disp('Diagnostic II: At least one component of the NNLS problem is zero.')
                % % Further tests
                % disp(['Current k: ',num2str(k)])
                % 
                % disp('Components of u')
                % disp(u)
                % 
                % disp('Corresponding indices that are nonzero')
                % ind_equi_set = find(equicorrelation_set);
                % disp(find(equicorrelation_set))
                % 
                % disp('D*u')
                % disp(v(equicorrelation_set))
                % 
                % disp('Components of x with |<-Atop(pk,ej>| = 1')
                % disp(sol_x(equicorrelation_set,k))
                % 
                % disp('<-Atop*pk>')
                % tmp = -A.'*sol_p(:,k);
                % disp(tmp(equicorrelation_set))
                % 
                % disp('Atop*d')
                % disp(A(:,equicorrelation_set).'*d)
                % 
                % % Note: Could it be that the uhat computed is the wrong one?
                % % This seems to be the case...
                % 
                % % Below we compute the ``correct u" for our problem.
                % % 1) Remove from the equicorrelation set the ``faulty index"
                % zero_ind = find(u == 0);
                % disp(find(u == 0))
                % disp(ind_equi_set(find(u == 0)))
                % 
                % new_equicorrelation_set = equicorrelation_set;
                % new_equicorrelation_set(ind_equi_set(find(u == 0))) = 0;
                % disp(find(new_equicorrelation_set))
                % 
                % % 2) Compute the lsq problem ||Ae(new)D(enew)*u - b ||_{2}^{2}
                % %    and check that it gives the same direction as before.
                % vector_of_signs = sign(-A.'*sol_p(:,k));
                % new_D = diag(vector_of_signs(new_equicorrelation_set));
                % new_K = A(:,new_equicorrelation_set);
                % K*(D*u)
                % new_u = lsqnonneg(new_K*new_D,b);
                % new_d = new_K*new_D*new_u - b;
                % disp('Display the norm between new d from lower equicorrelation_set')
                % disp(norm(new_d-d))
                % 
                % disp('Display u and new_u, starting with u first')
                % disp(u)
                % disp(new_u)
                % disp('Atop*new_d')
                % disp(A(:,new_equicorrelation_set).'*d) % SUCCESS!
                % 
                %break;
            else
                disp('Diagnostic II: All components of the NNLS are strictly positive.')
            end
            disp(' ')
            disp(' ')
        end
    
        % Increment
        k=k+1;
    end
end
end