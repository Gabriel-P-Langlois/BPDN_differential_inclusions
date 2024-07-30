%% Description -- lasso_homotopy_solver_opt
% Exact homotopy algorithm -- Written by Gabriel Provencher Langlois


%% NOTES


%%
function [sol_x,sol_p,exact_path] = ...
    lasso_solver_homotopy_debug(A,b,m,n,tol,display_iterations,run_diagnostics)


%% Initialization
rng('default')
tol_minus = 1-tol;

% Compute t0 and the variable moving_term_1 (= -A.'*p(t0))
bminus = -b;
Atop_times_pk = (bminus.'*A).';
t0 = norm(Atop_times_pk,inf);
Atop_times_pk = Atop_times_pk/t0;

% Create solution paths + initialize
buffer_path = 5*m;
sol_p = zeros(m,buffer_path); 
exact_path = zeros(1,buffer_path);
sol_x = zeros(n,buffer_path);

exact_path(1) = t0;
sol_p(:,1) = bminus/t0;


%% Algorithm
shall_continue = true;
k = 1;

while(shall_continue && k <= buffer_path)
    %%%%% 1. Compute the equicorrelation set to select the active columns,
    % Note: We also compute the relevant signs and store them
    equicorrelation_set = (abs(Atop_times_pk) >= tol_minus); 
    vector_of_signs = sign(-Atop_times_pk);



    D = diag(vector_of_signs(equicorrelation_set));
    K = A(:,equicorrelation_set);
    

    %%%%% 2. Solve the NNLS problem + store the solution u = D*v
    u = lsqnonneg(K*D,b);
    v = zeros(n,1); v(equicorrelation_set) = D*u;


    %%%%% 3. Compute the descent direction
    d = (K*v(equicorrelation_set) + bminus);

    % Compute the variable moving_term_2 (= A.'*d)
    moving_term_2 = (d.'*A).'; 

    % Compute the set of descent directions (abs(<d,Aej>) > 0)
    active_set_plus = abs(moving_term_2) > tol;
    
   

    %%%%% 4. Compute the kick time
    % Note 1: This is the `smallest' amount of time we can slide down
    % over the possible set of directions before changing the active set.

    % Note 2: If there are no possible set of directions, then 
    % timestep = inf and we have converged.
    if (any(active_set_plus))
        term1 = diag(vector_of_signs)*moving_term_2;
        sign_coeffs = sign(term1);

        term2 = -diag(vector_of_signs)*Atop_times_pk;

        vec = (1+sign_coeffs(active_set_plus).*term2(active_set_plus))./...
            abs(term1(active_set_plus));

        timestep = min(vec);
    else
        sol_x(:,k+1) = v; 
        break;
    end

    
    %% TEST
    if(any(u <= tol))
        ind = find(u <= tol);
        Ktest = K;  Ktest(:,ind) = [];
        tmp3 = vector_of_signs(equicorrelation_set); tmp3(ind) = [];
        Dtest = diag(tmp3);

        utest = lsqnonneg(Ktest*Dtest,b);
        dtest = Ktest*(Dtest*utest) - b;
        disp(norm(d-dtest)) % Identical...

        moving_term_test = (dtest.'*A).';
        active_set_test = abs(moving_term_test) > tol;
        
        if any(active_set_test)
            term1 = diag(vector_of_signs)*moving_term_test;
            sign_coeffs = sign(term1);
            term2 = -diag(vector_of_signs)*Atop_times_pk;

            vec = (1+sign_coeffs(active_set_test).*term2(active_set_test))./...
                abs(term1(active_set_test));

        timestep_test = min(vec);
        else
            break;
        end

        %%
        disp('------')
        disp(abs(timestep-timestep_test)) % still identical
        disp('------')
    end


    %%%%% 5. Update all solutions and variables
    % Update the dual solution + time parameter
    sol_p(:,k+1) = sol_p(:,k) + timestep*d;
    exact_path(k+1) = exact_path(k)/(1+exact_path(k)*timestep);

    % Update the variable moving_term_1 (= -A.'*p(exact_path(i + 1)))
    Atop_times_pk = Atop_times_pk + timestep*moving_term_2;

    % Update the primal solution
    alpha = exact_path(k+1)/exact_path(k);
    sol_x(:,k+1) = alpha*sol_x(:,k) + (1-alpha)*v; 

    %%%%% 6. Optionals
    % If enabled, display current iteration
    if(display_iterations)
        disp(['Iteration ',num2str(k),' is complete.'])
    end

    % If enable, run diagnostics
    if(run_diagnostics)
        disp(['Iteration: ', num2str(k)])
        disp(' ')
        % Compute the residual norm of the NNLS problem
        % min_{u >= 0} ||K*D*u - (b+tk*pk)||^2
        [~,quantity_1,~] = lsqnonneg(K*D,b + exact_path(k)*sol_p(:,k));
        disp(['Diagnostic I: Computing min_{u >= 0} ||K*D*u - (b+tk*pk)||^2: ',num2str(quantity_1)]);

        for h = [0.1,0.5,0.9,1]
            tmp = (1-h)*exact_path(k)*sol_p(:,k) + h*d;
            [~,tmp2,~] = lsqnonneg(K*D,b + tmp);
            disp(num2str(tmp2))
        end
        disp(' ')

        htest = 0.99;
        tmp3 = sol_p(:,k) + htest*timestep*d;
        tmp4 = exact_path(k)/(1+exact_path(k)*htest*timestep);
        tmp5 = A.'*tmp3;

        equitest = abs(tmp5) >= tol_minus;

        vector_of_signs = sign(-tmp5);
        Dtest = diag(vector_of_signs(equitest));
        Ktest = A(:,equitest);
        [~,quantity2,~] = lsqnonneg(Ktest*Dtest,b + tmp4*tmp3);
        disp(quantity2)

        % Display when the solution is strictly positive
        if(any(u <= tol))
            disp('Diagnostic II: At least one component of the NNLS problem is zero.')
        else
            disp('Diagnostic II: All components of the NNLS are strictly positive.')
        end
        
        % 

        disp(' ')
        disp(' ')
    end

    % Increment
    k=k+1;
end
end