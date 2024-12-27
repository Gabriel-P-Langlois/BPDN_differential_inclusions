function [sol_x, sol_p, sol_t, xtot, btot] = BP_test_algorithm(A,b,tol,disp_output_bp)
%% Description -- BP_test_algorithm
% This is some code to test a tentative algorithm for the BP problem
% NOTE: The first version of this code will test 
%       for convergence in m iterations.


%% Initialization
tol_minus = 1-tol;
[m,n] = size(A);
t0 = norm(-A.'*b,inf);

sol_x = zeros(n,m+1);
sol_p = zeros(m,m+1); sol_p(:,1) = -b/t0;
sol_t = zeros(1,m+1); sol_t(1) = t0;

equi_set = (abs(A.'*sol_p(:,1)) >= tol_minus);

xtot = sol_x(:,1);
btot = b;


%% Algorithm
for i=1:1:100
    % Assemble the linear system and solve it
    D = sign(-A.'*sol_p(:,i));
    Deps = D(equi_set);
    Aeps = A(:,equi_set);

    % Compute [Aeps.'Aeps]^{-1} Aeps.'* 1eps  
    tmp0 = -Deps.*(pinv(Aeps)*sol_p(:,i));   
    w = zeros(n,1); w(equi_set) = abs(min(0,tmp0));
    v = zeros(n,1); v(equi_set) = sol_t(i)*(tmp0 - w(equi_set));
    q = A*(D.*w);

    % Compute modified descent direction
    r = Aeps*(Deps.*v(equi_set)) + sol_t(i)*(q + sol_p(:,i));

    % Compute the set of ``admissible faces"
    tmp1 = D.*(A.'*sol_p(:,i));
    tmp2 = D.*(A.'*r);
    active_set_plus = (abs(tmp2) > tol);
    timestep = inf;

    % Compute the kick time
    if(any(active_set_plus))
        vec = (1 - sign(tmp2(active_set_plus)).*tmp1(active_set_plus))./...
            abs(tmp2(active_set_plus));
        timestep = min(vec);
    end

    % Check for convergence. If true, update and stop.
    if(isinf(timestep))
        sol_x(:,i+1) = sol_x(:,i) + D.*v;
        xtot = xtot + sol_t(i)*(D.*w);
        btot = btot - sol_t(i)*q;

        if(disp_output_bp)
            disp(norm(A*sol_x(:,i+1) - btot))
            disp('Done -- Absolute cinema!')
        end
        break;
    end

    % Update vectors and quantities
    sol_t(i+1) = sol_t(i)/(1+ timestep*sol_t(i));
    sol_p(:,i+1) = sol_p(:,i) + timestep * r;
    equi_set = (abs(A.'*sol_p(:,i+1)) >= tol_minus);

    alpha = sol_t(i+1)/sol_t(i);
    sol_x(:,i+1) = sol_x(:,i) + (1 - alpha)*(D.*v);
    
    xtot = xtot + (sol_t(i) - sol_t(i+1))*(D.*w);
    btot = btot - (sol_t(i) - sol_t(i+1))*q;


    % Sanity checks, if enabled.
    if(disp_output_bp)
        disp('Sanity checks.')
        disp(["Cone constraint norm(-Deps*Aeps.'*p,inf) <= 1: ",...
            num2str(norm(-Deps.*(Aeps.'*sol_p(:,i)),inf))])

        disp(["Cone constraint norm(-Deps*Aeps.'*p,inf) >= 1: ",...
            num2str(min(-Deps.*(Aeps.'*sol_p(:,i))))])

        disp(["Residual of reduced LSQ : ",num2str(norm(Aeps.'*r))])

        disp(["Residual of tkpk = Axk - bk: ", ...
            num2str(norm(sol_t(i+1)*sol_p(:,i+1) - (A*sol_x(:,i+1) - btot)))])

        disp(["Timestep at i = ",num2str(i), "is: ", num2str(timestep), "."])
        disp('-----')
        disp(' ')
    end

end

% Compute initial equicorrelation set
% Atop_times_pk = Atop_times_bminus/t0;   % Used to speed up the computations
% equi_set = (abs(Atop_times_pk) >= tol_minus); 

% 
% 
% %% Algorithm
% % Run the algorithm until either convergence is achieved or the maximum
% % number of iterations permitted is reached.
% shall_continue = true;
% k = 1;
% while(shall_continue && k <= max_iter)
%     %%%%% 1. Compute the relevant vectors and matrices
%     K = A(:,equi_set);
%     all_signs = sign(-Atop_times_pk);
%     Dvec = all_signs(equi_set);
% 
%     %%%%% 2. Solve the LSQ problem
%     while(true)
%         u = (K.*Dvec.')\b;
% 
%         % Retrieve components of the LSQ solution that are negative, and
%         % retrieve the corresponding indices in the equicorrelation set.
%         ind_u_neg = find(u<-tol); 
%         uhat = u(ind_u_neg);
% 
%         ind_equi_set = find(equi_set); 
%         my_ind = ind_equi_set(ind_u_neg);
% 
%         % Retrieve corresponding utilde
%         utilde = all_signs(my_ind).*sol_x(my_ind,k);
% 
%         % Remove indices violating the other condition
%         if(any(abs(utilde) < tol))
%             equi_set(my_ind(find(abs(utilde) < tol))) = 0;
%             Dvec = all_signs(equi_set);
%             K = A(:,equi_set);
%         else
%             break
%         end
%     end
% 
%     %%%%% 3. Compute the descent direction
%     v = zeros(n,1); v(equi_set) = Dvec.*u;
%     d = K*(Dvec.*u) + bminus;
% 
%     % Compute A.'*d and the set of indices abs(A.'*d) > 0
%     Atop_times_dk = (d.'*A).'; 
%     active_set_plus = abs(Atop_times_dk) > tol;
% 
% 
%     %%%%% 4. Compute the kick time
%     %%% 4.1. Check if the least square problem returned a solution
%     % with negative coefficient.
%     if(any(u<-tol))        
%         % Compute smallest t for which the convex combination of
%         % quantities are zero. Here, tc = factor*tk
%         factor = (-uhat./(utilde - uhat));
%         tc = max(factor)*sol_t(k);
%         timestep_1 = (1/tc - 1/sol_t(k));
%     end
% 
% 
%     %%% 4.2 Compute thee `smallest' amount of time we can descent over
%     % the set of admissible faces. If the set of indices abs(A.'*d) > 0
%     % is empty, then timestep = +\infty
%     if (any(active_set_plus))
%         term1 = all_signs.*Atop_times_dk;
%         sign_coeffs = sign(term1);
%         term2 = -all_signs.*Atop_times_pk;
% 
%         vec = (1+sign_coeffs(active_set_plus).*term2(active_set_plus))./...
%             abs(term1(active_set_plus));
% 
%         timestep_2 = min(vec);
%     end
% 
%     % Decide on the timestep to use and update the equicorrelation set
%     if(any(u<-tol) && any(active_set_plus))
%         timestep = min(timestep_1,timestep_2);
%     elseif(any(u<-tol))
%         timestep = timestep_1;
%     elseif(any(active_set_plus))
%         timestep = timestep_2;
%     else
%         % Convergence achieved --  compute final solutions
%         sol_x(:,k+1) = v; 
%         sol_p(:,k+1) = sol_p(:,k);
% 
%         % Truncate the path and exit
%         sol_x(:,k+2:end) = [];
%         sol_p(:,k+2:end) = [];
%         sol_t(:,k+2:end) = [];
%         break;
%     end
% 
% 
%     %%%%% 5. Update all solutions and variables
%     % Update the dual solution + time parameter
%     sol_p(:,k+1) = sol_p(:,k) + timestep*d;
%     sol_t(k+1) = sol_t(k)/(1+sol_t(k)*timestep);
% 
%     % Update the primal solution
%     alpha = sol_t(k+1)/sol_t(k);
%     sol_x(:,k+1) = alpha*sol_x(:,k) + (1-alpha)*v; 
% 
%     % Compute next equicorrelation set
%     Atop_times_pk = Atop_times_pk + timestep*Atop_times_dk;
%     equi_set = (abs(Atop_times_pk) >= tol_minus); 
% 
% 
%     %%%%% 6. (Optional) Display information about the current iteration
%     % If enabled, display the current iteration and the timestep
%     % used in the calculation.
%     if(disp_output_bp)
%         disp(['Iteration: ', num2str(k),' has t = ',num2str(sol_t(k)) ' and timestep = ',num2str(timestep)])
%     end
% 
%     % If enabled, display some diagnostics
%     % WARNING: Enabling this may considerably slow down the algorithm
%     if(run_opt_cond_checks)
%         % Compute the residual norm of the NNLS problem
%         % min_{u >= 0} ||K*D*u - (b+tk*pk)||^2
%         [~,quantity_1,~] = lsqnonneg(K.*Dvec.',b + sol_t(k)*sol_p(:,k));
% 
%         if(disp_output_bp)
%             disp(['Diagnostic I: Computing min_{u >= 0} ||K*D*u - (b+tk*pk)||^2: ',num2str(quantity_1)]);
%         end
% 
%         if(quantity_1 < tol)
%             count_opt_cond_satisfied = count_opt_cond_satisfied + 1;
%         end
%     end
% 
%     % Add a space if anything was printed
%     if(disp_output_bp)
%         disp(' ') 
%     end
% 
%     %%%%% 7. Increment
%     k=k+1;
% end
% % If enable, count the number of times the residual in run_opt_cond_checks
% % is satisfied.
% if(run_opt_cond_checks)
%     disp(['Number of times the residual norm ||K*D*u - (b+tk*pk)||^2 is < tol: ',num2str(count_opt_cond_satisfied),' out of k ',num2str(k-1),' iterations.'])
% end
% 
% % Display warning if max_iter has been reached
% % Note: May possibly be reached for some contorted matrix A.
% if(k >= max_iter)
%     disp('Warning: Maximum number of iterations reached!')
% end
% end