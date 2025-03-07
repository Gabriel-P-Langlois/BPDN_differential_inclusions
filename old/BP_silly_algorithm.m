function [sol_x, sol_p, sol_t, x_vec, b_vec] = BP_silly_algorithm(A,b,tol,disp_output_bp)

%% Description -- BP_silly_algorithm
% We perform a homotopy descent in both the hyperparameter and data.
% For the data descent, we choose the data do that the LSQ problem is zero.
% This is why we call it silly...

% This gives a feasible point Ax=b but not equal to the true minimizer.
% Is this point, though, useful as a warm-up for the REAL basis pursuit
% algorithm?


%% Initialization
tol_minus = 1-tol;
[m,n] = size(A);
t0 = norm(-A.'*b,inf);

sol_x = zeros(n,m+1);
sol_p = zeros(m,m+1); sol_p(:,1) = -b/t0;
sol_t = zeros(1,m+1); sol_t(1) = t0;

q_vec = zeros(m,m);
w_vec = zeros(n,m);
b_vec = zeros(m,m+1);
x_vec = zeros(n,m+1);

equi_set = (abs(A.'*sol_p(:,1)) >= tol_minus);
b_vec(:,1) = b;

% Testing
num_w_with_neg = 0;


%% Algorithm -- Homotopy in the hyperparameter and data
% This descent is guaranteed to converge in at most m steps.

for i=1:1:m
    % Assemble the linear system and solve it
    D = sign(-A.'*sol_p(:,i));
    Deps = D(equi_set);
    Aeps = A(:,equi_set);

    % Compute [Aeps.'Aeps]^{-1} * 1eps  
    tmp0 = -Deps.*(pinv(Aeps)*sol_p(:,i));   

    % Silly approach: We choose w to be the largest admissible value,
    % For this choice of w, the modified descent direction r is given by
    r = sol_t(i)*(Aeps*(Deps.*tmp0) + sol_p(:,i));
    
    % Descent computation
    if(i<m)
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
    else
        sol_p(:,i+1) = r;
        w_vec(equi_set,i) = tmp0;
        q_vec(:,i) = A*(D.*w_vec(:,i));

        sol_x(:,i+1) = sol_x(:,i);
        x_vec(:,i+1) = x_vec(:,i) + sol_t(i)*(D.*w_vec(:,i));
        b_vec(:,i+1) = b_vec(:,i) - sol_t(i)*q_vec(:,i);
        
        if(min(w_vec(equi_set,i)) < -tol)
            num_w_with_neg = num_w_with_neg + 1;
        end
        break;
    end

    % Update time and dual solution
    sol_t(i+1) = sol_t(i)/(1+ timestep*sol_t(i));
    sol_p(:,i+1) = sol_p(:,i) + timestep * r;

    % Update retroactively w,q and v.
    w_vec(equi_set,i) = tmp0;
    q_vec(:,i) = A*(D.*w_vec(:,i));

    
    % Update primal variable, equicorrelation set, and other quantities
    sol_x(:,i+1) = sol_x(:,i);
    b_vec(:,i+1) = b_vec(:,i) - (sol_t(i) - sol_t(i+1))*q_vec(:,i);
    x_vec(:,i+1) = x_vec(:,i) + (sol_t(i) - sol_t(i+1))*(D.*w_vec(:,i));

    equi_set = (abs(A.'*sol_p(:,i+1)) >= tol_minus);

    % Testing
    if(min(w_vec(equi_set,i)) < -tol)
        num_w_with_neg = num_w_with_neg + 1;
    end

    % Sanity checks, if enabled.
    if(disp_output_bp)
        % disp('Sanity checks.')
        % disp(["Cone constraint norm(-Deps*Aeps.'*p,inf) <= 1: ",...
        %     num2str(norm(-Deps.*(Aeps.'*sol_p(:,i)),inf))])
        % 
        % disp(["Cone constraint norm(-Deps*Aeps.'*p,inf) >= 1: ",...
        %     num2str(min(-Deps.*(Aeps.'*sol_p(:,i))))])
        % 
        % disp(["Residual of reduced LSQ : ",num2str(norm(Aeps.'*r))])
        % 
        % disp(["Residual of tkpk = Axk - bk: ", ...
        %     num2str(norm(sol_t(i+1)*sol_p(:,i+1) - (A*sol_x(:,i+1) - b_vec(:,i+1))))])
        % 
        % disp(["Sum of w should be zero: ", num2str(sum(w_vec(:,i)))])
        % 
        % disp(["Minimum of v is >= 0: ", num2str(min(v(equi_set)))])
        % 
        % disp(["<pk,q>: ",num2str(norm(sol_p(:,i).'*q_vec(:,i)))])
        % 
        % disp(["Timestep at i = ",num2str(i), "is: ", num2str(timestep), "."])
        % disp('-----')
        % disp(' ')
    end
end

% Testing
if(disp_output_bp)
    disp(['Number of times we picked up a vector w with a negative component: ',num2str(num_w_with_neg)])
end