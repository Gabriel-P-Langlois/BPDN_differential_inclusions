%% runall script
% Invoke this script from the main directory by calling
%     run ./results/gaussian_greedy/runall.m


%% Initialization
% Nb of samples and features
m = 100;
n = 200;

% Signal-to-noise ratio, value of nonzero coefficients, and
% proportion of nonzero coefficients in the signal.
SNR = 1;
val_nonzero = 1;
prop = 0.05;

% Tolerance levels
tol = 1e-08;
tol_glmnet = 1e-08;
tol_fista = tol;


%% Generate data
% Set random seed and generate Gaussian data
rng('default')
[A,b] = generate_gaussian_data(m,n,SNR,val_nonzero,prop);


% Calculate the smallest hyperparameter for which we have the trivial
% primal solution (zero) and dual solution -b/t0
t0 = norm(A.'*b,inf);
x0 = zeros(n,1);
p0 = -b/t0;


%% Run 1: Direct Basis Pursuit (BP) Solver
disp(' ')
disp('1. Running the differential inclusions Basis Pursuit solver...')
tic
[sol_inclBP_x, sol_inclBP_p, bp_count] = BP_inclusions_solver(A,b,p0,tol);
time_inclBP_alg = toc;
disp(['Done. Total time = ', num2str(time_inclBP_alg), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(bp_count)])
disp(' ')


%% Run 2: Greedy homotopy algorithm via the matroid property
% Note: This computes sol_g_xf and sol_g_eqset such that
%    A*sol_g_xf(sol_g_eqset) = b
disp(' ')
disp('2. Running the greedy algorithm \w thresholding.')
tic
[sol_g_x, sol_g_p, sol_g_b, sol_g_xf, sol_g_eqset] = ...
    greedy_homotopy_threshold(A,b,tol);
time_greedy_alg = toc;
disp(['Done. Total time = ', num2str(time_greedy_alg), ' seconds.'])
disp(' ')

% Note: The equicorrelation set we get from sol_g_p can be used to obtain
% sol_g_xf by solving A(:,sol_g_eqset)\b;

%% Run 3: Use the dual greedy solution as a ``warm start" for the BP solver
disp(' ')
disp(['3. Running the BP solver using the dual solution' ...
    ' obtained from the greedy algorithm as a warm start.'])
tic
[~,~, warm_count] = ...
    BP_inclusions_solver(A,b,sol_g_p(:,end),tol);
time_warm = toc;
disp(['Done. Total time = ', num2str(time_warm), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(warm_count)])
disp(' ')


%% Run 4: Use the greedy dual solution and a change of variable
disp(' ')
disp(['4. Running the BP solver using the dual greedy solution' ...
    ' with a change of variable as a warm start'])
tic

% Compute the feasible point and the equicorrelation set
b_new = -sol_g_xf(sol_g_eqset);
p_new = -(A(:,sol_g_eqset).')*sol_g_p(:,end);
A_new = -inv(A(:,sol_g_eqset))*A;


% Invoke the bp solver on the modified problem
[sol_test_x, sol_test_p, test_count] = ...
    BP_inclusions_solver(A_new,b_new,p_new,tol);
time_new = toc;

% Invert the solution to get the corresponding vector p.
% Check that it is the same as the dual solution
sol_test_p = -inv(A(:,sol_g_eqset).')*sol_test_p;
disp(['Done. Total time = ', num2str(time_new), ' seconds.'])
disp(['Total number of NNLS solves: ', num2str(test_count)])
disp(['Furthermore, norm(sol_test_p-sol_inclBP_p) = ', ...
    num2str(norm(sol_test_p-sol_inclBP_p))])
disp(' ')


%% Set summarize flag to true
summarize_greedy = true;




%% NOTES
% 1) As expected, positive components of w --> component of x is zero.
% 2) The t*res = Axk - bk-1 factorization does NOT work
% 3) A stepwise approach to solving the BP problem with bK, bK-1,... b
%    seems to take MORE steps than just plain BP denoising. Hmmm.
%
% 4) Something interesting happens at m = n = 100. Using the warm start
%    sol_p(:,end) does not help much -- instead of 214 iterations for the 
%    BP inclusion solver, it takes 125 iterations for the warm start.
%    Sadly, this is the case even though the feasible point obtained
%    from the greedy system is the correct solution!
%
%   Possibility: Given sol_g_p(:,end), can we do a transformation or 
%   modify it to get a better start to the basis pursuit problem?
%
% 5) There's something fishy going on with the optimality condition
%    -Atop*p \in \partial norm{x}_{1}. Well, it works! But when xk = 0
%    and sol_g_y picks up a term, it ends up being the opposite sign
%   than what it should be.
%
% 6) After using the greedy algorithm, one can, in principle, reduce both
% the l1 norm of xk + norm of sol_g_y. Try this.
% sign(sol_g_x) = sign(sol_g_y). Hmmm.

% 6) Sequential thresholding does not work well either, it seems.
% HOWEVER: What we must check is also the number of solves WITHIN the 
% hinge algorithm. Update: The number of NNLs are low in all cases, but
% nothing special occurs. Hmmm... Nothing discernible... 
%
% 7) The idea I had to use the Bregman distance, perform the change
% of variable z = -A(:,eqset).'*sol_g_p(:,end)... works? I think the
% number of solves correspond to how sol_g_x and sol_g_y are divided...
% 