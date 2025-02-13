function ind = lasso_screening_rule(t_next,t,p,A)
% lasso_screening_rule      Computes a set of indices {1,\dots,n} over 
%                           which the solution is guaranteed to land on.
%                           This uses an improved version of the LASSO
%                           screening rule from Eq. 7 of the paper "Strong 
%                           rules for  discarding predictors in lasso-type 
%                           problems" by Tibshirani et al. (2012).
%
%   Input
%       t_next  -   positive hyperparameter of the next LASSO solution.
%       t       -   positive hyperparameter of the previous LASSO solution.
%       p       -   m dimensional col vector. Previous dual solution at
%                   parameter t.
%       A       -   m by n design matrix of the LASSO problem.

% Take a convex combination of the previous solution with the empirical
% distribution, using the current and previous hyperparameters.
beta = t_next/t;

% Compute the lhs and rhs of the criterion and compare them.
lhs = abs((p.'*A).');   
rhs = beta*(t_next-(vecnorm(A,2).'*norm(p)*(1-beta)));
    
ind = lhs >= rhs;
end