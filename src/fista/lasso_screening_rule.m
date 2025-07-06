function ind = lasso_screening_rule(ratio,p,vecA2,Atimesp)
% lasso_screening_rule      Computes a set of indices {1,\dots,n} over 
%                           which the solution is guaranteed to land on.
%                           This uses an improved version of the LASSO
%                           screening rule from Eq. 7 of the paper "Strong 
%                           rules for  discarding predictors in lasso-type 
%                           problems" by Tibshirani et al. (2012).
%
%                           Note: The selection rule is adjusted to the
%                           problem min_{x} {0.5/t||Ax-b||^2_2 + ||x||_1 
%
%   Input
%       ratio   -   ratio of tnext/t positive hyperparameters
%       p       -   m dimensional col vector. Previous dual solution at
%                   parameter t.
%       vecA2   -   1 by n vector vecnorm(A,2)
%       Atimesp -   A.'*p;
%
%   Output
%       ind     -   n-dim col vector of boolean values. An entry is true
%                   if the next x solution can be nonnegative. An entry
%                   of false guarantees the solution is zero.


% Compute the lhs and rhs of the criterion and compare them.
lhs = abs(Atimesp) - 1e-08;   
tmp2 = norm(p)*(1-ratio)/ratio;
rhs = ratio*ratio * (1- vecA2*tmp2);
    
ind = lhs >= rhs.';
end