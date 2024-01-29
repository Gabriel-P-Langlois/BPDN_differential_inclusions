%% Description
% Screening rule for Lasso problem
%
%   (0.5/t)*normsq(A*x-b) + normone(x)
%
% This is a slightly improved version of the rule in Tibshirani, and it's
% guaranteed to be correct.
%
% TBC.
%
% Eventually should adapt it to the problem
%
%   (0.5/t)*normsq(A*x-b) + sum_{j=1}^{n} c_{j}abs(x_j) + d_{j}x_{j}
%
function ind = lasso_screening_rule(lambda_plus,lambda,x,p,A)
    
    % Take a convex combination of the previous solution with the empirical
    % distribution, using the current and previous hyperparameters.
    beta = lambda_plus/lambda;

    % Compute the lhs and rhs of the criterion and compare them.
    lhs = abs((p.'*A).');   
    rhs = beta*(lambda_plus-(vecnorm(A,2).'*norm(p)*(lambda-lambda_plus)/(lambda)));
    
    ind = lhs >= rhs;
end