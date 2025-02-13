function [A,b,x_true] = generate_gaussian_data(m,n,SNR,val_nonzero,prop)
% generate_gaussian_data    Generate synthetic data with Gaussian noise
%                           Taken from ``Sparse Regression: Scalable 
%                           Algorithms and Empirical Performance" by 
%                           Dimitris Bertsimas et al. (2020).
%   Input
%       m           -   Number of samples
%       n           -   Number of features
%       SNR         -   Desired signal to noise ratio
%       val_nonzero -   Value of the nonzero vector
%       prop        -   Proportion \in (0,1) of nonzero coefficients
%   
%   Output
%       A           -   (m x n) design normalized matrix A with rows N(O,I)
%       b           -   m dimensional col of noisy observations
%       x_true      -   n dimensional col of the true signal      


% Generate design matrix
A = randn(m,n);
A = A./sqrt(sum(A.^2)/m);

% Compute ``true" solution
x_true = zeros(n,1); x_true(randsample(n,floor(n*prop))) = val_nonzero; 

% Generate the noisy observation
sigma = norm(A*x_true)/sqrt(SNR);
b = (A*x_true + sqrt(sigma)*randn(m,1));
end