function [A,b,xsol] = ...
    generate_correlated_toeplitz_data(m,n,rho,SNR,nb_nnz)
%GENERATE_CORRELATED_TOEPLITZ_DATA
%                           Generate structured, synthetic data as per
%                           the methodology of Sections 3.1 and 3.3 from
%                           ``Sparse Regression: Scalable 
%                           Algorithms and Empirical Performance" by 
%                           Dimitris Bertsimas et al. (2020).
%   Input
%       m           -   Number of samples
%       n           -   Number of features
%       rho         -   Correlation coefficient of the Toeplitz matrix used
%                   -   in generating the samples (columns of the matrix A)
%       SNR         -   Desired signal to noise ratio
%       nnb_nnz -   Number of nonzero coefficients desired
%   
%   Output
%       A           -   (m x n) design normalized matrix A with rows N(O,E)
%                   -   where E = (rho^{|i-j|})_{i,j}.
%       b           -   m dimensional col of noisy observations
%       x_true      -   n dimensional col of the true signal      

% Check if rho \in [0,1].
if(rho < 0 || rho > 1)
    error("The number rho must be in [0,1]. Exiting...")
end

if(nb_nnz < 0 || nb_nnz > n)
    error("The number of desired nonzero components of the " + ...
        "true solution is either less than 0 or exceed n. Exiting...")
end

% Compute the first row of the matrix.
row = (rho*ones(1,m)).^(0:1:m-1);

% Compute the toeplitz matrix using MATLAB's native toeplitz function.
T = toeplitz(row); R = chol(T);

% Generate samples.
A = R*randn(m,n);
clear T R

% Compute ``true" solution
xsol = zeros(n,1); 
population = [-1,0,1];
ind = randsample(1:n,nb_nnz);
xsol(ind) = randsample(population,length(ind),true);

% Generate the noisy observation
sigma = norm(A*xsol)/sqrt(SNR);
b = (A*xsol + sqrt(sigma)*randn(m,1));
end