% This script generates a pathological example with (3^p+1)/2 linear segments
% The homotopy implementation is not designed to be fast, but to be numerically stable.

% GPL: This will construct the matrix A and b. Excellent.

%% Original parameters
format compact;
% p=11;       % should be small since the path has exponential size 
%             %(it gives the correct path for p <= 11, but fails because of numerical precision for p>=12)
% 
% y=1;
% X=1;
% eps=0.99;
% epsprecision=1e-16;
% lambdamin=0;


%% Modified parameters
p=5;       % should be small since the path has exponential size 
            %(it gives the correct path for p <= 11, but fails because of numerical precision for p>=12)

b=1;
A=1;
eps=0.99;
epsprecision=1e-16;
lambdamin=0;


%% Algorithm
for ii = 1:p;
   [path constantsegment lambdapath]=approx_lasso(A'*b,A'*A,b,A,0,lambdamin,epsprecision,false);
   if (ii ~= p)
      lambda1=lambdapath(end-1);
      w=path(:,end-1);
      yn=1;
      alpha=eps*lambda1/(2*b'*(b-A*w)+yn^2);
      beta=2*alpha;
      A=[A, beta*b;
      zeros(1,size(A,2)), alpha*yn];
      b=[b; yn];
   end
end
fprintf('Problem size: n=%d, p=%d\n',size(path,1),size(path,1));
fprintf('Computed path has %d linear segmnts\n',size(path,2));
fprintf('Theoretical path has %d linear segments\n',(3^p+1)/2);
