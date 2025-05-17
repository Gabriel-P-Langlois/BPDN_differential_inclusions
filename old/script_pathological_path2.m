% This script generates a curious example with n=2 and 2p linear segments
% (not in the icml paper)
n=2;
d=20;
eps=(pi/2)/(2*d);

y=[1 1]';
rho=sqrt(sum(y.^2));
y=y/rho;
rho=1;
theta=atan(y(2)/y(1));

X=zeros(2,d);
for ii=1:d
   rhox=1/((1+ii/d)^(1/d)*sin(ii*eps)/eps);
   thetax=theta+pi/2-ii*eps;
   X(1,ii)=rhox*cos(thetax);
   X(2,ii)=rhox*sin(thetax);
end

[path constantsegment lambdapath]=approx_lasso(X'*y,X'*X,y,X,0,0,1e-16,false);
fprintf('Problem size: n=%d, p=%d\n',n,size(path,1));
fprintf('Computed path has %d linear segmnts\n',size(path,2));
