clear all;
addpath('../build/');
format compact;

p=1000;
n=1100;
lambda1=1e-11;
epsprecision=1e-11;  % used for floating point comparisons comparisons

randn('seed',0);
rand('seed',0);
y=randn(n,1);
X=randn(n,p);

Xty=X'*y;
XtX=X'*X;  

length_path=[];
percent_constant=[];
classifperf=[];
tabeps=[0 0.5 0.25 0.1 0.01 0.001 1e-4 1e-5];
for eps=tabeps
   tic
   [path constantsegment lambdapath]=approx_lasso(Xty,XtX,y,X,eps,lambda1,epsprecision,true);
   length_path=[length_path size(path,2)];
end
length_path
