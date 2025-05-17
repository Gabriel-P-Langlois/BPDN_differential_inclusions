%%
% This package contains 
% 
% approx_lasso.m: a matlab implementation of the homotopy algorithm for solving the Lasso with its variant presented
%                 in the ICML paper. When the parameter eps equals zero, it is the exact homotopy algorithm.
%                 When eps > 0, it uses the approximate homotopy variant (only works on linux 64bits computers).
%                 Note that this implementation is designed to privilege numerical precision over speed.
%                 For faster (but less numerically stable) implementation of the homotopy, see the 
%                 software package SPAMS: http://www.di.ens.fr/willow/SPAMS/
% 
% script_pathological_path1.m: an implementation of the pathological path with exponential complexity. The above
%                 implementation works with p <=11. For p>=12, numerical precision issues appear.
% 
% script_pathological_path2.m: an interesting path example, with n=2, but 2p kinks.
% 
% script_random_path.m: a study of the number of linear segments with random data.
% 


function [path constantsegment lambdapath] = approx_lasso(Xty,XtX,y,X,eps,lambdamin,epsprecision,verbose)

p=size(XtX,2);
path=zeros(p,1);
[lambda J]=max(abs(Xty));
lambdapath=lambda;
constantsegment=[];
wtilde=zeros(p,1);
theta=1+eps/2-sqrt(eps)/2;
XtR=Xty;
iter=1;
maxeps1=eps/2;
tic
while lambda >= lambdamin+epsprecision
   t=toc;
   if verbose
      fprintf('iter: %d, lambda: %d, time: %d\n',iter,lambda,t);
   end
   iter=iter+1;  
   eta=XtR(J)/lambda;
   co=rcond(XtX(J,J));      % compute a condition number to check if the computations are reliable or not
   deltaw=XtX(J,J) \ eta;   % not optimal in terms of speed, but good for numerical precision
   if co < 1e-16 && eps==0  
      break;
   end

   deltaXtR=XtX(:,J)*deltaw;

   % choose tau
   ind=find(sign(XtR(J)) == -sign(deltaw));
   [maxtau1 ind2]=min(-wtilde(J(ind)) ./ deltaw(ind));
   if isempty(maxtau1)
      maxtau1=inf;
   end
   removeJ1=J(ind(ind2));
   Jc=setdiff(1:p,J);
   if ~isempty(Jc)
      tmp=(XtR(Jc)-lambda*(1+maxeps1)) ./ (deltaXtR(Jc) - 1-maxeps1);
      ind2=find(tmp > epsprecision);
      [maxtau2 ind]=min(tmp(ind2));
      addJ2=Jc(ind2(ind));
      tmp=(XtR(Jc)+lambda*(1+maxeps1)) ./ (deltaXtR(Jc) + 1+maxeps1);
      ind2=find(tmp > epsprecision);
      [maxtau3 ind]=min(tmp(ind2));
      addJ3=Jc(ind2(ind));
      if isempty(maxtau2)
         maxtau2=inf;
      end
      if isempty(maxtau3)
         maxtau3=inf;
      end
   else
      maxtau2=inf;
      maxtau3=inf;
   end

   tau=min([maxtau1 maxtau2 maxtau3 lambda]);

   if tau >= theta*lambda*sqrt(eps)
      lambda = lambda - tau;
      % update w 
      wtilde(J)=wtilde(J)+tau*deltaw;
      % update J 
      if tau==maxtau1
         J=setdiff(J,removeJ1);
         wtilde(removeJ1)=0;
      elseif tau==maxtau2
         J=union(J,addJ2);
      elseif tau==maxtau3
         J=union(J,addJ3);
      end
      constantsegment=[constantsegment false];
   else
      lambda=lambda*(1-theta*sqrt(eps));
      % use coordinate descent technique, update w and XtR
      param.lambda=lambda;
      param.tol=eps/2;
      param.itermax=1000000;   
      wtilde(J)=wtilde(J)+tau*deltaw; 
      J1=find(wtilde);
      XtR=Xty-XtX(:,J1)*wtilde(J1);
      [wtilde]=mexApproxCD(XtR,XtX,wtilde,param);
      % can be changed by any first-order solver
      J=find(wtilde);
      constantsegment=[constantsegment true];
   end
   J1=find(wtilde);
   XtR=Xty-XtX(:,J1)*wtilde(J1);   
   path=[path wtilde];
   lambdapath=[lambdapath lambda];
   J1=find(wtilde);
end

