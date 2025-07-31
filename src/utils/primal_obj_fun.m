function Vprimal = primal_obj_fun(x,t,A,b)
% PRIMAL_OBJ_FUN    Computes the primal objective function of the basis
%                   pursuit denoising problem with t>0.
%
%                   (0.5/t)*norm(A*x-b)^2 + ||x||_{1}.
kmax = length(t);
Vprimal = zeros(kmax,1);
for i=1:1:kmax-1
    Vprimal(i) = (0.5/t(i))*norm(A*x(:,i) - b,2)^2 + ...
        norm(x(:,i),1);
end
Vprimal(kmax) = norm(x(:,end),1);
end