function Vdual = dual_obj_fun(p,t,b)
% DUAL_OBJ_FUN Computes the dual objective function of the basis pursuit
%              denoising problem:
%
%              V(p;t,b) = 0.5*t||p||_{2}^{2} + <b,p>
%   
%              *assuming* that norm(A.'*p,inf) <= 1.
kmax = length(t);
Vdual = zeros(kmax,1);
for i=1:1:kmax
    Vdual(i) = 0.5*t(i).*(norm(p(:,i),2).^2) + b.'*p(:,i);
end
end

