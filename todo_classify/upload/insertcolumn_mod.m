function [Q, R] = insertcolumn_mod(Q, R, v, k)
%INSERTCOLUMN insert a vector v as the kth column of A, return the updated
%Q, R
[m, n] = size(Q);
n = n + 1;
R = [R;zeros(1,n-1)];
R = [R zeros(n,1)];
R(:, k+1:end) = R(:, k:end-1);
u = zeros(n,1);

% Orthogonalization
u(1:n-1) = zeros(n-1,1);
if(m == n-1)
    v = zeros(m, 1);
    u(n) = 0;
else
    u(n) = norm(v); 
    rho_0 = u(n);
    count = 0; v_null = false;
    while count <= 4
        s = (v.'*Q).';
        w = Q * s;        
        v = v - w;

        rho_1 = norm(v);

        u(1:n-1) = u(1:n-1) + s;

        if rho_0  >= sqrt(2) * rho_1
            rho_0 = rho_1;
        else
            v = v ./ rho_1;
            if v_null == false
                u(n) = rho_1;
            else
                u(n) = 0;
            end
            break;
        end
        count = count + 1;
    end
end

Q = [Q, v];
for l = n-1:-1:k
    % Compute reflector
    t =  sqrt(u(l)^2+u(l+1)^2);
    if u(l) < 0
        t = -t;
    end
    c = u(l) / t;
    s = u(l+1) / t;
    u(l) = t;
    u(l+1) = 0;
    
    % Reflector 1
%     nu = s / (1 + c);
%     vx = R(l,l+1:n); 
%     R(l,l+1:n) = R(l,l+1:n)*c + R(l+1,l+1:n)*s;
%     R(l+1,l+1:n) = (R(l,l+1:n)+vx)*nu - R(l+1,l+1:n);
% 
% 
%     % Reflector 2
%     qx = Q(1:m, l);
%     Q(1:m, l) = qx*c + Q(1:m, l+1)*s;
%     Q(1:m, l+1) = (Q(1:m, l) + qx)*nu - Q(1:m, l+1);

    [R(l,l+1:n), R(l+1,l+1:n)] = applyreflector_mod(c, s, R(l,l+1:n), R(l+1,l+1:n));
    [Q(1:m, l), Q(1:m, l+1)] = applyreflector_mod(c, s, Q(1:m, l), Q(1:m, l+1));
end
R(1:k, k) = u(1:k);
end

