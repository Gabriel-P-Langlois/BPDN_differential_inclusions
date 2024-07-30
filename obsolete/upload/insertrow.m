function [Q, R] = insertrow(Q, R, u, k)
%INSERTROW inset nx1 vector u as the kth row of A and update Q, R
[m, n] = size(Q);
[rn, rc] = size(R);
if rn ~= n || rc ~= n
    error("The size of Q and R does not match.")
end
m = m + 1;
if length(u) ~= n
    error("The length of u is incorrect.")
end
v = zeros(m, 1);
v(k) = 1;
for l = 1:n
    for i = m-1:-1:k
        Q(i+1,l) = Q(i,l);
        
    end
    Q(k,l) = 0;
    [R(l,l), u(l), c, s] = computereflector(R(l,l), u(l));
    [R(l,l+1:n), u(l+1:n)] = applyreflector(c, s, R(l,l+1:n), u(l+1:n));
    [Q(:,l),v] = applyreflector(c, s, Q(:,l),v);
end
end

