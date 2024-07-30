function [Q, R] = deleterow(Q, R, k)
%DELETEROW delete the kth row of A and update Q, R
[m, n] = size(Q);
[rn, rc] = size(R);
if rn ~= n || rc ~= n
    error("The size of Q and R does not match.")
end
if k < 1 || k > m
    error("k should be in 1 : m.")
end
v = zeros(m, 1);
v(k) = 1;
[v, u, t] = orthogonalize(m, n, Q, v);
v(k:m-1) = v(k+1:m);
for l = n:-1:1
    for i = k:m-1
        Q(i,l) = Q(i+1,l);
    end
    [t, u(l), c, s] = computereflector(t, u(l));
    [u(l:n), R(l, l:n)] = applyreflector(c, s, u(l:n), R(l, l:n));
    [v(1:m-1), Q(1:m-1,l)] = applyreflector(c, s, v(1:m-1), Q(1:m-1,l));
    Q(m,l) = 0;
end
Q = Q(1:end-1,:);
end

