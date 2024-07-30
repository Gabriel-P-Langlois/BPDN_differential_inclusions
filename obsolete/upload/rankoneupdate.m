function [Q, R] = rankoneupdate(Q, R, v, u)
%UNTITLED update Q, R when A = A + vu'
[m, n] = size(Q);
[rn, rc] = size(R);
if rn ~= n || rc ~= n
    error("The size of Q and R does not match.")
end
if length(v) ~= m || length(u) ~= n
    error("The size of v or u is wrong.")
end
[v, t, rho] = orthogonalize(m, n, Q, v);
[t(n), rho, c, s] = computereflector(t(n), rho);
[R(n, n), rho] = applyreflector(c, s, R(n, n), rho);
[Q(1:m, n), v(1:m)] = applyreflector(c, s, Q(1:m, n), v(1:m));
for k = n-1:-1:1
    [t(k), t(k+1), c, s] = computereflector(t(k), t(k+1));
    [R(k,k:n), R(k+1, k:n)] = applyreflector(c, s, R(k,k:n), R(k+1, k:n));
    [Q(1:m,k), Q(1:m, k+1)] = applyreflector(c, s, Q(1:m,k), Q(1:m, k+1));
end
R(1,1:n) = R(1,1:n) + (t(1) * u)';
for k = 1:n-1
    [R(k,k),R(k+1,k),c, s] = computereflector(R(k,k),R(k+1,k));
    [R(k,k+1:n), R(k+1, k+1:n)] = applyreflector(c, s, R(k,k+1:n), R(k+1, k+1:n));
    [Q(1:m,k), Q(1:m, k+1)] = applyreflector(c, s, Q(1:m,k), Q(1:m, k+1));
end
[R(n,n), ~, c, s] = computereflector(R(n,n), rho);
[Q(1:m, n), ~] = applyreflector(c, s, Q(1:m, n), v(1:m));
end


