function [Q, R] = insertcolumn(Q, R, v, k)
%INSERTCOLUMN insert a vector v as the kth column of A, return the updated
%Q, R
[m, n] = size(Q);
[rn, rc] = size(R);
if k > n+1
    error("k cannot be bigger than n+1")
end
if rn ~= n || rc ~= n
    error("The size of Q and R does not match.")
end
if length(v) ~= m
    error("The length of v doesn't fit, please make sure v is a m-by-1 vector.")
end
n = n + 1;
if m < n
    error("Cannot insert column when m < n")
end
R = [R;zeros(1,n-1)];
R = [R zeros(n,1)];
R(:, k+1:end) = R(:, k:end-1);
u = zeros(n,1);
[v, u(1:n-1),u(n)] = orthogonalize(m, n-1, Q, v);
Q = [Q, v];
for l = n-1:-1:k
    [u(l), u(l+1), c, s] = computereflector(u(l), u(l+1));
    [R(l,l+1:n), R(l+1,l+1:n)] = applyreflector(c, s, R(l,l+1:n), R(l+1,l+1:n));
    [Q(1:m, l), Q(1:m, l+1)] = applyreflector(c, s, Q(1:m, l), Q(1:m, l+1));
end
R(1:k, k) = u(1:k);
end

