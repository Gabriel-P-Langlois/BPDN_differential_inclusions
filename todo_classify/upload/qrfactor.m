function [Q, R] = qrfactor(A)
[~, n] = size(A);
a = norm(A(:,1));
Q = A(:,1)/a;
R = a;
for k = 2:n
    v = A(:,k);
    [Q, R] = insertcolumn(Q, R, v, k);
end
end

