function [Q, R] = deletecolumn(Q, R, k)
%DELETECOLUMN delete the kth column of A and update Q, R
[m,n] = size(Q);
[rn, rc] = size(R);
if rn ~= n || rc ~= n
    error("The size of Q and R does not match.")
end
if k < 1 || k > n
    error("k should be in 1 : n.")
end
for l = k:n-1
    if l < n-1
    
    [R(l, l+1), R(l+1,l+1),c,s] = computereflector(R(l, l+1), R(l+1,l+1));
    [R(l,l+2:n), R(l+1,l+2:n)] = applyreflector(c, s, R(l,l+2:n), R(l+1,l+2:n));
    [Q(1:m,l),Q(1:m,l+1)] = applyreflector(c, s, Q(1:m,l),Q(1:m,l+1));
    end
    if l == n-1
        [R(l, l+1), R(l+1, l+1), c, s] = computereflector(R(l, l+1), R(l+1, l+1));
        [Q(1:m,l), Q(1:m,l+1)] = applyreflector(c, s, Q(1:m,l), Q(1:m,l+1));
    end
end
R = [R(:, 1:k-1) R(:, k+1:end)];
R = R(1:end-1,:);
Q = Q(:, 1:end-1);
end

