function [x, y, c, s] = computereflector(x, y)
%COMPUTEREFLECTOR compute the entries of givens matrix
u = x;
v = y;

if v ~= 0
    mu = max(abs(u), abs(v));
    t =  mu * sqrt((u / mu) ^ 2 + (v / mu) ^ 2);
    if u < 0
        t = -t;
    end
    c = u / t;
    s = v / t;
    x = t;
    y = 0;
else
    c = 1;
    s = 0;
end

end

