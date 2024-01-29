function [x, y] = applyreflector(c, s, x, y)
%APPLYREFLECTOR Apply givens matrix
        nu = s / (1 + c);
        k = length(x);
        for j = 1:k
            u = x(j);
            x(j) = u * c + y(j) * s;
            y(j) = (x(j) + u) * nu - y(j);
        end

end

