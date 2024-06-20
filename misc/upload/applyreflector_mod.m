function [x, y] = applyreflector_mod(c, s, x, y)
%APPLYREFLECTOR Apply givens matrix
% x = vector
% y = vector
% c = number
% s = number
        nu = s / (1 + c);
        u = x;
        x = x*c + y*s;
        y = (x+u)*nu - y;
end
