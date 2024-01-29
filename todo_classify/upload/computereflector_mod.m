function [x, y, c, s] = computereflector_mod(x, y)
%COMPUTEREFLECTOR compute the entries of givens matrix
if y ~= 0
    t =  sqrt(x^2+y^2);
    if x < 0
        t = -t;
    end
    c = x / t;
    s = y / t;
    x = t;
    y = 0;
else
    c = 1;
    s = 0;
end

end

