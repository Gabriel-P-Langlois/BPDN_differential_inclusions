function [v, r, rho] = orthogonalize(m, n, Q, v)
%adding a vector to an orthonormal matrix Q
%m, n is the nb of row and column of Q
%v is the new vector

    r = zeros(n,1);
    if m == n
        v = zeros(m, 1);
        rho = 0;
        return
    end
    rho = norm(v); rho_0 = rho;
    count = 0; v_null = false;
    while count <= 4
        s = (v.'*Q).';
        u = Q * s;        
        v = v - u;

        rho_1 = norm(v);

        r = r + s;

        if rho_0  >= sqrt(2) * rho_1
            rho_0 = rho_1;
        else
            v = v ./ rho_1;
            if v_null == false
                rho = rho_1;
            else
                rho = 0;
            end
        return
        end
        count = count + 1;
    end
end

