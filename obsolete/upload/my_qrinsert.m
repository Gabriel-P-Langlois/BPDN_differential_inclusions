% Faster implementation of qrinsert...
function [Q,R] = my_qr_col_insert(Q,R,j,x)
    [~,nr] = size(R);

    R(:,j+1:nr+1) = R(:,j:nr);
    R(:,j) = Q.'*x;
    [Q,R] = matlab.internal.math.insertCol(Q,R,j);
end