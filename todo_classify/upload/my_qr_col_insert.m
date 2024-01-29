% Faster implementation of qrinsert...
function [Q,R] = my_qr_col_insert(Q,R,j,val,nr)
    R(:,j+1:nr+1) = R(:,j:nr);
    R(:,j) = val;
    [Q,R] = matlab.internal.math.insertCol(Q,R,j);
end