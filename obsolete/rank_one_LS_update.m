function new_M = rank_one_LS_update(K,M,v,j)
%RANK_ONE_LS_UPDATE Summary of this function goes here
%   Consider an m x n matrix K with m >= n. Let M = inv(K.'*K).
%   Then this script computes new_M = inv(Ktilde.'Ktilde) where Ktilde
%   is constructed by adding a column vector v at position j in the matrix K.

% Code based on https://emtiyaz.github.io/Writings/OneColInv.pdf

% Quantities from the matrix-inversion lemma
u1 = (v.'*K).';
u2 = M*u1;
tmp = v.'*v - u1.'*u2;
d = 1/tmp;
u3 = d*u2;
F11inv = M + d*(u2.'*u2);

new_M = [F11inv, -u3; -u3.',d];

% Permute column j and row j of new_M to last column and last row
if(j <= size(K,2))
    tmp = new_M(:,end);
    new_M(:,end) = new_M(:,j);
    new_M(:,j) = tmp;

    tmp2 = new_M(end,:);
    new_M(end,:) = new_M(j,:);
    new_M(j,:) = tmp2;
end
end

