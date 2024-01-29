%% Function
function yprox = l1_prox_operator(x,t)

% Soft thresholding operation
yprox = abs(x)-t;
yprox = sign(x).*(yprox+abs(yprox))*0.5;

end