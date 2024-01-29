%% Function
function stop = lasso_conv_criterion(num_iters,max_iters,...
                t,val,tol)

    stop = norm(val,inf) <= t*(1 + tol) || (num_iters >= max_iters);
end