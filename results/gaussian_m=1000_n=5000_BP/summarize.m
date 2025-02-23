%%  summarize script
%   This script summarizes the results from the runall.m file located for
%   the gaussian_m=1000_n=5000_BP runall script.

if(summarize_1000_5000_BP)
    disp(['Infinity norm between the primal solutions obtained ...' ...
        'from the BP and BPDN solvers.'])
    disp(norm(sol_incl_BP_x - sol_incl_BPDN_x(:,end),inf)/...
        norm(sol_incl_BP_x,inf))
    disp(' ')

    disp(['Infinity norm between the dual solutions obtained ...' ...
        'from the BP and BPDN solvers.'])
    disp(norm(sol_incl_BP_p - sol_incl_BPDN_p(:,end),inf)/...
        norm(sol_incl_BP_p,inf))
    disp(' ')
end