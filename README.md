1)  This repository contains the scripts and functions for the exact 
    homotopy method based on gradient inclusions.

2)  As for Jan 25 2024, there are two major scripts: one for comparing
    different implementations of the exact Lasso method, and one that
    compares one implementation of the exact lasso method with the  
    i) glmnet algorithm and ii) first-order FISTA with improved screening
    rule.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Some notes regarding the script that compares different implementations
of the exact lasso algorithm:

i) The QR rank (one) update implementation is faster, but it is
far less stable than the regular implementation.

    - Further speed up by computing the QR matrix of (Aeff,b)?
    - Rank update is less stable and may require a larger tolerance.
        e.g., for m = 1000, n = 100000, I cannot use tol = 1e-08.
        I need to use tol = 1e-07


Literature review based off this (who cited them):
 Keywords: Greville algorithm

Important reference:
@book{bjorck1996numerical,
  title={Numerical methods for least squares problems},
  author={Bj{\"o}rck, {\AA}ke},
  year={1996},
  publisher={SIAM}
}

Solving a modified system by the Woodbury or Sherman-Morrison formula will 
not always lead to stable methods. Stability will be a problem whenever 
the unmodified problem is worse-conditionedthan the modified problem. 
In general, methods which instead rely on modifying
a matrix factorization of A are to be preferred.

Section 3.2.4 has what we need.
Furthermore, it may be that all we need is to compute the QR
factorization of [K,b] instead of just K... maybe then we get
the updated solution for free?

QR rank append is the way to go, but we want to compute the QR
factorization of the augmented matrix (K,b)

ii) The numerical stability of a rank-one update is still unknown (to the
    best of GPL's knowledge, at least).

iii) Maybe we can use the solution of the previous least-square problem
     in some unusual way, by exploiting the properties of the problem
     and gradient inclusion?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Some notes regarding the script that compares different algorithms for
solving the Lasso problem:

See comments in the code. 

- The gradient inclusion method systematically produce solutions with 
lower mean squared error (MSE). This is so to a surprising extend; 
in many experiments I find MSE_{GI} ~ 1.25 MSE_{glmnet} near 
the end of the path. That is quite a large deviation. 

- One important observation is that the deviation is large despite
  how accurate the optimality conditions are for both the FISTA and GLMNET
  methods (up to 1e-08). The GI method is much more accurate (up to 1e-13).
  This gives complementary evidence that most numerical methods for solving
  BP aren't accurate enough. In fact, it seems, they are not accurate
  enough to solve the BPDN problem altogether.

- We'll need systematic numerical experiments to clarify this... But we
  need to be concise and quick about this. This cannot obscure the true
  result of the paper...