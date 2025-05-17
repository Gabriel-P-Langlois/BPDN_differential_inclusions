This package contains 

approx_lasso.m: a matlab implementation of the homotopy algorithm for solving the Lasso with its variant presented
                in the ICML paper. When the parameter eps equals zero, it is the exact homotopy algorithm.
                When eps > 0, it uses the approximate homotopy variant (only works on linux 64bits computers).
                Note that this implementation is designed to privilege numerical precision over speed.
                For faster (but less numerically stable) implementation of the homotopy, see the 
                software package SPAMS: http://www.di.ens.fr/willow/SPAMS/

script_pathological_path1.m: an implementation of the pathological path with exponential complexity. The above
                implementation works with p <=11. For p>=12, numerical precision issues appear.

script_pathological_path2.m: an interesting path example, with n=2, but 2p kinks.

script_random_path.m: a study of the number of linear segments with random data.


