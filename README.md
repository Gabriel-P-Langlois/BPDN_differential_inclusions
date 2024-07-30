1)  This repository contains the scripts and functions for the exact 
    homotopy method based on gradient inclusions.

2)  The solvers and functions are in the ./src directory. 
    The gaussian example is in the ./bin directory. 
    The ./results folder contains subidrectories containing several numerical results. 
    These subdirectories contain runall.m files that can be ran from the project directory. 

3)  The script_gaussian_example.m file in the ./bin folder allows for several options
    See the runall.m files in the ./results subdirectories for more information.
    GPL will add some comments later to describe what these inputs are, but it should
    be clear from the runall.m files what they do.

%%% Some notes %%%
(29/07/2024) There's a lot of work that was done to get to that point...
  A few remarks:
    - The tolerance for the exact BPDN algorithm should not be too small; 
      I set it to 1e-08 and everything was fine, even in the high-dimensional example.

    -  TBC (Note about the calculation of the descent direction)