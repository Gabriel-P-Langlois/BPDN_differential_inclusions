1)  This repository contains the scripts and functions for the exact 
    homotopy method based on gradient inclusions.

2)  The scripts are in the main working directory, while the functions
    are located in the appropriate ./src directories.

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







%%%%%%
The general principle is that you should record every operation that you perform, and make those operations as transparent and reproducible as possible. In practice, this means that I create either a README file, in which I store every command line that I used while performing the experiment, or a driver script (I usually call this runall) that carries out the entire experiment automatically. The choices that you make at this point will depend strongly upon what development environment you prefer. If you are working in a language such as Matlab or R, you may be able to store everything as a script in that language. If you are using compiled code, then you will need to store the command lines separately. Personally, I work in a combination of shell scripts, Python, and C. The appropriate mix of these three languages depends upon the complexity of the experiment. Whatever you decide, you should end up with a file that is parallel to the lab notebook entry. The lab notebook contains a prose description of the experiment, whereas the driver script contains all the gory details.

Here are some rules of thumb that I try to follow when developing the driver script:

    Record every operation that you perform.
    Comment generously. The driver script typically involves little in the way of complicated logic, but often invokes various scripts that you have written, as well as a possibly eclectic collection of Unix utilities. Hence, for this type of script, a reasonable rule of thumb is that someone should be able to understand what you are doing solely from reading the comments. Note that I am refraining from advocating a particular mode of commenting for compiled code or more complex scripts—there are many schools of thought on the correct way to write such comments.
    Avoid editing intermediate files by hand. Doing so means that your script will only be semi-automatic, because the next time you run the experiment, you will have to redo the editing operation. Many simple editing operations can be performed using standard Unix utilities such as sed, awk, grep, head, tail, sort, cut, and paste.
    Store all file and directory names in this script. If the driver script calls other scripts or functions, then files and directory names should be passed from the driver script to these auxiliary scripts. Forcing all of the file and directory names to reside in one place makes it much easier to keep track of and modify the organization of your output files.
    Use relative pathnames to access other files within the same project. If you use absolute pathnames, then your script will not work for people who check out a copy of your project in their local directories (see “The Value of Version Control” below).
    Make the script restartable. I find it useful to embed long-running steps of the experiment in a loop of the form if (<output file does not exist>) then <perform operation>. If I want to rerun selected parts of the experiment, then I can delete the corresponding output files.

For experiments that take a long time to run, I find it useful to be able to obtain a summary of the experiment's progress thus far. In these cases, I create two driver scripts, one to run the experiment (runall) and one to summarize the results (summarize). The final line of runall calls summarize, which in turn creates a plot, table, or HTML page that summarizes the results of the experiment. The summarize script is written in such a way that it can interpret a partially completed experiment, showing how much of the computation has been performed thus far.
Handling and Preventing Errors

During the development of a complicated set of experiments, you will introduce errors into your code. Such errors are inevitable, but they are particularly problematic if they are difficult to track down or, worse, if you don't know about them and hence draw invalid conclusions from your experiment. Here are three suggestions for error handling.

First, write robust code to detect errors. Even in a simple script, you should check for bogus parameters, invalid input, etc. Whenever possible, use robust library functions to read standard file formats rather than writing ad hoc parsers.

Second, when an error does occur, abort. I typically have my program print a message to standard error and then exit with a non-zero exit status. Such behavior might seem like it makes your program brittle; however, if you try to skip over the problematic case and continue on to the next step in the experiment, you run the risk that you will never notice the error. A corollary of this rule is that your code should always check the return codes of commands executed and functions called, and abort when a failure is observed.

Third, whenever possible, create each output file using a temporary name, and then rename the file after it is complete. This allows you to easily make your scripts restartable and, more importantly, prevents partial results from being mistaken for full results.
Command Lines versus Scripts versus Programs

The design question that you will face most often as you formulate and execute a series of computational experiments is how much effort to put into software engineering. Depending upon your temperament, you may be tempted to execute a quick series of commands in order to test your hypothesis immediately, or you may be tempted to over-engineer your programs to carry out your experiment in a pleasingly automatic fashion. In practice, I find that a happy medium between these two often involves iterative improvement of scripts. An initial script is designed with minimal functionality and without the ability to restart in the middle of partially completed experiments. As the functionality of the script expands and the script is used more often, it may need to be broken into several scripts, or it may get “upgraded” from a simple shell script to Python, or, if memory or computational demands are too high, from Python to C or a mix thereof.

In practice, therefore, the scripts that I write tend to fall into these four categories:

    Driver script. This is a top-level script; hence, each directory contains only one or two scripts of this type.
    Single-use script. This is a simple script designed for a single use. For example, the script might convert an arbitrarily formatted file associated with this project into a format used by some of your existing scripts. This type of script resides in the same directory as the driver script that calls it.
    Project-specific script. This type of script provides a generic functionality used by multiple experiments within the given project. I typically store such scripts in a directory immediately below the project root directory (e.g., the msms/bin/parse-sqt.py file in Figure 1).
    Multi-project script. Some functionality is generic enough to be useful across many projects. I maintain a set of these generic scripts, which perform functions such as extracting specified sequences from a FASTA file, generating an ROC curve, splitting a file for n-fold cross-validation, etc.

Regardless of how general a script is supposed to be, it should have a clearly documented interface. In particular, every script or program, no matter how simple, should be able to produce a fairly detailed usage statement that makes it clear what the inputs and outputs are and what options are available.
