/////////////////////////////////////
//lsqnoneg, solve for Ax=b, where x >= 0
//OpenMP + MKL
//Yuancheng Luo, 2/2011
////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <string.h>
// #include "omp.h"
//#include "mkl.h"
#include "mex.h"
#include "blas.h"
#include "lapack.h"
#include "nnlsmkl.h"

#include <limits>

#define HUGE std::numeric_limits<double>::infinity()


// void loadData(REAL **A, REAL **b, REAL **x,  ptrdiff_t nSys, ptrdiff_t m, ptrdiff_t n, char *fileNameA, char*fileNameB){
//   //m rows, n cols, use fortran column-major ordering
//   *A = (REAL*) malloc(sizeof(REAL) * nSys *  m * n ); //A[nSys][m][n]
//   *x = (REAL*) malloc(sizeof(REAL) *nSys * n);        //x[nSys][n]
//   *b = (REAL*) malloc(sizeof(REAL) *nSys * m);        //b[nSys][m]

//   //Grab b from file
//   FILE *fi = fopen(fileNameB, "rb");
//   fread(*b, 1, sizeof(REAL) * m * nSys, fi);
//   fclose(fi);

//   //Grab A from file
//   fi = fopen(fileNameA, "rb");
//   fread(*A, 1, sizeof(REAL) * m * n * nSys, fi);
//   fclose(fi);

// }//End of genRandomData

//OMP/MKL  systems, no update/downdate
void nnlsOMPSysMKL(const REAL *A, REAL *b, REAL *x, ptrdiff_t isTransposed, ptrdiff_t maxNNLSIters, ptrdiff_t maxLSIters, ptrdiff_t nSys, ptrdiff_t m, ptrdiff_t n, ptrdiff_t MKLT, ptrdiff_t OMPT, REAL TOL_TERMINATION){

  char transaN = 'N', transaT = 'T';
  REAL alpha = 1.0f, negAlpha = -1.0f,  beta = 0.0f;
  ptrdiff_t idx = 1, idy = 1;

  REAL *f = (REAL*)malloc(sizeof(REAL) * OMPT * m);
  REAL *g = (REAL*)malloc(sizeof(REAL) * OMPT * n);  //g[OMPT][n]
  ptrdiff_t *z = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * OMPT * n);         //z[OMPT][n]
  ptrdiff_t *zRemoved =  (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * OMPT* n); 

  ptrdiff_t *kIdx = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * OMPT * n);

  ptrdiff_t maxmn = MAX(m, n);
  REAL *Apt = (REAL*)malloc(sizeof(REAL) * OMPT * n * m);
  REAL *xp = (REAL*)malloc(sizeof(REAL) * OMPT * maxmn );
  REAL *tx = (REAL*) malloc(sizeof(REAL) * OMPT * n);

  ptrdiff_t NB = 1;
  ptrdiff_t lwork =  (MAX(m, n) * (2 + NB));
  REAL *work = (REAL*) malloc(sizeof(REAL) *OMPT * lwork);
  ptrdiff_t info;

  ptrdiff_t s;
  ptrdiff_t i, j, k;

  //Set number of threads
  // mkl_set_num_threads(MKLT); 
  // omp_set_num_threads(OMPT); 

  double startT, endT;
  // startT = omp_get_wtime();
  
  //Set initial solution to 0
  memset(x, 0, sizeof(REAL) * nSys * n);

  //#pragma omp parallel private(s, i, j, k, info)
  {
    ptrdiff_t tid = 0;//omp_get_thread_num();
    ptrdiff_t nnlsIter, lsIter;
    
    //#pragma omp for
    for(s = 0; s < nSys; ++s){
      //Initials
      nnlsIter = lsIter = 0;
      ptrdiff_t kCols = 0;
      //Set active-set to all variables
      for(i = 0; i < n; ++i)
	z[M2(tid, i, n)] = 1;
      
      do{
	//prptrdiff_tf("nnlsIter %d, lsIter %d\n", nnlsIter, lsIter);
	////Compute negative gradient of f=1/2|Ax-b|^2 w.r.t. x and store ptrdiff_to g=A'(b-Ax)
	
#ifndef USE_DOUBLE
	if(isTransposed){
	  //Compute f=-Ax, treat as A transpose for fortran call
	  SGEMV(&transaT, &n,  &m,  &negAlpha, &A[M3(s, 0, 0, m, n)], &n, &x[M2(s, 0, n)], &idx, &beta, &f[M2(tid, 0, m)], &idy);
	  //Compute f=b-Ax
	  SAXPY(&m, &alpha, &b[M2(s, 0, m)], &idx, &f[M2(tid, 0, m)], &idy);
	  //Compute g=A'f, treat as A transpose for fortran call
	  SGEMV(&transaN, &n, &m, &alpha, &A[M3(s, 0, 0, m, n)], &n, &f[M2(tid, 0, m)], &idx, &beta, &g[M2(tid, 0, n)], &idy);
	}else{
	  //Compute f=-Ax
	  SGEMV(&transaN, &m,  &n,  &negAlpha, &A[M3(s, 0, 0, n, m)], &m, &x[M2(s, 0, n)], &idx, &beta, &f[M2(tid, 0, m)], &idy);
	  //Compute f=b-Ax
	  SAXPY(&m, &alpha, &b[M2(s, 0, m)], &idx, &f[M2(tid, 0, m)], &idy);
	  //Compute g=A'f
	  SGEMV(&transaT, &m, &n, &alpha, &A[M3(s, 0, 0, n, m)], &m, &f[M2(tid, 0, m)], &idx, &beta, &g[M2(tid, 0, n)], &idy);
	}
#else
	if(isTransposed){
	  //Compute f=-Ax, treat as A transpose for fortran call
	   dgemv(&transaT, &n,  &m,  &negAlpha, &A[M3(s, 0, 0, m, n)], &n, &x[M2(s, 0, n)], &idx, &beta, &f[M2(tid, 0, m)], &idy);
	   //DGEMV(&transaT, &n,  &m,  &negAlpha, &A[M3(s, 0, 0, m, n)], &n, &x[M2(s, 0, n)], &idx, &beta, &f[M2(tid, 0, m)], &idy);
	  //Compute f=b-Ax
	  daxpy(&m, &alpha, &b[M2(s, 0, m)], &idx, &f[M2(tid, 0, m)], &idy);
	  //Compute g=A'f, treat as A transpose for fortran call
	  dgemv(&transaN, &n, &m, &alpha, &A[M3(s, 0, 0, m, n)], &n, &f[M2(tid, 0, m)], &idx, &beta, &g[M2(tid, 0, n)], &idy);
	}else{
	  //Compute f=-Ax
	  dgemv(&transaN, &m,  &n,  &negAlpha, &A[M3(s, 0, 0, n, m)], &m, &x[M2(s, 0, n)], &idx, &beta, &f[M2(tid, 0, m)], &idy);
	  //Compute f=b-Ax
	  daxpy(&m, &alpha, &b[M2(s, 0, m)], &idx, &f[M2(tid, 0, m)], &idy);
	  //Compute g=A'f
	  dgemv(&transaT, &m, &n, &alpha, &A[M3(s, 0, 0, n, m)], &m, &f[M2(tid, 0, m)], &idx, &beta, &g[M2(tid, 0, n)], &idy);
	}
#endif
	//Check for termination condition by finding a variable in z, s.t. max(g)>0 and remove from z
	ptrdiff_t vmax = -1;
	REAL maxg = TOL_TERMINATION;
	for(i = 0; i < n; ++i){
	  if(z[M2(tid, i, n)] && g[M2(tid, i, n)] > maxg){
	    maxg = g[M2(tid, i, n)];
	    vmax = i;
	  }
	}
	//All gradients are non-positive so terminate
	if(vmax == -1)
	  break;
	
	//Remove vmax from z
	z[M2(tid, vmax, n)] = 0;
	ptrdiff_t colAdd = 1;
	
	//Solve uncontrained linear least squares subproblem
	do{
	  ++lsIter;
	  
	  if(colAdd){
	    //Build unconstrained system
	    for(i = 0; i < kCols; ++i){
	      ptrdiff_t kdx = kIdx[M2(tid, i, n)];
	      if(isTransposed){
		for(j = 0; j < m; ++j)
		  Apt[M3(tid, i, j, n, m)] = A[M3(s, j, kdx, m, n)];
	      }else{
		memcpy(&Apt[M3(tid, i, 0, n, m)], &A[M3(s, kdx, 0, n, m)], sizeof(REAL) * m);
	      }
	    }
	    //Add new column
	    if(isTransposed){
	      for(j = 0; j < m; ++j)
		Apt[M3(tid, kCols, j, n, m)] = A[M3(s, j, vmax, m, n)];
	    }else{
	      memcpy(&Apt[M3(tid, kCols, 0, n, m)], &A[M3(s, vmax, 0, n, m)], sizeof(REAL) * m);
	    }
	    kIdx[M2(tid, kCols, n)] = vmax;
	    kCols++;
	    
	  }else{
	    //Shift sK (column list in set P) list left by one
	    for(i = kCols - 1; i >= 0; --i){
	      ptrdiff_t kdx =  kIdx[M2(tid, i, n)];
	      //Deleted column
	      if(zRemoved[M2(tid, kdx, n)]){
		for(j = i + 1; j < kCols; ++j)
		  kIdx[M2(tid, j-1, n)] = kIdx[M2(tid, j, n)];
		--kCols;
	      }
	    }
	    //Build unconstrained system
	    for(i = 0; i < kCols; ++i){
	      ptrdiff_t kdx = kIdx[M2(tid, i, n)];
	      if(isTransposed){
		for(j = 0; j < m; ++j)
		  Apt[M3(tid, i, j, n, m)] = A[M3(s, j, kdx, m, n)];
	      }else{
		memcpy(&Apt[M3(tid, i, 0, n, m)], &A[M3(s, kdx, 0, n, m)], sizeof(REAL) * m);
	      }
	    }
	  }
	  //Solve unconstrained system
	  ptrdiff_t one = 1;
	  memcpy(&xp[M2(tid, 0, maxmn)], &b[M2(s, 0, m)], sizeof(REAL) * m);
#ifndef USE_DOUBLE
	  sgels(&transaN, &m, &kCols, &one, &Apt[M3(tid, 0, 0, n, m)], &m, &xp[M2(tid, 0, maxmn)], &m, &work[M2(tid, 0, lwork)], &lwork, &info);
#else
	  dgels(&transaN, &m, &kCols, &one, &Apt[M3(tid, 0, 0, n, m)], &m, &xp[M2(tid, 0, maxmn)], &m, &work[M2(tid, 0, lwork)], &lwork, &info);
#endif
	  //Load solution xp ptrdiff_t tx
	  REAL mptrdiff_tx = HUGE;
	  memset(&tx[M2(tid, 0, n)], 0, sizeof(REAL) * n);
	  for(i = kCols - 1; i >= 0; --i){
	    ptrdiff_t kdx = kIdx[M2(tid, i, n)];
	    tx[M2(tid, kdx, n)] = xp[M2(tid, i, maxmn)];
	    mptrdiff_tx = MIN(tx[M2(tid, kdx, n)], mptrdiff_tx);
	  }

	  if(mptrdiff_tx > 0){
	    //Accept solution, update x
	    //prptrdiff_tf("Accept\n");
	    memcpy(&x[M2(s, 0, n)], &tx[M2(tid, 0, n)], sizeof(REAL) * n);
	    break;
	  }else{
	    //Reject solution, update subproblem
	    //Find index q in set P for negative z such that x/(x-z) is minimized
	    //prptrdiff_tf("Reject\n");
	    REAL minAlpha = HUGE;
	    for(i = 0; i < kCols; ++i){
	      ptrdiff_t kdx = kIdx[M2(tid, i, n)];
	      if(tx[M2(tid, kdx, n)] <= 0)
		minAlpha = MIN(minAlpha, x[M2(s, kdx, n)] / (x[M2(s, kdx, n)] - tx[M2(tid, kdx, n)]) );
	    }	  
	    for(i = 0; i < n; ++i)
	      tx[M2(tid, i, n)] -=  x[M2(s, i, n)];
	    
#ifndef USE_DOUBLE
	    saxpy(&n, &minAlpha, &tx[M2(tid, 0, n)], &idx, &x[M2(s, 0, n)], &idy);
#else
	    daxpy(&n, &minAlpha, &tx[M2(tid, 0, n)], &idx, &x[M2(s, 0, n)], &idy);
#endif	    
	    memset(&zRemoved[M2(tid, 0, n)], 0, sizeof(ptrdiff_t) * n);
	    //Move from set P to set Z all elements whose corresponding X is 0 (guaranteed to have one element)
	    for(i = 0; i < kCols; ++i){
	      ptrdiff_t kdx = kIdx[M2(tid, i, n)];
	      zRemoved[M2(tid, kdx, n)] = z[M2(tid, kdx, n)] = fabs(x[M2(s, kdx, n)]) <= TOL_TERMINATION;
	    }
	    
	    colAdd = 0;
	  }
	  
	}while(lsIter < maxLSIters);
	
	if(lsIter >= maxLSIters)
	  break;
	
	++nnlsIter;
      }while(nnlsIter < maxNNLSIters);
      
      // prptrdiff_tf("nSys %d, nnlsIter %d, lsIter %d\n", s, nnlsIter, lsIter);  
    }
  }

  // endT = omp_get_wtime();
  // prptrdiff_tf("Elapsed time %f\n", endT - startT);
  
  free(work); 
  free(Apt); 
  free(xp); 
  free(tx); 
  free(kIdx);
  free(zRemoved); 
  free(z); 
  free(f);
  free(g);
}//End of nnlsOMPSysMKL



// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------
// --------------------------------------------------------------------------------


//OMP systems, MKL updates
void nnlsOMPSysMKLUpdates(const REAL *A, const REAL *b, REAL *x, ptrdiff_t isTransposedNotUsed,  ptrdiff_t maxNNLSIters, ptrdiff_t maxLSIters, ptrdiff_t nSys, ptrdiff_t m, ptrdiff_t n, ptrdiff_t MKLT, ptrdiff_t OMPT, REAL TOL_TERMINATION){

  constexpr ptrdiff_t isTransposed = 0;
  constexpr char transaN = 'N', transaT = 'T';
  constexpr REAL alpha = 1.0f, negAlpha = -1.0f,  beta = 0.0f;
  constexpr ptrdiff_t idx = 1, idy = 1;

  REAL *f = (REAL*)malloc(sizeof(REAL) * OMPT * m);
  REAL *g = (REAL*)malloc(sizeof(REAL) * OMPT * n);  //g[OMPT][n]
  ptrdiff_t *z = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * OMPT * n);         //z[OMPT][n]
  ptrdiff_t *zRemoved =  (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * OMPT* n); 

  REAL *newCol = (REAL*)malloc(sizeof(REAL) * OMPT * m);
  REAL *oldCol = (REAL*)malloc(sizeof(REAL) * OMPT * m);

  ptrdiff_t *kIdx = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) * OMPT * n);

  REAL *Qt = (REAL*) malloc(sizeof(REAL) * OMPT * n * m);
  REAL *R = (REAL*) malloc(sizeof(REAL) * OMPT * n * n );
  REAL *Qtb = (REAL*) malloc(sizeof(REAL) * OMPT * n);

  REAL *tx = (REAL*) malloc(sizeof(REAL) * OMPT * n);

  ptrdiff_t s;
  ptrdiff_t i, j, k;

  //Set number of threads
  // mkl_set_num_threads(MKLT);
  // omp_set_num_threads(OMPT);

  double startT, endT;
  // startT = omp_get_wtime();
  
  //Set initial solution to 0
  memset(x, 0, sizeof(REAL) * nSys * n);

  //#pragma omp parallel private(s, i, j, k)
  {
    ptrdiff_t tid = 0;//omp_get_thread_num();
    ptrdiff_t nnlsIter, lsIter;
    
    //#pragma omp for
    for(s = 0; s < nSys; ++s){
      //Initials
      nnlsIter = lsIter = 0;
      ptrdiff_t kCols = 0;
      //Set active-set to all variables
      for(i = 0; i < n; ++i)
	z[M2(tid, i, n)] = 1;
      
      do{
	//prptrdiff_tf("nnlsIter %d, lsIter %d\n", nnlsIter, lsIter);
	////Compute negative gradient of f=1/2|Ax-b|^2 w.r.t. x and store ptrdiff_to g=A'(b-Ax)
// #ifndef USE_DOUBLE
// 	if(isTransposed){
// 	  //Compute f=-Ax, treat as A transpose for fortran call
// 	  SGEMV(&transaT, &n,  &m,  &negAlpha, &A[M3(s, 0, 0, m, n)], &n, &x[M2(s, 0, n)], &idx, &beta, &f[M2(tid, 0, m)], &idy);
// 	  //Compute f=b-Ax
// 	  SAXPY(&m, &alpha, &b[M2(s, 0, m)], &idx, &f[M2(tid, 0, m)], &idy);
// 	  //Compute g=A'f, treat as A transpose for fortran call
// 	  SGEMV(&transaN, &n, &m, &alpha, &A[M3(s, 0, 0, m, n)], &n, &f[M2(tid, 0, m)], &idx, &beta, &g[M2(tid, 0, n)], &idy);
// 	}else{
// 	  //Compute f=-Ax
// 	  SGEMV(&transaN, &m,  &n,  &negAlpha, &A[M3(s, 0, 0, n, m)], &m, &x[M2(s, 0, n)], &idx, &beta, &f[M2(tid, 0, m)], &idy);
// 	  //Compute f=b-Ax 
// 	  SAXPY(&m, &alpha, &b[M2(s, 0, m)], &idx, &f[M2(tid, 0, m)], &idy);
// 	  //Compute g=A'f
// 	  SGEMV(&transaT, &m, &n, &alpha, &A[M3(s, 0, 0, n, m)], &m, &f[M2(tid, 0, m)], &idx, &beta, &g[M2(tid, 0, n)], &idy);
// 	}
// #else	
	// if(isTransposed){
	//   //Compute f=-Ax, treat as A transpose for fortran call
	//   DGEMV(&transaT, &n,  &m,  &negAlpha, &A[M3(s, 0, 0, m, n)], &n, &x[M2(s, 0, n)], &idx, &beta, &f[M2(tid, 0, m)], &idy);
	//   //Compute f=b-Ax
	//   DAXPY(&m, &alpha, &b[M2(s, 0, m)], &idx, &f[M2(tid, 0, m)], &idy);
	//   //Compute g=A'f, treat as A transpose for fortran call
	//   DGEMV(&transaN, &n, &m, &alpha, &A[M3(s, 0, 0, m, n)], &n, &f[M2(tid, 0, m)], &idx, &beta, &g[M2(tid, 0, n)], &idy);
	// }else{
	  //Compute f=-Ax
	  dgemv(&transaN, &m,  &n,  &negAlpha, &A[M3(s, 0, 0, n, m)], &m, &x[M2(s, 0, n)], &idx, &beta, &f[M2(tid, 0, m)], &idy);
	  //Compute f=b-Ax 
	  daxpy(&m, &alpha, &b[M2(s, 0, m)], &idx, &f[M2(tid, 0, m)], &idy);
	  //Compute g=A'f
	  dgemv(&transaT, &m, &n, &alpha, &A[M3(s, 0, 0, n, m)], &m, &f[M2(tid, 0, m)], &idx, &beta, &g[M2(tid, 0, n)], &idy);
//	}
// #endif
	//Check for termination condition by finding a variable in z, s.t. max(g)>0 and remove from z
	ptrdiff_t vmax = -1;
	REAL maxg = TOL_TERMINATION;
	for(i = 0; i < n; ++i){
	  if(z[M2(tid, i, n)] && g[M2(tid, i, n)] > maxg){
	    maxg = g[M2(tid, i, n)];
	    vmax = i;
	  }
	}
	//All gradients are non-positive so terminate
	if(vmax == -1)
	  break;
	
	//Remove vmax from z
	z[M2(tid, vmax, n)] = 0;
	ptrdiff_t colAdd = 1;
	
	//Solve uncontrained linear least squares subproblem
	do{
	  ++lsIter;
	  
	  if(colAdd){
	    //prptrdiff_tf("Column update\n");
	    ////Modified Gram-Schmidt update
	    //load old col
	    if(isTransposed){
	      for(i = 0 ; i < m; ++i) 
		newCol[M2(tid, i, m)] = A[M3(s, i, vmax, m, n)];
	    }else{
	      memcpy(&newCol[M2(tid, 0, m)], &A[M3(s, vmax, 0, n, m)], sizeof(REAL) * m);
	    }
	    memcpy(&oldCol[M2(tid, 0, m)], &newCol[M2(tid, 0, m)], sizeof(REAL) * m);
	 
	    //Orthogonalize newCol with all previous kCols columns in Q
	    for(i = 0; i < kCols ;++i){
	      ptrdiff_t kdx = kIdx[M2(tid, i, n)];
// #ifndef USE_DOUBLE
// 	      REAL sc = -SDOT(&m, &Qt[M3(tid, kdx, 0, n,  m)], &idx, &newCol[M2(tid, 0, m)], &idy);
// 	      SAXPY(&m, &sc, &Qt[M3(tid, kdx, 0, n,  m)], &idx, &newCol[M2(tid, 0, m)], &idy);
// 	      //Update R
// 	      R[M3(tid, kdx, vmax, n, n)] = SDOT(&m, &Qt[M3(tid, kdx, 0, n, m)], &idx, &oldCol[M2(tid, 0, m)], &idy);
// #else
	      REAL sc = -ddot(&m, &Qt[M3(tid, kdx, 0, n,  m)], &idx, &newCol[M2(tid, 0, m)], &idy);
	      daxpy(&m, &sc, &Qt[M3(tid, kdx, 0, n,  m)], &idx, &newCol[M2(tid, 0, m)], &idy);
	      //Update R
	      R[M3(tid, kdx, vmax, n, n)] = ddot(&m, &Qt[M3(tid, kdx, 0, n, m)], &idx, &oldCol[M2(tid, 0, m)], &idy);
// #endif    
	    }
	   
	    //REAL norm2 = sqrtf( SDOT(&m, newCol, &idx, newCol, &idy));
// #ifndef USE_DOUBLE
// 	    REAL norm2 = SNRM2(&m, &newCol[M2(tid, 0, m)], &idx);
// #else
	    REAL norm2 = dnrm2(&m, &newCol[M2(tid, 0, m)], &idx);
// #endif
	    R[M3(tid, vmax, vmax, n, n)] = norm2;
	    
	    norm2 = 1.0f / norm2;
// #ifndef USE_DOUBLE
// 	    SSCAL(&m, &norm2, &newCol[M2(tid, 0, m)], &idx);
// #else   
	    dscal(&m, &norm2, &newCol[M2(tid, 0, m)], &idx);
// #endif
	    memcpy(&Qt[M3(tid, vmax, 0, n, m)], &newCol[M2(tid, 0, m)], sizeof(REAL) * m);
// #ifndef USE_DOUBLE
// 	    Qtb[M2(tid, vmax, n)] = SDOT(&m, &newCol[M2(tid, 0, m)], &idx, &b[M2(s, 0, m)], &idy);
// #else
	    Qtb[M2(tid, vmax, n)] = ddot(&m, &newCol[M2(tid, 0, m)], &idx, &b[M2(s, 0, m)], &idy);
// #endif
	    //Append vmax to kIdx
	    kIdx[M2(tid, kCols, n)] = vmax;
	    kCols++;
	    
	  }else{
	    //prptrdiff_tf("Column Downdate\n");
	    //Given's Rotations downdate
	    
	    //Rebuild kIdx list
	    for(i = kCols - 1; i >= 0; --i){
	      ptrdiff_t kdx = kIdx[M2(tid, i, n)];
	      
	      //Deleted column
	      if(zRemoved[M2(tid, kdx, n)]){
		for(j = i + 1; j < kCols; ++j){
		  ptrdiff_t jdx  = kIdx[M2(tid, j, n)];
		  
		  REAL gx = R[M3(tid, kdx, jdx, n, n)];
		  REAL ax = R[M3(tid, jdx, jdx, n, n)];
		  
		  //Compute c,s,-s,c coefficients for row/col jdx and row/col kdx
		  REAL givenc, givens;
		  //SROTG(&ax, &gx, &givenc, &givens); //Not accurate, use below
		  if(ax == 0 && gx == 0){
		    givenc = 0;
		    givens = 0;
		  }else if(ax == 0){
		    givenc = 0;
		    givens = 1;
		  }else if(gx == 0){
		    givenc = 1;
		    givens = 0;
		  }else if(fabs(gx) > fabs(ax)){
		    REAL r = -ax / gx;
		    givens = 1 / sqrt(1 + r * r);
		    givenc = -givens * r;
		  }else{
		    REAL r = -gx / ax;
		    givenc = 1 / sqrt(1 + r * r);
		    givens = -givenc * r;
		  }
		  
// #ifndef USE_DOUBLE		  
// 		  //Update R
// 		  SROT(&n, &R[M3(tid, jdx, 0, n, n)], &idx, &R[M3(tid, kdx, 0, n, n)], &idy, &givenc, &givens);
		  
// 		  //Update Qt
// 		  SROT(&m, &Qt[M3(tid, jdx, 0, n, m)], &idx, &Qt[M3(tid, kdx, 0, n, m)], &idy, &givenc, &givens);
		  
// 		  //Update Qtb
// 		  ptrdiff_t one = 1;
// 		  SROT(&one, &Qtb[M2(tid, jdx, n)], &idx, &Qtb[M2(tid, kdx, n)], &idy, &givenc, &givens);
// #else
		  //Update R
		  drot(&n, &R[M3(tid, jdx, 0, n, n)], &idx, &R[M3(tid, kdx, 0, n, n)], &idy, &givenc, &givens);
		  
		  //Update Qt
		  drot(&m, &Qt[M3(tid, jdx, 0, n, m)], &idx, &Qt[M3(tid, kdx, 0, n, m)], &idy, &givenc, &givens);
		  
		  //Update Qtb
		  constexpr ptrdiff_t one = 1;
		  drot(&one, &Qtb[M2(tid, jdx, n)], &idx, &Qtb[M2(tid, kdx, n)], &idy, &givenc, &givens);
// #endif
		}
		
		//Shift sK (column list in set P) list left by one
		for(j = i + 1; j < kCols; ++j)
		  kIdx[M2(tid, j-1, n)] = kIdx[M2(tid, j, n)];
		
		--kCols;
	      }
	    }
	  }
	  
	  //Compute solution tx	
	  REAL mptrdiff_tx = HUGE;
	  memset(&tx[M2(tid, 0, n)], 0, sizeof(REAL) * n);
	  for(i = kCols - 1; i >= 0; --i){
	    ptrdiff_t kdx = kIdx[M2(tid, i, n)];
	    //Compute tx via backsubstitution
	    REAL coeff = R[M3(tid, kdx, kdx, n, n)];
	    REAL pSum = 0;
	    for(j = 0; j < kCols; ++j){
	      ptrdiff_t jdx = kIdx[M2(tid, j, n)];
	      pSum += (kdx != jdx) * R[M3(tid, kdx, jdx, n, n)] * tx[M2(tid, jdx, n)];
	    }
	    tx[M2(tid, kdx, n)] = (Qtb[M2(tid, kdx, n)] - pSum) / coeff;
	    mptrdiff_tx = MIN(tx[M2(tid, kdx, n)], mptrdiff_tx);
	  }
	  
	  if(mptrdiff_tx > 0){
	    //Accept solution, update x
	    //prptrdiff_tf("Accept\n");
	    memcpy(&x[M2(s, 0, n)], &tx[M2(tid, 0, n)], sizeof(REAL) * n);
	    break;
	  }else{
	    //Reject solution, update subproblem
	    //Find index q in set P for negative z such that x/(x-z) is minimized
	    //prptrdiff_tf("Reject\n");
	    REAL minAlpha = HUGE;
	    for(i = 0; i < kCols; ++i){
	      ptrdiff_t kdx = kIdx[M2(tid, i, n)];
	      if(tx[M2(tid, kdx, n)] <= 0)
		minAlpha = MIN(minAlpha, x[M2(s, kdx, n)] / (x[M2(s, kdx, n)] - tx[M2(tid, kdx, n)]) );
	    }	  
	    for(i = 0; i < n; ++i)
	      tx[M2(tid, i, n)] -=  x[M2(s, i, n)];
	    
// #ifndef USE_DOUBLE
// 	    SAXPY(&n, &minAlpha, &tx[M2(tid, 0, n)], &idx, &x[M2(s, 0, n)], &idy);
// #else
	    daxpy(&n, &minAlpha, &tx[M2(tid, 0, n)], &idx, &x[M2(s, 0, n)], &idy);
// #endif	    
	    memset(&zRemoved[M2(tid, 0, n)], 0, sizeof(ptrdiff_t) * n);
	    //Move from set P to set Z all elements whose corresponding X is 0 (guaranteed to have one element)
	    for(i = 0; i < kCols; ++i){
	      ptrdiff_t kdx = kIdx[M2(tid, i, n)];
	      zRemoved[M2(tid, kdx, n)] = z[M2(tid, kdx, n)] = fabs(x[M2(s, kdx, n)]) <= TOL_TERMINATION;
	    }
	    
	    colAdd = 0;
	  }
	  
	}while(lsIter < maxLSIters);
	
	if(lsIter >= maxLSIters)
	  break;
	
	++nnlsIter;
      }while(nnlsIter < maxNNLSIters);
      
      // prptrdiff_tf("nSys %d, nnlsIter %d, lsIter %d\n", s, nnlsIter, lsIter);  
    }
  }

  // endT = omp_get_wtime();
  // prptrdiff_tf("Elapsed time %f\n", endT - startT);

  free(tx);
  free(Qtb);
  free(R);
  free(Qt);
  free(kIdx);
  free(newCol);
  free(oldCol);
  free(zRemoved);
  free(z);
  free(f);
  free(g);
}//End of nnlsOMPSysMKLUpdates

// void writeOutput(REAL *A, REAL *b, REAL *x,  ptrdiff_t nSys, ptrdiff_t m, ptrdiff_t n, char* fileName){
//   //Write x in binary
//   FILE *f_cpu = fopen(fileName, "wb");
//   fwrite(x, 1,  sizeof(REAL) * nSys *  n, f_cpu);
//   fclose(f_cpu);
  
//   //Write x, Ax, and b in txt format
//   //Compute error ||Ax-b||2

//   f_cpu = fopen("cpu_result.txt","w");
//   ptrdiff_t s, i, j, k;

//   for(s = 0; s < nSys; ++s){
//     fprptrdiff_tf(f_cpu, "\nnSys %d Ax\n", s);
//     REAL norm2 = 0;
//     for(i = 0; i < m; ++i){
//       REAL temp = 0;
//       for(j = 0; j < n; ++j)
// 	temp += A[M3(s, i, j, m, n)] * x[M2(s, j, n) ]; 
//       fprptrdiff_tf(f_cpu, "%f ", temp);
//       temp -= b[M2(s, i, m)];
//       norm2 += temp * temp;
//     }
//     norm2 = sqrt(norm2);
//     prptrdiff_tf("nSys %d, norm %f\n", s, norm2);
//     fprptrdiff_tf(f_cpu, "\nnorm %f", norm2);
    
//     fprptrdiff_tf(f_cpu, "\n\nb\n");
//     for(i = 0; i < m; ++i) 
//       fprptrdiff_tf(f_cpu, "%f ", b[M2(s, i, m) ]);
//     fprptrdiff_tf(f_cpu, "\n");

//     fprptrdiff_tf(f_cpu, "\n\nx\n");
//     for(i = 0; i < n; ++i) 
//       fprptrdiff_tf(f_cpu, "%f ", x[M2(s, i, n) ]);
//     fprptrdiff_tf(f_cpu, "\n");
    
//   }
//   fclose(f_cpu);
// } //end of writeOutput


/* ptrdiff_t main(ptrdiff_t argc, char *argv[]){ */
/*   ptrdiff_t method = 0; */
/*   ptrdiff_t  M = 512; //Equations */
/*   ptrdiff_t  N = 512; //Unknowns */
/*   ptrdiff_t  NSYS = 256; //1-256 */
/*   ptrdiff_t isTransposed = 0; */
/*   REAL TOL_TERMINATION = 1e-6; */
/*   ptrdiff_t MKLThreads = 1, OMPThreads = 1; */
/*   REAL *A = NULL, *b = NULL, *x = NULL; */
/*   if(argc < 9){ */
/*     prptrdiff_tf("Expect: method, nSys, isTransposed, mEquations, nUnknowns, tol, MKLThreads, OMPThreads\n"); */
/*   }else{ */
/*     method =  atoi(argv[1]); */
/*     NSYS = atoi(argv[2]); */
/*     isTransposed = atoi(argv[3]); */
/*     M = atoi(argv[4]); */
/*     N = atoi(argv[5]); */
/*     TOL_TERMINATION = atof(argv[6]); */
/*     MKLThreads = atoi(argv[7]); */
/*     OMPThreads = atoi(argv[8]); */

/*     srand(2011); */
/*     loadData(&A, &b, &x, NSYS, M, N, "sysA.bin", "sysB.bin"); */
    
/*     if(method == 0) */
/*       nnlsOMPSysMKL(A, b, x, 0, MAX_ITER_NNLS(M, N), MAX_ITER_LS(M, N), NSYS, M, N, MKLThreads, OMPThreads, TOL_TERMINATION); */
/*     else */
/*       nnlsOMPSysMKLUpdates(A, b, x, isTransposed,  MAX_ITER_NNLS(M, N), MAX_ITER_LS(M, N), NSYS, M, N, MKLThreads, OMPThreads, TOL_TERMINATION);   */
    
/*     writeOutput(A, b, x, NSYS, M, N, "sysX.bin"); */
    
/*     if(A) free(A); */
/*     if(b) free(b); */
/*     if(x) free(x); */
/*   } */
/*   return 0; */
/* }//End of main */



//For mex calls
void mexFunction2(ptrdiff_t nlhs, mxArray *plhs[], ptrdiff_t nrhs, const mxArray *prhs[]) {
  // prptrdiff_tf("in mex function\n");
  if(nrhs != 10 || nlhs != 1)
    mexErrMsgTxt("Expect: x = NNLS(method, A, b, nSys, isTransposed, mEquations, nUnknowns, tol, MKLThreads, OMPThreads)\n");
#ifndef USE_DOUBLE
  else if(!mxIsSingle(prhs[1]) || !mxIsSingle(prhs[2]) )
    mexErrMsgTxt("A, b must be single precision\n");
#else
  else if(!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) )
    mexErrMsgTxt("A, b must be double precision\n");
#endif
  else{
    ptrdiff_t method = mxGetScalar(prhs[0]);       
    REAL *A = (REAL*)mxGetData(prhs[1]);
    REAL *b = (REAL*)mxGetData(prhs[2]);
    ptrdiff_t NSYS =  mxGetScalar(prhs[3]);
    ptrdiff_t isTransposed =  mxGetScalar(prhs[4]); //!0 => A transposed in memory, 0 => A not transposed in memory
    ptrdiff_t M = mxGetScalar(prhs[5]);           //Number of equations
    ptrdiff_t N = mxGetScalar(prhs[6]);           //Number of unknowns
    REAL TOL_TERMINATION = mxGetScalar(prhs[7]);           //Tolerance for termination (0)
    ptrdiff_t MKLT = mxGetScalar(prhs[8]);        //Number of MKL threads
    ptrdiff_t OMPT = mxGetScalar(prhs[9]);        //Number of OMP threads

#ifndef USE_DOUBLE
    plhs[0] = mxCreateNumericMatrix(N, NSYS, mxSINGLE_CLASS, mxREAL);
#else
    plhs[0] = mxCreateNumericMatrix(N, NSYS, mxDOUBLE_CLASS, mxREAL);
#endif
    REAL *x = (REAL*)mxGetPr(plhs[0]);   
   
    //Naive (no update/downdates)
    if(method == 0)
      nnlsOMPSysMKL(A, b, x, isTransposed, MAX_ITER_NNLS(M, N), MAX_ITER_LS(M, N), NSYS, M, N, MKLT, OMPT, TOL_TERMINATION);           //Naive implementation, no update/downdate,  MKL, OMP per system
    else 
      nnlsOMPSysMKLUpdates(A, b, x, isTransposed,  MAX_ITER_NNLS(M, N), MAX_ITER_LS(M, N), NSYS, M, N, MKLT, OMPT, TOL_TERMINATION);   //OMP per system, MKL update/downdate 
  }
  
  
}//end of mexFunction

#include <iostream>

//For mex calls
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  //  mexPrintf("Test\n");
  // prptrdiff_tf("in mex function\n");
  // if(nrhs != 10 || nlhs != 1)
  //   mexErrMsgTxt("Expect: x = NNLS(method, A, b, nSys, isTransposed, mEquations, nUnknowns, tol, MKLThreads, OMPThreads)\n");
  // #ifndef USE_DOUBLE
  //   else if(!mxIsSingle(prhs[1]) || !mxIsSingle(prhs[2]) )
  //     mexErrMsgTxt("A, b must be single precision\n");
  // #else
  //   else if(!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) )
  //     mexErrMsgTxt("A, b must be double precision\n");
  // #endif
  //   else{
  // ptrdiff_t method = mxGetScalar(prhs[0]);       
  const ptrdiff_t method = 0;
  // REAL *A = (REAL*)mxGetData(prhs[0]);
  // REAL *b = (REAL*)mxGetData(prhs[1]);
  REAL *A = (REAL*)mxGetPr(prhs[0]);
  REAL *b = (REAL*)mxGetPr(prhs[1]);
  constexpr ptrdiff_t NSYS=1;
  constexpr ptrdiff_t isTransposed =0;
  ptrdiff_t M = mxGetScalar(prhs[2]);           
  ptrdiff_t N = mxGetScalar(prhs[3]);           
  REAL TOL_TERMINATION = 1e-16;
  const ptrdiff_t MKLT =1;
  const ptrdiff_t OMPT =1;
  // ptrdiff_t NSYS =  mxGetScalar(prhs[3]);
  // ptrdiff_t isTransposed =  mxGetScalar(prhs[4]); //!0 => A transposed in memory, 0 => A not transposed in memory
  // ptrdiff_t M = mxGetScalar(prhs[5]);           //Number of equations
  // ptrdiff_t N = mxGetScalar(prhs[6]);           //Number of unknowns
  // REAL TOL_TERMINATION = mxGetScalar(prhs[7]);           //Tolerance for termination (0)
  // ptrdiff_t MKLT = mxGetScalar(prhs[8]);        //Number of MKL threads
  // ptrdiff_t OMPT = mxGetScalar(prhs[9]);        //Number of OMP threads

#ifndef USE_DOUBLE
  //  plhs[0] = mxCreateNumericMatrix(N, NSYS, mxSINGLE_CLASS, mxREAL);
#else
  //  plhs[0] = mxCreateNumericMatrix(N, NSYS, mxDOUBLE_CLASS, mxREAL);
#endif
  plhs[0] = mxCreateDoubleMatrix((mwSize)N, (mwSize)1, mxREAL);
  REAL *x = (REAL*)mxGetPr(plhs[0]);   
  // initialize
  for (ptrdiff_t i=0; i<N; ++i)
      x[i] = 0.;


  // for (int i=0; i<M*N; ++i)
  //   std::cout << A[i] << std::endl;
  // for (int i=0; i<M; ++i)
  //   std::cout << b[i] << std::endl;


  // std::cout << "init" << std::endl;
  //   for (int i=0; i<N; ++i)
  //   std::cout << x[i] << std::endl;

  // std::cout << M << " " << N <<std::endl;
  
  if (N == 0)
    for (ptrdiff_t i=0; i<N; ++i)
      x[i] = 0.;
  else{
    //Naive (no update/downdates)
    // if(method == 0)
        nnlsOMPSysMKL(A, b, x, isTransposed, MAX_ITER_NNLS(M, N), MAX_ITER_LS(M, N), NSYS, M, N, MKLT, OMPT, TOL_TERMINATION);           //Naive implementation, no update/downdate,  MKL, OMP per system
    // else 
    // CETTE VERSION EST BIEN MEILLEUR
	    // nnlsOMPSysMKLUpdates(A, b, x, isTransposed,  MAX_ITER_NNLS(M, N), MAX_ITER_LS(M, N), NSYS, M, N, MKLT, OMPT, TOL_TERMINATION); 
	    //OMP per system, MKL update/downdate 

  // for (int i=0; i<N; ++i)
  //   std::cout << x[i] << std::endl;

    // }
  } // end trivial case


}//end of mexFunction

