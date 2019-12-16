#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen\Eigen>
#include "List.h"
#endif

#include <algorithm>
#include <vector>

using namespace std;
vector<int> uniqueR(vector<int> x){
  vector<int>::iterator it;
  sort (x.begin(),x.end());
  it = unique(x.begin(), x.end());
  x.resize(distance(x.begin(),it));
  return x;
}
// [[Rcpp::export]]
Eigen::MatrixXd EigenR(Eigen::MatrixXd X){
  Eigen::EigenSolver<Eigen::MatrixXd> eig(X);
  Eigen::MatrixXd value = eig.pseudoEigenvalueMatrix();
  Eigen::MatrixXd vectors = eig.pseudoEigenvectors();
  Eigen::MatrixXd y = vectors*value.cwiseSqrt()*vectors.adjoint();
  return y;
}
