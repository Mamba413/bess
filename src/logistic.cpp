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
Eigen::VectorXd logistic(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, int max_steps = 20, double err = 10e-7) {
  int n = X.rows();
  int p = X.cols();
  double deviance_cur;
  Eigen::MatrixXd X_A(n, p);
  Eigen::VectorXd xbeta(n);
  Eigen::VectorXd xbeta_exp(n);
  Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd pr(n);
  Eigen::VectorXd w(n);
  Eigen::VectorXd z(n);
  Eigen::VectorXd beta_A = beta0;
  double deviance_pre = 10e4;  // initial deviance
  for(int i=0;i<max_steps;i++){
    xbeta = X*beta_A;
    xbeta_exp = xbeta;
    for(int j=0;j<n;j++) {
      if(xbeta_exp(j)>25.0) xbeta_exp(j) = 25.0;
      if(xbeta_exp(j)<-25.0) xbeta_exp(j) = -25.0;
    }
    xbeta_exp = xbeta_exp.array().exp();
    pr = xbeta_exp.array()/(xbeta_exp+one).array();
    deviance_cur = -2*(weights.array()*(y.array()*pr.array().log())+(one-y).array()*(one-pr).array().log()).sum();  // update deviance
    if(abs((deviance_pre - deviance_cur)/deviance_pre)<err)
      break;
    else {
      w = ((pr.cwiseProduct(one-pr)).cwiseProduct(weights)).cwiseSqrt();     // weights
      for(int j=0;j<p;j++){
        X_A.col(j) = X.col(j).cwiseProduct(w);
      }
      z = (y-pr).cwiseQuotient(w);
      beta_A += X_A.colPivHouseholderQr().solve(z);
      deviance_pre = deviance_cur;
    }
  }
  //cout<<deviance_cur<<endl;
  return beta_A;
}
