#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
#else
#include <Eigen\Eigen>
#include "List.h"
#endif
#include <algorithm>
#include <vector>

using namespace std;
Eigen::VectorXd coxPH(Eigen::MatrixXd& X, Eigen::VectorXd& status, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, int max_steps = 20, double eta = 0.2, double err = 10e-7) {
  int n = X.rows();
  int p = X.cols();
  double deviance_cur;
  Eigen::VectorXd beta = beta0;
  Eigen::VectorXd theta(n);
  Eigen::VectorXd cum_theta(n);
  Eigen::VectorXd ratio(n);
  Eigen::MatrixXd xtheta(n, p);
  Eigen::VectorXd xijtheta(n);
  Eigen::VectorXd score(p);
  Eigen::MatrixXd Hessian(p, p);
  double deviance_pre = 10e4;  // initial deviance
  for(int i=0;i<max_steps;i++){
    theta = X*beta;
    // initial A
    for(int k=0;k<n;k++) {
      if(theta(k)>25.0) theta(k) = 25.0;
      if(theta(k)<-25.0) theta(k) = -25.0;
    }
    theta = weights.array()*theta.array().exp();
    cum_theta(n-1) = theta(n-1);
    for(int k=n-2;k>=0;k--) {
      cum_theta(k) = cum_theta(k+1)+theta(k);
    }
    ratio = (theta.cwiseQuotient(cum_theta)).array().log();
    deviance_cur = -2*(ratio.dot((weights.cwiseProduct(status))));
    //cout<<deviance_cur<<endl;
    if(abs((deviance_pre - deviance_cur)/deviance_pre)<err)
      break;
    else {
      for(int k=0;k<p;k++) {
        xtheta.col(k) = theta.cwiseProduct(X.col(k));
      }
      for(int k=n-2;k>=0;k--) {
        xtheta.row(k) = xtheta.row(k+1)+xtheta.row(k);
      }
      for(int k=0;k<=p-1;k++) {
        xtheta.col(k) = xtheta.col(k).cwiseQuotient(cum_theta);
      }
      score = (X - xtheta).adjoint()*(weights.cwiseProduct(status));
      for(int k1=0;k1<p;k1++)
        for(int k2=k1;k2<p;k2++) {
          xijtheta = (theta.cwiseProduct(X.col(k1))).cwiseProduct(X.col(k2));
          for(int j=n-2;j>=0;j--) {
            xijtheta(j) = xijtheta(j+1) + xijtheta(j);
          }
          Hessian(k1, k2) = -(xijtheta.cwiseQuotient(cum_theta) - xtheta.col(k1).cwiseProduct(xtheta.col(k2))).dot(weights.cwiseProduct(status));
          Hessian(k2, k1) = Hessian(k1, k2);
        }
      beta += -eta*Hessian.colPivHouseholderQr().solve(score);            // update beta
      deviance_pre = deviance_cur;
    }
  }
  //cout<<deviance_cur<<endl;
  return beta;
}
