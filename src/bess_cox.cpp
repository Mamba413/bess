#include <Rcpp.h>
#include <RcppEigen.h>
#include <algorithm>
#include <vector>
#include "normalize.h"
#include "coxph.h"

#include "utilities.h"

// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace std;
double bess_cox_pdas(Eigen::MatrixXd& X, Eigen::VectorXd& status, int T0, int max_steps, Eigen::VectorXd& beta, Eigen::VectorXd& weights, Eigen::VectorXi& A_out, int& l, int cox_max = 20, double eta = 0.5) {
  int n = X.rows();
  int p = X.cols();
  double deviance;
  vector<int>A(T0);
  vector<int>B(T0);
  Eigen::MatrixXd X_A(n, T0);
  Eigen::VectorXd beta_A(T0);
  Eigen::VectorXd l1(p);
  Eigen::VectorXd l2(p);
  Eigen::VectorXd theta = X*beta;
  Eigen::VectorXd cum_theta(n);
  Eigen::VectorXd d(p);
  Eigen::VectorXd bd(p);
  Eigen::VectorXd ratio(n);
  Eigen::MatrixXd xtheta(n,p);
  Eigen::MatrixXd x2theta(n,p);
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(T0);
  for(int i=0;i<=n-1;i++) {
    if(theta(i)>25.0) theta(i) = 25.0;
    if(theta(i)<-25.0) theta(i) = -25.0;
  }
  theta = weights.array()*theta.array().exp();
  cum_theta(n-1) = theta(n-1);
  for(int k=n-2;k>=0;k--) {
    cum_theta(k) = cum_theta(k+1)+theta(k);
  }
  for(int k=0;k<=p-1;k++) {
    xtheta.col(k) = theta.cwiseProduct(X.col(k));
  }
  for(int k=0;k<=p-1;k++) {
    x2theta.col(k) = X.col(k).cwiseProduct(xtheta.col(k));
  }
  for(int k=n-2;k>=0;k--) {
    xtheta.row(k) = xtheta.row(k+1)+xtheta.row(k);
  }
  for(int k=n-2;k>=0;k--) {
    x2theta.row(k) = x2theta.row(k+1)+x2theta.row(k);
  }
  for(int k=0;k<=p-1;k++) {
    xtheta.col(k) = xtheta.col(k).cwiseQuotient(cum_theta);
  }
  for(int k=0;k<=p-1;k++) {
    x2theta.col(k) = x2theta.col(k).cwiseQuotient(cum_theta);
  }
  l1 = -(X - xtheta).adjoint()*(weights.cwiseProduct(status));
  l2 = (x2theta-xtheta.cwiseAbs2()).adjoint()*(weights.cwiseProduct(status));
  d = -l1.cwiseQuotient(l2);
  bd = (beta+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
  for(int k=0;k<=T0-1;k++) {
    bd.maxCoeff(&A[k]);
    bd(A[k]) = 0.0;
    //cout<<A[k]<<" ";
  }
  //cout<<endl;
  sort(A.begin(),A.end());
  for(l=1;l<=max_steps;l++) {
    for(int mm=0;mm<T0;mm++) {
      X_A.col(mm) = X.col(A[mm]);
    }
    beta_A = coxPH(X_A, status, zero, weights, cox_max, eta);  //update beta_A
    beta = Eigen::VectorXd::Zero(p);
    for(int mm=0;mm<T0;mm++) {
      beta(A[mm]) = beta_A(mm);
    }
    theta = X_A*beta_A;
    for(int i=0;i<=n-1;i++) {
      if(theta(i)>25.0) theta(i) = 25.0;
      if(theta(i)<-25.0) theta(i) = -25.0;
    }
    theta = weights.array()*theta.array().exp();
    cum_theta(n-1) = theta(n-1);
    for(int k=n-2;k>=0;k--) {
      cum_theta(k) = cum_theta(k+1)+theta(k);
    }
    for(int k=0;k<=p-1;k++) {
      xtheta.col(k) = theta.cwiseProduct(X.col(k));
    }
    for(int k=0;k<=p-1;k++) {
      x2theta.col(k) = X.col(k).cwiseProduct(xtheta.col(k));
    }
    for(int k=n-2;k>=0;k--) {
      xtheta.row(k) = xtheta.row(k+1)+xtheta.row(k);
    }
    for(int k=n-2;k>=0;k--) {
      x2theta.row(k) = x2theta.row(k+1)+x2theta.row(k);
    }
    for(int k=0;k<=p-1;k++) {
      xtheta.col(k) = xtheta.col(k).cwiseQuotient(cum_theta);
    }
    for(int k=0;k<=p-1;k++) {
      x2theta.col(k) = x2theta.col(k).cwiseQuotient(cum_theta);
    }
    l1 = -(X - xtheta).adjoint()*(weights.cwiseProduct(status));
    l2 = (x2theta-xtheta.cwiseAbs2()).adjoint()*(weights.cwiseProduct(status));
    d = -l1.cwiseQuotient(l2);
    bd = (beta+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
    for(int k=0;k<T0;k++) {
      bd.maxCoeff(&B[k]);
      bd(B[k]) = 0.0;
      //cout<<B[k]<<" ";
    }
    //cout<<endl;
    ratio = (theta.cwiseQuotient(cum_theta)).array().log();
    deviance = -2*(ratio.dot((weights.cwiseProduct(status))));
    //cout<<deviance<<endl;
    sort(B.begin(),B.end());
    if(A == B) break;
    else A = B;
  }
  for(int i=0;i<T0;i++){
    A_out(i) = A[i] + 1;
  }
  ratio = (theta.cwiseQuotient(cum_theta)).array().log();
  deviance = -2*(ratio.dot((weights.cwiseProduct(status))));
  return deviance;
}
// [[Rcpp::export]]
List bess_cox(Eigen::MatrixXd& X, Eigen::VectorXd& status, int T0, int max_steps, Eigen::VectorXd& beta, Eigen::VectorXd& weights, int cox_max = 10, double eta = 0.5, bool normal = true){
  int n = X.rows();
  int p = X.cols();
  int l;
  double deviance;
  double aic;
  double bic;
  double gic;
  Eigen::VectorXd meanx(p);
  Eigen::VectorXd normx(p);
  Eigen::VectorXi A_out(T0);
  if(normal){
    Normalize3(X, weights, meanx, normx);
  }
  deviance = bess_cox_pdas(X, status, T0, max_steps, beta, weights, A_out, l, cox_max, eta);
  aic = deviance + 2*T0;
  bic = deviance + log(double(n))*T0;
  gic = deviance + log(double(p))*log(log(double(n)))*T0;
  if(normal){
    beta = sqrt(double(n))*beta.cwiseQuotient(normx);
  }
//  return List::create(Named("beta") = beta, Named("deviance")=deviance, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic, Named("A")=A_out, Named("l")=l);
  List mylist;
  mylist.add("beta", beta);
  mylist.add("deviance", deviance);
  mylist.add("aic", aic);
  mylist.add("bic", bic);
  mylist.add("gic", gic);
  mylist.add("A", A);
  mylist.add("l", l);
  return mylist;
}
// [[Rcpp::export]]
List bess_coxs(Eigen::MatrixXd& X, Eigen::VectorXd& status, Eigen::VectorXi& T_list, int max_steps, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, bool warm_start = false, int cox_max = 20, double eta = 0.2, bool normal = true){
  int n = X.rows();
  int p = X.cols();
  int m = T_list.size();
  int l;
  double tmp;
  Eigen::VectorXd deviance(m);
  Eigen::VectorXd aic(m);
  Eigen::VectorXd bic(m);
  Eigen::VectorXd gic(m);
  Eigen::VectorXd meanx(p);
  Eigen::VectorXd normx(p);
  Eigen::VectorXi A_out(p);
  Eigen::VectorXd beta = beta0;
  Eigen::MatrixXd beta_out(p, m);
  if(normal){
    Normalize3(X, weights, meanx, normx);
  }
  int i = 0;
  for(i=0;i<m;i++){
    tmp = bess_cox_pdas(X, status, T_list(i), max_steps, beta, weights, A_out, l, cox_max, eta);
    deviance(i) = tmp;
    beta_out.col(i) = beta;
    if(!warm_start) {
      beta = beta0;
    }
    aic(i) = deviance(i) + 2.0*T_list(i);
    bic(i) = deviance(i) + log(double(n))*T_list(i);
    gic(i) = deviance(i) + log(double(p))*log(log(double(n)))*T_list(i);
  }
  if(normal){
    for(i=0;i<m;i++){
      beta_out.col(i) = sqrt(double(n))*beta_out.col(i).cwiseQuotient(normx);
    }
  }
//  return List::create(Named("beta")=beta_out, Named("deviance")=deviance, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic);
  List mylist;
  mylist.add("beta", beta);
  mylist.add("deviance", deviance);
  mylist.add("aic", aic);
  mylist.add("bic", bic);
  mylist.add("gic", gic);
  return mylist;
}
// [[Rcpp::export]]
List bess_cox_gs(Eigen::MatrixXd& X, Eigen::VectorXd& status, int s_min, int s_max, int K_max, int max_steps, double epsilon, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, bool warm_start = false, int cox_max = 20, double eta = 0.2, bool normal = true){
  int n = X.rows();
  int p = X.cols();
  int l;
  int k;
  int sL = s_min;
  int sR = s_max;
  int sM;
  double nulldev;
  double devL1;
  double devL;
  double devR;
  double devM;
  double devML;
  double devMR;
  Eigen::VectorXi T_list(K_max+2);
  Eigen::VectorXd deviance(K_max+2);
  Eigen::VectorXd aic(K_max+2);
  Eigen::VectorXd bic(K_max+2);
  Eigen::VectorXd gic(K_max+2);
  Eigen::VectorXd meanx(p);
  Eigen::VectorXd normx(p);
  Eigen::VectorXi A_out(p);
  Eigen::VectorXd beta_0R = beta0;
  Eigen::VectorXd beta_0L = beta0;
  Eigen::VectorXd beta_0M(p);
  Eigen::VectorXd beta_0MR(p);
  Eigen::VectorXd beta_0ML(p);
  Eigen::MatrixXd beta_out(p, K_max+2);
  if(normal){
    Normalize3(X, weights, meanx, normx);
  }
  Eigen::VectorXd theta = X*beta0;
  Eigen::VectorXd cum_theta(n);
  Eigen::VectorXd ratio(n);
  for(int i=0;i<=n-1;i++) {
    if(theta(i)>25.0) theta(i) = 25.0;
    if(theta(i)<-25.0) theta(i) = -25.0;
  }
  theta = weights.array()*theta.array().exp();
  cum_theta(n-1) = theta(n-1);
  for(int k=n-2;k>=0;k--) {
    cum_theta(k) = cum_theta(k+1)+theta(k);
  }
  ratio = (theta.cwiseQuotient(cum_theta)).array().log();
  nulldev = -2*(ratio.dot((weights.cwiseProduct(status))));
  deviance(0) = bess_cox_pdas(X, status, sL, max_steps, beta_0L, weights, A_out, l, cox_max);
  beta_out.col(0) = beta_0L;
  if(warm_start) beta_0R = beta_0L;
  deviance(1) = bess_cox_pdas(X, status, sR, max_steps, beta_0R, weights, A_out, l, cox_max);
  beta_out.col(1) = beta_0R;
  //cout<<beta_out.leftCols(2)<<endl;
  if(warm_start) beta_0M = beta_0R; else beta_0M = beta0;
  devL = deviance(0);
  devL1 = devL;
  devR = deviance(1);
  T_list(0) = sL;
  T_list(1) = sR;
  for(k=2;k<=K_max+1;k++){
    sM = round(sL + 0.618*(sR - sL));
    T_list(k) = sM;
    deviance(k) = bess_cox_pdas(X, status, sM, max_steps, beta_0M, weights, A_out, l, cox_max);
    //cout<<sL<<" "<<sM<<" "<<sR<<endl;
    beta_out.col(k) = beta_0M;
    if(!warm_start) beta_0M = beta0;
    devM = deviance(k);
    if((abs(devL - devM)/abs(nulldev*(sM - sL)) > epsilon) && (abs(devR - devM)/abs(nulldev*(sM - sR)) < epsilon)) {
      sR = sM;
      devR = devM;
    } else if((abs(devL - devM)/abs(nulldev*(sM - sL)) > epsilon) && (abs(devR - devM)/abs(nulldev*(sM - sR)) > epsilon)) {
      sL = sM;
      devL = devM;
    } else {
      sR = sM;
      devR = devM;
      sL = s_min;
      devL = devL1;
    }
    if(sR - sL == 1) break;
    if(warm_start) {
      beta_0ML = beta_0M;
      beta_0MR = beta_0M;
    } else {
      beta_0ML = beta0;
      beta_0MR = beta0;
    }
    devML = bess_cox_pdas(X, status, sM-1, max_steps, beta_0ML, weights, A_out, l, cox_max);
    devMR = bess_cox_pdas(X, status, sM+1, max_steps, beta_0MR, weights, A_out, l, cox_max);
    if((abs(devML - devM)/nulldev > epsilon) && (2*abs(devMR - devM)/nulldev < epsilon)) break;
  }
  if(k>K_max+1) k = K_max+1;
  if(normal){
    for(int kk=0;kk<=k;kk++){
      beta_out.col(kk) = sqrt(double(n))*beta_out.col(kk).cwiseQuotient(normx);
    }
  }
  for(int i=0;i<=k;i++){
    aic(i) = deviance(i)+2.0*T_list(i);
    bic(i) = deviance(i)+log(double(n))*T_list(i);
    gic(i) = deviance(i)+log(double(p))*log(log(double(n)))*T_list(i);
  }
//  return List::create(Named("beta")=beta_out.leftCols(k+1), Named("s_list")=T_list.head(k+1), Named("dev")=deviance.head(k+1), Named("nulldev")=nulldev, Named("aic")=aic.head(k+1), Named("bic")=bic.head(k+1), Named("gic")=gic.head(k+1));
  #ifdef R_BUILD
    return List::create(Named("beta")=beta, Named("coef0")=coef0, Named("deviance")=deviance, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic, Named("A")=A_out, Named("l")=l);
  #else
    List mylist;
    mylist.add("beta", beta_out.leftCols(k+1));
    mylist.add("s_list", T_list.head(k+1))
    mylist.add("dev", deviance.head(k+1));
    mylist.add("nulldev", nulldev)
    mylist.add("aic", aic.head(k+1));
    mylist.add("bic", bic.head(k+1));
    mylist.add("gic", gic.head(k+1));
    return mylist;
  #endif
}

void pywrap_bess_cox(double* X, int X_row, int X_col, double * status, int status_len, int T0, int max_steps, double * beta, int beta_len, double * weights, int weights_len, int cox_max = 10, double eta = 0.5, bool normal = true)
{
    Eigen::MatrixXd X_Mat;
    Eigen::VectorXd status_Vec;
    Eigen::VectorXd beta_Vec;
    Eigen::VectorXd weights_Vec;

    X_Mat = array2MatrixXd(X, X_row, X_col);
    status_Vec = array2VectorXd(status, status_len);
    beta_Vec = array2VectorXd(beta, beta_len);
    weights_Vec = array2VectorXd(weights, weights_len);

    List result = bess_cox(X_Mat, status_Vec, T0, max_steps, beta_Vec, weights_Vec, cox_max, eta, normal);




}



