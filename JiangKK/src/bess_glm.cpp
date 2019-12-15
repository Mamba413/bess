#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
#else
#include <Eigen\Eigen>
#include "List.h"
#endif

#include <algorithm>
#include <vector>
#include "normalize.h"
#include "logistic.h"
// [[Rcpp::depends(RcppEigen)]]
//using namespace Rcpp;
using namespace std;
double bess_glm_pdas(Eigen::MatrixXd& X, Eigen::VectorXd& y, int T0, int max_steps, Eigen::VectorXd& beta, double& coef0, Eigen::VectorXd& weights, Eigen::VectorXi& A_out, int& l, int glm_max = 20) {
  int n = X.rows();
  int p = X.cols();
  vector<int>A(T0);
  vector<int>B(T0);
  Eigen::MatrixXd X_A(n, T0+1);
  Eigen::MatrixXd Xsquare(n, p);
  X_A.col(T0) = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd beta_A(T0+1);
  Eigen::VectorXd bd(p);
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(T0+1);
  Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd coef(n);
  for(int i=0;i<=n-1;i++) {
    coef(i) = coef0;
  }
  Eigen::VectorXd xbeta_exp = X*beta+coef;
  for(int i=0;i<=n-1;i++) {
    if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
    if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
  }
  xbeta_exp = xbeta_exp.array().exp();
  Eigen::VectorXd pr = xbeta_exp.array()/(xbeta_exp+one).array();
  Eigen::VectorXd l1 = -X.adjoint()*((y-pr).cwiseProduct(weights));
  Xsquare = X.array().square();
  Eigen::VectorXd l2 = (Xsquare.adjoint())*((pr.cwiseProduct(one-pr)).cwiseProduct(weights));
  Eigen::VectorXd d = -l1.cwiseQuotient(l2);
  bd = (beta+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
  for(int k=0;k<=T0-1;k++) {
    bd.maxCoeff(&A[k]);
    bd(A[k]) = 0.0;
  }
  sort(A.begin(),A.end());
  double deviance;
  for(l=1;l<=max_steps;l++) {
    for(int mm=0;mm<T0;mm++) {
      X_A.col(mm) = X.col(A[mm]);
    }
    beta_A = logistic(X_A, y, zero, weights, glm_max);  //update beta_A
    beta = Eigen::VectorXd::Zero(p);
    for(int mm=0;mm<T0;mm++) {
      beta(A[mm]) = beta_A(mm);
    }
    coef0 = beta_A(T0);
    xbeta_exp = X_A*beta_A;
    for(int i=0;i<=n-1;i++) {
      if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
      if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
    }
    xbeta_exp = xbeta_exp.array().exp();
    pr = xbeta_exp.array()/(xbeta_exp+one).array();
    l1 = -X.adjoint()*((y-pr).cwiseProduct(weights));
    l2 = (Xsquare.adjoint())*((pr.cwiseProduct(one-pr)).cwiseProduct(weights));
    d = -l1.cwiseQuotient(l2);
    bd = (beta+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
    for(int k=0;k<T0;k++) {
      bd.maxCoeff(&B[k]);
      bd(B[k]) = 0.0;
    }
    sort(B.begin(),B.end());
    if(A==B) break;
    else A = B;
  }
  for(int i=0;i<T0;i++){
    A_out(i) = A[i] + 1;
  }
  deviance = -2*(weights.array()*(y.array()*pr.array().log())+(one-y).array()*(one-pr).array().log()).sum();
  return deviance;
}
// [[Rcpp::export]]
List bess_glm(Eigen::MatrixXd& X, Eigen::VectorXd& y, int T0, int max_steps, Eigen::VectorXd& beta, double& coef0, Eigen::VectorXd& weights, int glm_max = 20, bool normal = true){
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
  deviance = bess_glm_pdas(X, y, T0, max_steps, beta, coef0, weights, A_out, l, glm_max);
  aic = double(n)*log(deviance)+2*T0;
  bic = double(n)*log(deviance)+log(double(n))*T0;
  gic = double(n)*log(deviance)+log(double(p))*log(log(double(n)))*T0;
  if(normal){
    beta = sqrt(double(n))*beta.cwiseQuotient(normx);
    coef0 = coef0 - beta.dot(meanx);
  }
  #ifdef R_BUILD
    return List::create(Named("beta")=beta, Named("coef0")=coef0, Named("deviance")=deviance, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic, Named("A")=A_out, Named("l")=l);
  #else
    List mylist;
    mylist.add("beta", beta);
    mylist.add("coef0", coef0);
    mylist.add("deviance", deviance);
    mylist.add("aic", aic);
    mylist.add("bic", bic);
    mylist.add("gic", gic);
    mylist.add("A", A);
    mylist.add("l", l);
    return mylist;
  #endif
}

// [[Rcpp::export]]
List bess_glms(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXd& T_list, int max_steps, Eigen::VectorXd& beta0, double& intercept, Eigen::VectorXd& weights, bool warm_start = false, int glm_max = 20, bool normal = true){
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
  double coef0_ = intercept;
  Eigen::MatrixXd beta_out(p, m);
  Eigen::VectorXd coef0(m);
  if(normal){
    Normalize3(X, weights, meanx, normx);
  }
  int i = 0;
  for(i=0;i<m;i++){
    tmp = bess_glm_pdas(X, y, T_list(i), max_steps, beta, coef0_, weights, A_out, l, glm_max);
    deviance(i) = tmp;
    beta_out.col(i) = beta;
    coef0(i) = coef0_;
    if(!warm_start) {
      beta = beta0;
      coef0_ = intercept;
    }
    aic(i) = deviance(i)+2.0*T_list(i);
    bic(i) = deviance(i)+log(double(n))*T_list(i);
    gic(i) = deviance(i)+log(double(p))*log(log(double(n)))*T_list(i);
  }
  if(normal){
    for(i=0;i<m;i++){
      beta_out.col(i) = sqrt(double(n))*beta_out.col(i).cwiseQuotient(normx);
      coef0(i) = coef0(i) - beta_out.col(i).dot(meanx);
    }
  }
//  return List::create(Named("beta")=beta_out, Named("coef0")=coef0, Named("deviance")=deviance, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic);
  #ifdef R_BUILD
    return List::create(Named("beta")=beta, Named("coef0")=coef0, Named("deviance")=deviance, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic, Named("A")=A_out, Named("l")=l);
  #else
    List mylist;
    mylist.add("beta", beta);
    mylist.add("coef0", coef0);
    mylist.add("deviance", deviance);
    mylist.add("aic", aic);
    mylist.add("bic", bic);
    mylist.add("gic", gic);
    mylist.add("A", A);
    mylist.add("l", l);
    return mylist;
  #endif
}
// [[Rcpp::export]]
List bess_glm_gs(Eigen::MatrixXd& X, Eigen::VectorXd& y, int s_min, int s_max, int K_max, int max_steps, double epsilon, Eigen::VectorXd& beta0, double coef0, Eigen::VectorXd& weights, bool warm_start = false, bool normal = true, int glm_max = 20){
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
  double coef0_0R = coef0;
  double coef0_0L = coef0;
  double coef0_0M;
  double coef0_0MR;
  double coef0_0ML;
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
  Eigen::VectorXd coef0_out(K_max+2);
  if(normal){
    Normalize3(X, weights, meanx, normx);
  }
  nulldev = -2*log(0.5)*weights.sum();
  deviance(0) = bess_glm_pdas(X, y, sL, max_steps, beta_0L, coef0_0L, weights, A_out, l, glm_max);
  beta_out.col(0) = beta_0L;
  coef0_out(0) = coef0_0L;
  if(warm_start) {
    beta_0R = beta_0L;
    coef0_0R = coef0_0L;
  }
  deviance(1) = bess_glm_pdas(X, y, sR, max_steps, beta_0R, coef0_0R, weights, A_out, l, glm_max);
  beta_out.col(1) = beta_0R;
  coef0_out(1) = coef0_0R;
  //cout<<beta_out.leftCols(2)<<endl;
  if(warm_start) {
    beta_0M = beta_0R;
    coef0_0M = coef0_0R;
  } else {
    beta_0M = beta0;
    coef0_0M = coef0;
  }
  devL = deviance(0);
  devL1 = devL;
  devR = deviance(1);
  T_list(0) = sL;
  T_list(1) = sR;
  for(k=2;k<=K_max+1;k++){
    sM = round(sL + 0.618*(sR - sL));
    T_list(k) = sM;
    deviance(k) = bess_glm_pdas(X, y, sM, max_steps, beta_0M, coef0_0M, weights, A_out, l, glm_max);
    //cout<<sL<<" "<<sM<<" "<<sR<<endl;
    beta_out.col(k) = beta_0M;
    coef0_out(k) = coef0_0M;
    if(!warm_start) {
      beta_0M = beta0;
      coef0_0M = coef0;
    }
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
      coef0_0ML = coef0_0M;
      coef0_0MR = coef0_0M;
    } else {
      beta_0ML = beta0;
      beta_0MR = beta0;
      coef0_0ML = coef0;
      coef0_0MR = coef0;
    }
    devML = bess_glm_pdas(X, y, sM-1, max_steps, beta_0ML, coef0_0ML, weights, A_out, l, glm_max);
    devMR = bess_glm_pdas(X, y, sM+1, max_steps, beta_0MR, coef0_0MR, weights, A_out, l, glm_max);
    if((abs(devML - devM)/nulldev > epsilon) && (2*abs(devMR - devM)/nulldev < epsilon)) break;
  }
  if(k>K_max+1) k = K_max+1;
  if(normal){
    for(int kk=0;kk<=k;kk++){
      beta_out.col(kk) = sqrt(double(n))*beta_out.col(kk).cwiseQuotient(normx);
      coef0_out(kk) = coef0_out(kk) - beta_out.col(kk).dot(meanx);
    }
  }
  for(int i=0;i<=k;i++){
    aic(i) = deviance(i)+2.0*T_list(i);
    bic(i) = deviance(i)+log(double(n))*T_list(i);
    gic(i) = deviance(i)+log(double(p))*log(log(double(n)))*T_list(i);
  }
//  return List::create(Named("beta")=beta_out.leftCols(k+1), Named("coef0")=coef0_out.head(k+1), Named("s_list")=T_list.head(k+1), Named("dev")=deviance.head(k+1), Named("nulldev")=nulldev, Named("aic")=aic.head(k+1), Named("bic")=bic.head(k+1), Named("gic")=gic.head(k+1));
  #ifdef R_BUILD
    return List::create(Named("beta")=beta, Named("coef0")=coef0, Named("deviance")=deviance, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic, Named("A")=A_out, Named("l")=l);
  #else
    List mylist;
    mylist.add("beta", beta_out.leftCols(k+1));
    mylist.add("coef0", coef0_out.head(k+1));
    mylist.add("s_list", T_list.head(k+1))
    mylist.add("deviance", deviance.head(k+1));
    mylist.add("nulldev", nulldev)
    mylist.add("aic", aic.head(k+1));
    mylist.add("bic", bic.head(k+1));
    mylist.add("gic", gic.head(k+1));
    return mylist;
  #endif
}
