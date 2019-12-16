#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
#else
#include <Eigen\Eigen>
#include "List.h"
#include "utilities.h"
#include "bess_lm.h"
#endif

#include <algorithm>
#include <vector>
#include "normalize.h"

// [[Rcpp::depends(RcppEigen)]]
using namespace std;
void bess_lm_pdas(Eigen::MatrixXd& X, Eigen::VectorXd& y, int T0, int max_steps, Eigen::VectorXd& beta, Eigen::VectorXi& A_out, int& l)
{
  int n = X.rows();
  int p = X.cols();
  vector<int>A(T0);
  vector<int>B(T0);
  Eigen::MatrixXd X_A(n, T0);
  Eigen::VectorXd beta_A(T0);
  Eigen::VectorXd res = (y-X*beta)/double(n);
  Eigen::VectorXd d(p);
  for(int i=0;i<p;i++){
    d(i) = res.dot(X.col(i));
  }
  Eigen::VectorXd bd = beta+d;
  bd = bd.cwiseAbs();
  for(int k=0;k<T0;k++) {             //update A
    bd.maxCoeff(&A[k]);
    bd(A[k]) = 0.0;
  }
  sort(A.begin(),A.end());
  for(l=1;l<=max_steps;l++) {
    for(int mm=0;mm<T0;mm++) {
      X_A.col(mm) = X.col(A[mm]);
    }
    beta_A = X_A.colPivHouseholderQr().solve(y);  //update beta_A
    beta = Eigen::VectorXd::Zero(p);
    for(int mm=0;mm<T0;mm++) {
      beta(A[mm]) = beta_A(mm);
    }
    res = (y-X_A*beta_A)/double(n);
    for(int mm=0;mm<p;mm++){     //update d_I
      bd(mm) = res.dot(X.col(mm));
    }
    for(int mm=0;mm<T0;mm++) {
      bd(A[mm]) = beta_A(mm);
    }
    bd = bd.cwiseAbs();
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
}

// [[Rcpp::export]]
List bess_lm(Eigen::MatrixXd& X, Eigen::VectorXd& y, int T0, int max_steps, Eigen::VectorXd& beta, Eigen::VectorXd& weights, bool normal)
{
  int n = X.rows();
  int p = X.cols();
  int l;
  double mse;
  double nullmse;
  double aic;
  double bic;
  double gic;
  double coef0 = 0.0;
  Eigen::VectorXd meanx(p);
  Eigen::VectorXd normx(p);
  Eigen::VectorXi A_out(T0);
  double meany = 0.0;
  if(normal){
    Normalize(X, y, weights, meanx, meany, normx);
  }
  for(int i=0;i<n;i++){
    X.row(i) = X.row(i)*sqrt(weights(i));
    y(i) = y(i)*sqrt(weights(i));
  }
  bess_lm_pdas(X, y, T0, max_steps, beta, A_out, l);
  mse = (y-X*beta).squaredNorm()/double(n);
  nullmse = y.squaredNorm()/double(n);
  aic = double(n)*log(mse)+2.0*T0;
  bic = double(n)*log(mse)+log(double(n))*T0;
  gic = double(n)*log(mse)+log(double(p))*log(log(double(n)))*T0;
  if(normal){
    beta = sqrt(double(n))*beta.cwiseQuotient(normx);
    coef0 = meany - beta.dot(meanx);
  }
  #ifdef R_BUILD
  return List::create(Named("beta")=beta, Named("coef0")=coef0, Named("mse")=mse, Named("nullmse")=nullmse, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic, Named("A")=A_out);
  #else
  List mylist;
  mylist.add("beta", beta);
  mylist.add("coef0", coef0);
  mylist.add("mse", mse);
  mylist.add("nullmse", nullmse);
  mylist.add("aic", aic);
  mylist.add("bic", bic);
  mylist.add("gic", gic);
  mylist.add("A", A_out);
  return mylist;
  #endif
}

// [[Rcpp::export]]
List bess_lms(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXi& T_list, int max_steps, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, bool warm_start, bool normal)
{
  int n = X.rows();
  int p = X.cols();
  int l;
  int m = T_list.size();
  double nullmse;
  Eigen::VectorXd mse(m);
  Eigen::VectorXd aic(m);
  Eigen::VectorXd bic(m);
  Eigen::VectorXd gic(m);
  Eigen::VectorXd coef0(m);
  Eigen::VectorXd meanx(p);
  Eigen::VectorXd normx(p);
  Eigen::VectorXi A_out(p);
  Eigen::VectorXd beta = beta0;
  Eigen::MatrixXd beta_out(p, m);
  double meany = 0.0;
  if(normal){
    Normalize(X, y, weights, meanx, meany, normx);
  }
  int i = 0;
  for(i=0;i<n;i++){
    X.row(i) = X.row(i)*sqrt(weights(i));
    y(i) = y(i)*sqrt(weights(i));
  }
  for(i=0;i<m;i++){
    bess_lm_pdas(X, y, T_list(i), max_steps, beta, A_out, l);
    beta_out.col(i) = beta;
    if(!warm_start) beta = beta0;
    mse(i) = (y-X*beta_out.col(i)).squaredNorm()/double(n);
    aic(i) = double(n)*log(mse(i))+2.0*T_list(i);
    bic(i) = double(n)*log(mse(i))+log(double(n))*T_list(i);
    gic(i) = double(n)*log(mse(i))+log(double(p))*log(log(double(n)))*T_list(i);
  }
  nullmse = y.squaredNorm()/double(n);
  // cout<<m<<endl;
  if(normal){
    for(i=0;i<m;i++){
      beta_out.col(i) = sqrt(double(n))*beta_out.col(i).cwiseQuotient(normx);
      coef0(i) = meany - beta_out.col(i).dot(meanx);
    }
  }
  #ifdef R_BUILD
  return List::create(Named("beta")=beta_out, Named("coef0")=coef0, Named("mse")=mse, Named("nullmse")=nullmse, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic);
  #else
  List mylist;
  mylist.add("beta", beta_out);
  mylist.add("coef0", coef0);
  mylist.add("mse", mse);
  mylist.add("nullmse", nullmse);
  mylist.add("aic", aic);
  mylist.add("bic", bic);
  mylist.add("gic", gic);
  return mylist;
  #endif
}

// [[Rcpp::export]]
List bess_lm_gs(Eigen::MatrixXd& X, Eigen::VectorXd& y, int s_min, int s_max, int K_max, int max_steps, double epsilon, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, bool warm_start, bool normal)
{
  int n = X.rows();
  int p = X.cols();
  int l;
  int k;
  int sL = s_min;
  int sR = s_max;
  int sM;
  double nullmse;
  double mseL1;
  double mseL;
  double mseR;
  double mseM;
  double mseML;
  double mseMR;
  Eigen::VectorXi T_list(K_max+2);
  Eigen::VectorXd mse(K_max+2);
  Eigen::VectorXd aic(K_max+2);
  Eigen::VectorXd bic(K_max+2);
  Eigen::VectorXd gic(K_max+2);
  Eigen::VectorXd coef0(K_max+2);
  Eigen::VectorXd meanx(p);
  Eigen::VectorXd normx(p);
  Eigen::VectorXi A_out(p);
  Eigen::VectorXd beta_0R = beta0;
  Eigen::VectorXd beta_0L = beta0;
  Eigen::VectorXd beta_0M(p);
  Eigen::VectorXd beta_0MR(p);
  Eigen::VectorXd beta_0ML(p);
  Eigen::MatrixXd beta_out(p, K_max+2);
  double meany = 0.0;
  if(normal){
    Normalize(X, y, weights, meanx, meany, normx);
  }
  for(int i=0;i<n;i++){
    X.row(i) = X.row(i)*sqrt(weights(i));
    y(i) = y(i)*sqrt(weights(i));
  }
  bess_lm_pdas(X, y, sL, max_steps, beta_0L, A_out, l);
  beta_out.col(0) = beta_0L;
  if(warm_start) beta_0R = beta_0L;
  bess_lm_pdas(X, y, sR, max_steps, beta_0R, A_out, l);
  beta_out.col(1) = beta_0R;
  //cout<<beta_out.leftCols(2)<<endl;
  if(warm_start) beta_0M = beta_0R; else beta_0M = beta0;
  beta_0M = beta_0R;
  mse(0) = (y-X*beta_out.col(0)).squaredNorm()/double(n);
  mse(1) = (y-X*beta_out.col(1)).squaredNorm()/double(n);
  mseL = mse(0);
  mseL1 = mseL;
  mseR = mse(1);
  nullmse = y.squaredNorm()/double(n);
  T_list(0) = sL;
  T_list(1) = sR;
  for(k=2;k<=K_max+1;k++){
    sM = round(sL + 0.618*(sR - sL));
    T_list(k) = sM;
    bess_lm_pdas(X, y, sM, max_steps, beta_0M, A_out, l);
    //cout<<sL<<" "<<sM<<" "<<sR<<endl;
    beta_out.col(k) = beta_0M;
    if(!warm_start) beta_0M = beta0;
    mse(k) = (y-X*beta_out.col(k)).squaredNorm()/double(n);
    mseM = mse(k);
    if((abs(mseL - mseM)/abs(nullmse*(sM - sL)) > epsilon) && (abs(mseR - mseM)/abs(nullmse*(sM - sR)) < epsilon)) {
      sR = sM;
      mseR = mseM;
    } else if((abs(mseL - mseM)/abs(nullmse*(sM - sL)) > epsilon) && (abs(mseR - mseM)/abs(nullmse*(sM - sR)) > epsilon)) {
      sL = sM;
      mseL = mseM;
    } else {
      sR = sM;
      mseR = mseM;
      sL = s_min;
      mseL = mseL1;
    }
    if(sR - sL == 1) break;
    if(warm_start) {
      beta_0ML = beta_0M;
      beta_0MR = beta_0M;
    } else {
      beta_0ML = beta0;
      beta_0MR = beta0;
    }
    bess_lm_pdas(X, y, sM-1, max_steps, beta_0ML, A_out, l);
    bess_lm_pdas(X, y, sM+1, max_steps, beta_0MR, A_out, l);
    mseML = (y-X*beta_0ML).squaredNorm()/double(n);
    mseMR = (y-X*beta_0MR).squaredNorm()/double(n);
    if((abs(mseML - mseM)/nullmse > epsilon) && (2*abs(mseMR - mseM)/nullmse < epsilon)) break;
  }
  if(k>K_max+1) k = K_max+1;
  if(normal){
    for(int kk=0;kk<=k;kk++){
      beta_out.col(kk) = sqrt(double(n))*beta_out.col(kk).cwiseQuotient(normx);
      coef0(kk) = meany - beta_out.col(kk).dot(meanx);
    }
  }
  for(int i=0;i<=k;i++){
    aic(i) = double(n)*log(mse(i))+2.0*T_list(i);
    bic(i) = double(n)*log(mse(i))+log(double(n))*T_list(i);
    gic(i) = double(n)*log(mse(i))+log(double(p))*log(log(double(n)))*T_list(i);
  }
  #ifdef R_BUILD
  return List::create(Named("beta")=beta_out.leftCols(k+1), Named("coef0")=coef0.head(k+1), Named("s_list")=T_list.head(k+1), Named("mse")=mse.head(k+1), Named("nullmse")=nullmse, Named("aic")=aic.head(k+1), Named("bic")=bic.head(k+1), Named("gic")=gic.head(k+1));
  #else
  List mylist;
  mylist.add("beta", beta_out.leftCols(k+1).eval());
  mylist.add("coef0", coef0.head(k+1).eval());
  mylist.add("s_list", T_list.head(k+1).eval());
  mylist.add("mse", mse.head(k+1).eval());
  mylist.add("nullmse", nullmse);
  mylist.add("aic", aic.head(k+1).eval());
  mylist.add("bic", bic.head(k+1).eval());
  mylist.add("gic", gic.head(k+1).eval());
  return mylist;
  #endif
}


#ifndef R_BUILD

void pywrap_bess_lm(double* X, int X_row, int X_col, double* y, int y_len, int T0, int max_steps, double* beta, int beta_len, double* weights, int weights_len,
                    double* coef0, double* beta_return, int beta_return_len, double* mse, double * nullmse, double* aic, double* bic, double* gic, int* A, int A_len, bool normal)
{
    Eigen::MatrixXd X_Mat;
    Eigen::VectorXd y_Vec;
    Eigen::VectorXd beta_Vec;
    Eigen::VectorXd weights_Vec;

    X_Mat = Pointer2MatrixXd(X, X_row, X_col);
    y_Vec = Pointer2VectorXd(y, y_len);
    beta_Vec = Pointer2VectorXd(beta, beta_len);
    weights_Vec = Pointer2VectorXd(weights, weights_len);

    List mylist = bess_lm(X_Mat, y_Vec, T0, max_steps, beta_Vec, weights_Vec, normal);

    Eigen::VectorXd temp_VectorXd;
    Eigen::VectorXi temp_VectorXi;

    temp_VectorXd = mylist.get_value_by_name("beta", temp_VectorXd);
    VectorXd2Pointer(temp_VectorXd, beta_return);
//    beta_return_len = temp_VectorXd.size();

    *coef0 = mylist.get_value_by_name("coef0", *coef0);
    *mse = mylist.get_value_by_name("mse", *mse);
    *nullmse = mylist.get_value_by_name("nullmse", *nullmse);
    *aic = mylist.get_value_by_name("aic", *aic);
    *bic = mylist.get_value_by_name("bic", *bic);
    *gic = mylist.get_value_by_name("gic", *gic);

    temp_VectorXi = mylist.get_value_by_name("A", temp_VectorXi);
    VectorXi2Pointer(temp_VectorXi, A);
}

void pywrap_bess_lms(double* X, int X_row, int X_col, double* y, int y_len, int* T_list, int T_list_len, int max_steps, double* beta0, int beta0_len, double* weights, int weights_len,
                     double* coef0, double* beta, int beta_len, double* mse, double* nullmse, double* aic, double* bic, double* gic, bool warm_start, bool normal)
{
    Eigen::MatrixXd X_Mat;
    Eigen::VectorXd y_Vec;
    Eigen::VectorXi T_list_Vec;
    Eigen::VectorXd beta0_Vec;
    Eigen::VectorXd weights_Vec;

    X_Mat = Pointer2MatrixXd(X, X_row, X_col);
    y_Vec = Pointer2VectorXd(y, y_len);
    T_list_Vec = Pointer2VectorXi(T_list, T_list_len);
    beta0_Vec = Pointer2VectorXd(beta0, beta_len);
    weights_Vec = Pointer2VectorXd(weights, weights_len);

    List mylist = bess_lms(X_Mat, y_Vec, T_list_Vec, max_steps, beta0_Vec, weights_Vec, warm_start, normal);

    Eigen::VectorXd temp_VectorXd;
    Eigen::VectorXi temp_VectorXi;

    temp_VectorXd = mylist.get_value_by_name("beta", temp_VectorXd);
    VectorXd2Pointer(temp_VectorXd, beta);

    *coef0 = mylist.get_value_by_name("coef0", *coef0);
    *mse = mylist.get_value_by_name("mse", *mse);
    *nullmse = mylist.get_value_by_name("nullmse", *nullmse);
    *aic = mylist.get_value_by_name("aic", *aic);
    *bic = mylist.get_value_by_name("bic", *bic);
    *gic = mylist.get_value_by_name("gic", *gic);
}

void pywrap_bess_lm_gs(double* X, int X_row, int X_col, double* y, int y_len, int s_min, int s_max, int K_max, int max_steps, double epsilon, double* beta0, int beta0_len, double* weights, int weights_len,
                       double* coef0, double* beta, int beta_len, double* mse, double* nullmse, double* aic, double* bic, double* gic, bool warm_start, bool normal)
{
    Eigen::MatrixXd X_Mat;
    Eigen::VectorXd y_Vec;
    Eigen::VectorXd beta0_Vec;
    Eigen::VectorXd weights_Vec;

    X_Mat = Pointer2MatrixXd(X, X_row, X_col);
    y_Vec = Pointer2VectorXd(y, y_len);
    beta0_Vec = Pointer2VectorXd(beta0, beta_len);
    weights_Vec = Pointer2VectorXd(weights, weights_len);

    List mylist = bess_lm_gs(X_Mat, y_Vec, s_min, s_max, K_max, max_steps, epsilon, beta0_Vec, weights_Vec, warm_start, normal);

    Eigen::VectorXd temp_VectorXd;
    Eigen::VectorXi temp_VectorXi;

    temp_VectorXd = mylist.get_value_by_name("beta", temp_VectorXd);
    VectorXd2Pointer(temp_VectorXd, beta);

    *coef0 = mylist.get_value_by_name("coef0", *coef0);
    *mse = mylist.get_value_by_name("mse", *mse);
    *nullmse = mylist.get_value_by_name("nullmse", *nullmse);
    *aic = mylist.get_value_by_name("aic", *aic);
    *bic = mylist.get_value_by_name("bic", *bic);
    *gic = mylist.get_value_by_name("gic", *gic);
}

#endif

