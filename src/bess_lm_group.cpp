#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
#else
#include <Eigen.h>
#include "List.h"
#endif
#include <algorithm>
#include <vector>
#include "normalize.h"
#include "tmp.h"
// [[Rcpp::depends(RcppEigen)]]
//using namespace Rcpp;
using namespace std;
int gbess_lm_pdas(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXi& G, Eigen::VectorXi index, Eigen::VectorXi orderGi, List& PhiG, List& invPhiG, int T0, int max_steps, Eigen::VectorXd& beta0, int n, int p, int N, int& l, int& mark){
  Eigen::VectorXi B0 = Eigen::VectorXi::Zero(p);
  Eigen::VectorXd d0 = (X.transpose()*(y-X*beta0)) /double(n);
  Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
  vector<int>A(T0);
  vector<int>A0(T0);
  for(int i=0;i<T0;i++){
    A0[i] = 1;
  }
  vector<int>gr(T0);
  mark = 0;
  int gr_size = 0;
  for(l=1;l<=max_steps;l++) {
    for(int i=0;i<N-1;i++){
      Eigen::MatrixXd phiG = PhiG[i];
      Eigen::MatrixXd invphiG = invPhiG[i];
      betabar.segment(index(i)-1,index(i+1)-index(i)) = phiG*beta0.segment(index(i)-1,index(i+1)-index(i));
      dbar.segment(index(i)-1,index(i+1)-index(i)) = invphiG*d0.segment(index(i)-1,index(i+1)-index(i));
    }
    Eigen::MatrixXd phiG = PhiG[N-1];
    Eigen::MatrixXd invphiG = invPhiG[N-1];
    betabar.segment(index(N-1)-1,p-index(N-1)+1) = phiG*beta0.segment(index(N-1)-1,p-index(N-1)+1);
    dbar.segment(index(N-1)-1,p-index(N-1)+1) = invphiG*d0.segment(index(N-1)-1,p-index(N-1)+1);
    Eigen::VectorXd bd = (betabar+dbar).array().abs();
    for(int k=0;k<T0;k++) {
      bd.maxCoeff(&A[k]);
      bd(A[k])=0;
    }
    sort(A.begin(),A.end());
    if(A==A0) break;
    for(int i=0;i<T0;i++){
      gr[i] = G(A[i]);
    }
    vector<int>gr_unique = uniqueR(gr);
    mark = 0;
    gr_size = int(gr_unique.size());
    for(int i=0;i<gr_size-1;i++){
      B0.segment(mark, index(gr_unique[i])-index(gr_unique[i]-1)) = Eigen::VectorXi::LinSpaced(index(gr_unique[i])-index(gr_unique[i]-1), index(gr_unique[i]-1)-1, index(gr_unique[i])-2);
      mark = mark+index(gr_unique[i])-index(gr_unique[i]-1);
    }
    if(G(p-1)==gr_unique[gr_size-1]){
      B0.segment(mark, p-index(gr_unique[gr_size-1]-1)+1) = Eigen::VectorXi::LinSpaced(p-index(gr_unique[gr_size-1]-1)+1, index(gr_unique[gr_size-1]-1)-1, p-1);
      mark = mark + p-index(gr_unique[gr_size-1]-1)+1;
    }else{
      B0.segment(mark, index(gr_unique[gr_size-1])-index(gr_unique[gr_size-1]-1)) = Eigen::VectorXi::LinSpaced(index(gr_unique[gr_size-1])-index(gr_unique[gr_size-1]-1), index(gr_unique[gr_size-1]-1)-1, index(gr_unique[gr_size-1])-2);
      mark = mark + index(gr_unique[gr_size-1])-index(gr_unique[gr_size-1]-1);
    }
    Eigen::VectorXi B = B0.head(mark);
    Eigen::MatrixXd X_B(n, B.size());
    for(int i=0;i<B.size();i++){
      X_B.col(i) = X.col(B(i));
    }
    Eigen::VectorXd beta_B=X_B.colPivHouseholderQr().solve(y);
    Eigen::VectorXd d = (X.transpose()*(y-X_B*beta_B));
    beta = Eigen::VectorXd::Zero(X.cols());
    for(int i=0;i<B.size();i++){
      beta(B(i)) = beta_B(i);
      d(B(i)) = 0;
    }
    beta0 = beta;
    d0 = d;
    A0 = A;
  }
  return gr_size;
}
// [[Rcpp::export]]
List gbess_lm(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXi& G, Eigen::VectorXi& index, Eigen::VectorXi orderGi, List& PhiG, List& invPhiG, int T0, int max_steps, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, int n, int p, int N, bool normal = true){
  int l;
  int gr_size;
  int df;
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
  gr_size = gbess_lm_pdas(X, y, G, index, orderGi, PhiG, invPhiG, T0, max_steps, beta0, n, p, N, l, df);
  mse = (y-X*beta0).squaredNorm()/double(n);
  nullmse = y.squaredNorm()/double(n);
  aic = double(n)*log(mse)+2.0*df;
  bic = double(n)*log(mse)+log(double(n))*df;
  gic = double(n)*log(mse)+log(double(p))*log(log(double(n)))*df;
  if(normal){
    beta0 = sqrt(double(n))*beta0.cwiseQuotient(normx);
    coef0 = meany - beta0.dot(meanx);
  }
//  return List::create(Named("beta")=beta0, Named("coef0")=coef0, Named("mse")=mse, Named("nullmse")=nullmse, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic, Named("gr_size")=gr_size);
//
  List mylist;
  mylist.add("beta", beta0);
  mylist.add("coef0", coef0);
  mylist.add("mse", mse);
  mylist.add("nullmse", nullmse);
  mylist.add("aic", aic);
  mylist.add("bic", bic);
  mylist.add("gic", gic);
  mylist.add("gr_size", gr_size);
  return mylist;
}
// [[Rcpp::export]]
List gbess_lms(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXi& G, Eigen::VectorXi& index, Eigen::VectorXi& orderGi , List& PhiG, List& invPhiG, Eigen::VectorXi& T_list, int max_steps, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, int n, int p, int N, bool warm_start = false, bool normal = true){
  int l;
  int m = T_list.size();
  int df;
  double nullmse;
  Eigen::VectorXi gr_sizes(m);
  Eigen::VectorXd mse(m);
  Eigen::VectorXd aic(m);
  Eigen::VectorXd bic(m);
  Eigen::VectorXd gic(m);
  Eigen::VectorXd coef0(m);
  Eigen::VectorXd meanx(p);
  Eigen::VectorXd normx(p);
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
    gr_sizes(i) = gbess_lm_pdas(X, y, G, index, orderGi, PhiG, invPhiG, T_list(i), max_steps, beta, n, p, N, l, df);
    beta_out.col(i) = beta;
    if(!warm_start) beta = beta0;
    mse(i) = (y-X*beta_out.col(i)).squaredNorm()/double(n);
    aic(i) = double(n)*log(mse(i))+2.0*df;
    bic(i) = double(n)*log(mse(i))+log(double(n))*df;
    gic(i) = double(n)*log(mse(i))+log(double(p))*log(log(double(n)))*df;
  }
  nullmse = y.squaredNorm()/double(n);
  // cout<<m<<endl;
  if(normal){
    for(i=0;i<m;i++){
      beta_out.col(i) = sqrt(double(n))*beta_out.col(i).cwiseQuotient(normx);
      coef0(i) = meany - beta_out.col(i).dot(meanx);
    }
  }
//  return List::create(Named("beta")=beta_out, Named("coef0")=coef0, Named("mse")=mse, Named("nullmse")=nullmse, Named("aic")=aic, Named("bic")=bic, Named("gic")=gic, Named("gr_sizes") = gr_sizes);
  List mylist;
  mylist.add("beta", beta_out);
  mylist.add("coef0", coef0);
  mylist.add("mse", mse);
  mylist.add("nullmse", nullmse);
  mylist.add("aic", aic);
  mylist.add("bic", bic);
  mylist.add("gic", gic);
  mylist.add("gr_size", gr_sizes);
  return mylist;
}

