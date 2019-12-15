#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
#else
#include <Eigen\Eigen>
#include "List.h"
#endif
#include <algorithm>
#include <vector>
#include "tmp.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace std;
// [[Rcpp::export]]
List gget_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXi G, Eigen::VectorXi index, int T0, Eigen::VectorXd beta0, double coef0, int n, int p, int N, Eigen::VectorXd weights, Eigen::VectorXi B00){
  double max_T=0.0;
  Eigen::VectorXi A_out = Eigen::VectorXi::Zero(T0);
  Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
  Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
  Eigen::VectorXi B0 = Eigen::VectorXi::Zero(p);
  vector<int> A(T0);
  vector<int> gr(T0);
  Eigen::VectorXd coef(n);
  for(int i=0;i<=n-1;i++) {
    coef(i)=coef0;
  }
  int mark=0;
  Eigen::VectorXd eta = X*beta0+coef;
  for(int i=0;i<=n-1;i++) {
    if(eta(i)<-25.0) eta(i) = -25.0;
    if(eta(i)>25.0) eta(i) = 25.0;
  }
  Eigen::VectorXd exppr = eta.array().exp();
  Eigen::VectorXd pr = exppr.array()/(exppr+one).array();
  Eigen::VectorXd g = weights.array()*(y-pr).array();
  Eigen::VectorXd h = weights.array()*pr.array()*(one-pr).array();
  Eigen::VectorXd d0 = X.adjoint()*g;
  if(B00.size()<p){
    for(int i=0;i<B00.size();i++){
      d0(B00(i)-1)=0.0;
    }
  }
  for(int i=0;i<N-1;i++){
    Eigen::MatrixXd XG =X.middleCols(index(i)-1, index(i+1)-index(i));
    Eigen::MatrixXd XGbar = XG.adjoint()*h.asDiagonal()*XG;
    Eigen::MatrixXd phiG = -EigenR(XGbar);
    Eigen::MatrixXd invphiG = phiG.colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(index(i+1)-index(i),index(i+1)-index(i)));
    betabar.segment(index(i)-1,index(i+1)-index(i)) = phiG*beta0.segment(index(i)-1,index(i+1)-index(i));
    dbar.segment(index(i)-1,index(i+1)-index(i)) = invphiG*d0.segment(index(i)-1,index(i+1)-index(i));
  }
  Eigen::MatrixXd XG =X.middleCols(index(N-1)-1,p-index(N-1)+1);
  Eigen::MatrixXd XGbar = XG.adjoint()*h.asDiagonal()*XG;
  Eigen::MatrixXd phiG = -EigenR(XGbar);
  Eigen::MatrixXd invphiG = phiG.colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(p-index(N-1)+1,p-index(N-1)+1));
  betabar.segment(index(N-1)-1,p-index(N-1)+1) = phiG*beta0.segment(index(N-1)-1,p-index(N-1)+1);
  dbar.segment(index(N-1)-1,p-index(N-1)+1) = invphiG*d0.segment(index(N-1)-1,p-index(N-1)+1);
  bd = (betabar+dbar).array().abs();
  for(int k=0;k<T0;k++) {
    max_T=bd.maxCoeff(&A[k]);
    //cout<<bd(A[k])<<" ";
    bd(A[k])=0.0;
  }
  sort(A.begin(),A.end());
  for(int i=0;i<T0;i++){
    gr[i] = G(A[i]);
  }
  vector<int>gr_unique = uniqueR(gr);
  mark = 0;
  int gr_size = int(gr_unique.size());
  for(int i=0;i<gr_size-1;i++){
    B0.segment(mark, index(gr_unique[i])-index(gr_unique[i]-1)) = Eigen::VectorXi::LinSpaced(index(gr_unique[i])-index(gr_unique[i]-1), index(gr_unique[i]-1)-1, index(gr_unique[i])-2);
    mark = mark+index(gr_unique[i])-index(gr_unique[i]-1);
  }
  if(G(p-1)==gr_unique[gr_size-1]){
    B0.segment(mark, p-index(gr_unique[gr_size-1]-1)+1) = Eigen::VectorXi::LinSpaced(p-index(gr_unique[gr_size-1]-1)+1, index(gr_unique[gr_size-1]-1)-1, p-1);
    mark = mark+p-index(gr_unique[gr_size-1]-1)+1;
  }else{
    B0.segment(mark, index(gr_unique[gr_size-1])-index(gr_unique[gr_size-1]-1)) = Eigen::VectorXi::LinSpaced(index(gr_unique[gr_size-1])-index(gr_unique[gr_size-1]-1), index(gr_unique[gr_size-1]-1)-1, index(gr_unique[gr_size-1])-2);
    mark = mark+index(gr_unique[gr_size-1])-index(gr_unique[gr_size-1]-1);
  }
  Eigen::VectorXi B = B0.head(mark);
  for(int i=0;i<T0;i++)
    A_out(i) = A[i];
//  return List::create(Named("p")=pr,Named("A")=A_out,Named("B")=B,Named("max_T")=max_T,Named("gr_size")=gr_size);
  List mylist;
  mylist.add("p", pr);
  mylist.add("A", A_out);
  mylist.add("B", B);
  mylist.add("max_T", max_T);
  mylist.add("gr_size", gr_size);
  return mylist;
}
