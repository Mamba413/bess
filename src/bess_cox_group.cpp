#include <Rcpp.h>
#include <RcppEigen.h>
#include <algorithm>
#include <vector>
#include "tmp.h"
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace std;
// [[Rcpp::export]]
List ggetcox_A(Eigen::MatrixXd X, Eigen::VectorXi G, Eigen::VectorXi index, int T0, Eigen::VectorXd beta0, int n, int p, int N, Eigen::VectorXd status, Eigen::VectorXd weights, Eigen::VectorXi B00){
  double max_T=0.0;
  Eigen::VectorXi A_out = Eigen::VectorXi::Zero(T0);
  Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
  Eigen::VectorXi B0 = Eigen::VectorXi::Zero(p);
  Eigen::MatrixXd h(n, n);
  Eigen::VectorXd cum_theta(n);
  Eigen::VectorXd cum_theta2(n);
  Eigen::VectorXd cum_theta3(n);
  vector<int> A(T0);
  vector<int> gr(T0);
  int mark=0;
  Eigen::VectorXd theta = X*beta0;
  for(int i=0;i<=n-1;i++) {
    if(theta(i)<-25.0) theta(i) = -25.0;
    if(theta(i)>25.0) theta(i) = 25.0;
  }
  theta = weights.array()*theta.array().exp();
  cum_theta(n-1) = theta(n-1);
  for(int k=n-2;k>=0;k--) {
    cum_theta(k) = cum_theta(k+1)+theta(k);
  }
  cum_theta2(0) = (status(0)*weights(0))/cum_theta(0);
  for(int k=1;k<=n-1;k++) {
    cum_theta2(k) = (status(k)*weights(k))/cum_theta(k)+cum_theta2(k-1);
  }
  cum_theta3(0) = (status(0)*weights(0))/pow(cum_theta(0),2);
  for(int k=1;k<=n-1;k++) {
    cum_theta3(k) = (status(k)*weights(k))/pow(cum_theta(k),2)+cum_theta3(k-1);
  }
  h = -cum_theta3.replicate(1, n);
  h = h.cwiseProduct(theta.replicate(1, n));
  h = h.cwiseProduct(theta.replicate(1, n).adjoint());
  for(int i=0;i<n;i++)
    for(int j=i+1;j<n;j++)
      h(j, i) = h(i, j);
  Eigen::VectorXd g = cum_theta2.cwiseProduct(theta) - weights.cwiseProduct(status);
  h.diagonal() = cum_theta2.cwiseProduct(theta) + h.diagonal();
  Eigen::VectorXd d0 = X.adjoint()*g;
  if(B00.size()<p){
    for(int i=0;i<B00.size();i++){
      d0(B00(i)-1)=0.0;
    }
  }
  for(int i=0;i<N-1;i++){
    Eigen::MatrixXd XG =X.middleCols(index(i)-1, index(i+1)-index(i));
    Eigen::MatrixXd XGbar = XG.adjoint()*h*XG;
    //cout<<XGbar<<" ";
    Eigen::MatrixXd phiG = -EigenR(XGbar);
    Eigen::MatrixXd invphiG = phiG.colPivHouseholderQr().solve(Eigen::MatrixXd::Identity(index(i+1)-index(i),index(i+1)-index(i)));
    betabar.segment(index(i)-1,index(i+1)-index(i)) = phiG*beta0.segment(index(i)-1,index(i+1)-index(i));
    dbar.segment(index(i)-1,index(i+1)-index(i)) = invphiG*d0.segment(index(i)-1,index(i+1)-index(i));
    //cout<<XGbar<<endl;
  }
  Eigen::MatrixXd XG =X.middleCols(index(N-1)-1,p-index(N-1)+1);
  Eigen::MatrixXd XGbar = XG.adjoint()*h*XG;
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
//  return List::create(Named("A")=A_out,Named("B")=B,Named("max_T")=max_T,Named("gr_size")=gr_size);

  List mylist;
  mylist.add("A", A_out);
  mylist.add("B", B);
  mylist.add("max_T", max_T);
  mylist.add("gr_size", gr_size);
  return mylist;
}
