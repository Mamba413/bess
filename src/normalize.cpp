//#include <Rcpp.h>
//#include <RcppEigen.h>
#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
#else
#include <Eigen\Eigen>
#include "List.h"
#endif
#include <algorithm>
#include <vector>
// [[Rcpp::depends(RcppEigen)]]
//using namespace Rcpp;
using namespace std;
void Normalize(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXd& weights, Eigen::VectorXd& meanx, double& meany, Eigen::VectorXd& normx){
  int n = X.rows();
  int p = X.cols();
  Eigen::VectorXd tmp(n);
  for(int i=0;i<p;i++){
    meanx(i) = weights.dot(X.col(i))/double(n);
  }
  meany = (y.dot(weights))/double(n);
  for(int i=0;i<p;i++){
    X.col(i) = X.col(i).array() - meanx(i);
  }
  y = y.array() - meany;
  for(int i=0;i<p;i++){
    tmp = X.col(i);
    tmp = tmp.array().square();
    normx(i) = sqrt(weights.dot(tmp));
  }
  for(int i=0;i<p;i++){
    X.col(i) = sqrt(double(n))*X.col(i)/normx(i);
  }
}
void Normalize3(Eigen::MatrixXd& X, Eigen::VectorXd& weights, Eigen::VectorXd& meanx, Eigen::VectorXd& normx){
  int n = X.rows();
  int p = X.cols();
  Eigen::VectorXd tmp(n);
  for(int i=0;i<p;i++){
    meanx(i) = weights.dot(X.col(i))/double(n);
  }
  for(int i=0;i<p;i++){
    X.col(i) = X.col(i).array() - meanx(i);
  }
  for(int i=0;i<p;i++){
    tmp = X.col(i);
    tmp = tmp.array().square();
    normx(i) = sqrt(weights.dot(tmp));
  }
  for(int i=0;i<p;i++){
    X.col(i) = sqrt(double(n))*X.col(i)/normx(i);
  }
}
// [[Rcpp::export]]
List Normalize2(Eigen::MatrixXd& X, Eigen::VectorXd& weights){
  int n = X.rows();
  int p = X.cols();
  Eigen::VectorXd meanx(p);
  Eigen::VectorXd normx(p);
  Eigen::VectorXd tmp(n);
  for(int i=0;i<p;i++){
    meanx(i) = weights.dot(X.col(i))/double(n);
  }
  for(int i=0;i<p;i++){
    X.col(i) = X.col(i).array() - meanx(i);
  }
  for(int i=0;i<p;i++){
    tmp = X.col(i);
    tmp = tmp.array().square();
    normx(i) = sqrt(weights.dot(tmp));
  }
  for(int i=0;i<p;i++){
    X.col(i) = sqrt(double(n))*X.col(i)/normx(i);
  }
//  return List::create(Named("X")=X, Named("meanx")=meanx, Named("normx")=normx);
//
  List mylist;
  mylist.add("X", X);
  mylist.add("meanx", meanx);
  mylist.add("nornx", normx);
  return mylist;
}
