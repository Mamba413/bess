//
// Created by jk on 2020/3/8.
//
//#define R_BUILD
#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
#else
#include <Eigen\Eigen>
#include "List.h"
#endif
#include <algorithm>
#include <vector>

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
//    std::cout<<"normal_1"<<endl;
    for(int i=0;i<p;i++){
        meanx(i) = weights.dot(X.col(i))/double(n);
    }
//    std::cout<<"normal_2"<<endl;
    for(int i=0;i<p;i++){
        X.col(i) = X.col(i).array() - meanx(i);
    }
//    std::cout<<"normal_3"<<endl;
    for(int i=0;i<p;i++){
        tmp = X.col(i);
        tmp = tmp.array().square();
        normx(i) = sqrt(weights.dot(tmp));
    }
//    std::cout<<"normal_4"<<endl;
    for(int i=0;i<p;i++){
        X.col(i) = sqrt(double(n))*X.col(i)/normx(i);
    }
//    std::cout<<"normal_5"<<endl;
}
/*
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

    List mylist;
#ifndef R_BUILD
    mylist.add("X", X);
    mylist.add("meanx", meanx);
    mylist.add("nornx", normx);
#else
    mylist = List::create(Named("X")=X, Named("meanx")=meanx, Named("normx")=normx);
#endif

    return mylist;
}
*/
void Normalize4(Eigen::MatrixXd& X, Eigen::VectorXd& weights, Eigen::VectorXd& normx){
  int n = X.rows();
  int p = X.cols();
  // std::cout<<"n: "<<n<<", p:"<<p<<endl;
  Eigen::VectorXd tmp(n);
    for(int i=0;i<p;i++){
        tmp = X.col(i);
        tmp = tmp.array().square();
        normx(i) = sqrt(weights.dot(tmp));
    }
    for(int i=0;i<p;i++){
        X.col(i) = sqrt(double(n))*X.col(i)/normx(i);
    }
}
