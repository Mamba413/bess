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

double loglik_cox(Eigen::MatrixXd X, Eigen::VectorXd status, Eigen::VectorXd beta, Eigen::VectorXd weights)
{
  int n = X.rows();
  Eigen::VectorXd eta = X*beta;
  for (int i=0;i<n;i++)
  {
    if (eta(i) > 30)
    {
      eta(i) = 30;
    }
    else if (eta(i) < -30)
    {
      eta(i) = -30;
    }
  }
  Eigen::VectorXd expeta = eta.array().exp();
  Eigen::VectorXd cum_expeta(n);
  cum_expeta(n-1) = expeta(n-1);
  for (int i=n-2;i>=0;i--)
  {
    cum_expeta(i) = cum_expeta(i+1)+expeta(i);
  }
  Eigen::VectorXd ratio = (expeta.cwiseQuotient(cum_expeta)).array().log();
  return (ratio.dot((weights.cwiseProduct(status))));
}


Eigen::VectorXd cox_fit(Eigen::MatrixXd X, Eigen::VectorXd status, int n, int p, Eigen::VectorXd weights)
{
//  cout<<"cox_fit"<<endl;
  Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd eta(n);
  Eigen::VectorXd expeta(n);
  Eigen::VectorXd cum_expeta(n);
  Eigen::MatrixXd x_theta(n, p);
  Eigen::VectorXd xij_theta(n);
  Eigen::VectorXd g(p);
  Eigen::MatrixXd h(p, p);
  Eigen::VectorXd d(p);
  double loglik0;
  double loglik1;

  double step;
  int m;
  int l;
  for (l=1;l<=50;l++)
  {
    step = 0.5;
    m = 1;
    eta = X*beta0;
    for (int i=0;i<n;i++)
    {
      if (eta(i) > 30)
      {
        eta(i) = 30;
      }
      else if (eta(i) < -30)
      {
        eta(i) = -30;
      }
    }
    expeta = eta.array().exp();
    cum_expeta(n-1) = expeta(n-1);
    for (int i=n-2;i>=0;i--)
    {
      cum_expeta(i) = cum_expeta(i+1)+expeta(i);
    }
    for (int i=0;i<p;i++)
    {
      x_theta.col(i) = X.col(i).cwiseProduct(expeta);
    }
    for (int i=n-2;i>=0;i--)
    {
      x_theta.row(i) = x_theta.row(i)+x_theta.row(i+1);
    }
    for (int i=0;i<p;i++)
    {
      x_theta.col(i) = x_theta.col(i).cwiseQuotient(cum_expeta);
    }
    g = (X-x_theta).transpose()*(weights.cwiseProduct(status));
    for (int k1=0;k1<p;k1++)
    {
      for (int k2=k1;k2<p;k2++)
      {
        xij_theta = (expeta.cwiseProduct(X.col(k1))).cwiseProduct(X.col(k2));
        for(int j=n-2;j>=0;j--)
        {
          xij_theta(j) = xij_theta(j+1) + xij_theta(j);
        }
        h(k1, k2) = -(xij_theta.cwiseQuotient(cum_expeta) - x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(status));
        h(k2, k1) = h(k1, k2);
      }
    }
    d = h.ldlt().solve(g);
    beta1 = beta0-pow(step, m)*d;
    loglik0 = loglik_cox(X, status, beta0, weights);
    loglik1 = loglik_cox(X, status, beta1, weights);
    while ((loglik0 >= loglik1) && (m<10))
    {
      m = m+1;
      beta1 = beta0-pow(step, m)*d;
      loglik1 = loglik_cox(X, status, beta1, weights);
    }
    //cout<<"m: "<<m<<endl;
    beta0 = beta1;
    if (abs(loglik0-loglik1)/abs(loglik0) < 1e-8)
    {
      break;
    }
  }
//  cout<<"cox_fit end"<<endl;
  return beta0;
}

void getcox_A(Eigen::MatrixXd X, Eigen::VectorXd beta, int T0, Eigen::VectorXi B, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXi &A_out, Eigen::VectorXi &I_out)
{
//  cout<<"getcox_A"<<endl;
//  double max_T=0.0;
  int n=X.rows();
  int p=X.cols();
//  Eigen::VectorXi A_out = Eigen::VectorXi::Zero(T0);
//  Eigen::VectorXi I_out = Eigen::VectorXi::Zero(p-T0);
//  vector<double> status;
//  for(int i=0;i<y.size();i++)
//  {
//    if(y[i]==0.0) status.push_back(i);
//  }

//  A_out = Eigen::VectorXi::Zero(T0);
//  I_out = Eigen::VectorXi::Zero(p-T0);
  for(int i=0;i<T0;i++)
	  A_out(i) = 0;
  for(int i=0;i<p-T0;i++)
	  I_out(i) = 0;
  Eigen::VectorXd l1 = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd l2 = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd cum_theta=Eigen::VectorXd::Zero(n);
  Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd xtheta(n,p);
  Eigen::MatrixXd x2theta(n,p);
  vector<int>E(p);
  for(int k=0;k<=p-1;k++) {
    E[k]=k;
  }
  vector<int>A(T0);
  vector<int>I(p-T0);
  Eigen::VectorXd theta=X*beta;
    for(int i=0;i<=n-1;i++) {
    if(theta(i)>25.0) theta(i) = 25.0;
    if(theta(i)<-25.0) theta(i) = -25.0;
  }
  theta=weights.array()*theta.array().exp();
  cum_theta(n-1)=theta(n-1);
  for(int k=n-2;k>=0;k--) {
    cum_theta(k)=cum_theta(k+1)+theta(k);
  }
  for(int k=0;k<=p-1;k++) {
    xtheta.col(k)=theta.cwiseProduct(X.col(k));
  }
  for(int k=0;k<=p-1;k++) {
    x2theta.col(k)=X.col(k).cwiseProduct(xtheta.col(k));
  }
  for(int k=n-2;k>=0;k--) {
    xtheta.row(k)=xtheta.row(k+1)+xtheta.row(k);
  }
  for(int k=n-2;k>=0;k--) {
    x2theta.row(k)=x2theta.row(k+1)+x2theta.row(k);
  }
  for(int k=0;k<=p-1;k++) {
    xtheta.col(k)=xtheta.col(k).cwiseQuotient(cum_theta);
  }
  for(int k=0;k<=p-1;k++) {
    x2theta.col(k)=x2theta.col(k).cwiseQuotient(cum_theta);
  }
  x2theta=x2theta.array()-xtheta.array().square().array();
  xtheta=X.array()-xtheta.array();
  for(unsigned int k=0;k<y.size();k++) {
    if(y[k] == 0.0)
    {
      xtheta.row(k)=Eigen::VectorXd::Zero(p);
      x2theta.row(k)=Eigen::VectorXd::Zero(p);
    }
  }
  l1=-xtheta.adjoint()*weights;
  l2=x2theta.adjoint()*weights;
  d=-l1.cwiseQuotient(l2);
  if(B.size()<p) {
    for(int k=0;k<=B.size()-1;k++) {
      d(B(k)-1)=0.0;
    }
  }
  bd=beta+d;
  bd=bd.cwiseAbs();
  bd=bd.cwiseProduct(l2.cwiseSqrt());
  for(int k=0;k<=T0-1;k++) {
    bd.maxCoeff(&A[k]);
    bd(A[k])=0.0;
  }
  sort (A.begin(),A.end());
  set_difference(E.begin(),E.end(), A.begin(),A.end(),I.begin());
  for(int i=0;i<T0;i++)
	  A_out(i) = A[i];
  for(int i=0;i<p-T0;i++)
	  I_out(i) = I[i];
//  return A_out;
//  cout<<"getcox_A end"<<endl;
}
