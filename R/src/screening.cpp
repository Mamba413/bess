#define R_BUILD
#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
using namespace Rcpp;
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen\Eigen>
#include "List.h"
#endif

#include <algorithm>
#include <vector>
#include <cmath>
#include "screening.h"
#include "logistic.h"
#include "poisson.h"
#include "coxph.h"

using namespace std;
using namespace Eigen;

vector<int> screening(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, int algorithm_type, int screening_size, Eigen::VectorXi &g_index)
{
    int n = x.rows();
    int p = x.cols();
    int p2 = screening_size;//floor(n / log(n)) > screening_size ? floor(n / log(n)) : screening_size;
    vector<int> screening_A((unsigned int) p2);

    Eigen::VectorXd coef = Eigen::VectorXd::Zero(p);
    if(g_index.size() == p) g_index = g_index.head(p2).eval();

    for(int i=0;i<p;i++)
    {
        Eigen::MatrixXd x_2(n, 1);
        x_2.col(0) = x.col(i);
        Eigen::VectorXd beta;
        if(algorithm_type == 1)
        {
            beta=x_2.colPivHouseholderQr().solve(y);
            coef(i) = beta(0);
        }
        else if(algorithm_type == 2)
        {
            beta=logit_fit(x_2, y, n, 1, weight);
            coef(i) = beta(1);
        }
        else if(algorithm_type == 3)
        {
            beta=poisson_fit(x_2, y, n, 1, weight);
            coef(i) = beta(0);
        }
        else if(algorithm_type == 4)
        {
            beta=cox_fit(x_2, y, n, 1, weight);
            coef(i) = beta(0);
        }
    }
//
//    for(int i=0;i<p;i++)
//    {
//        cout<<coef(i)<<" ";
//    }
//    cout<<endl;

    coef=coef.cwiseAbs();
    for(int k=0;k<p2;k++) {
      coef.maxCoeff(&screening_A[k]);
      coef(screening_A[k])=-1.0;
    }
    sort(screening_A.begin(),screening_A.end());

    Eigen::MatrixXd x_A = Eigen::MatrixXd::Zero(n, p2);
    for(int k=0;k<p2;k++) {
        x_A.col(k) = x.col(screening_A[k]).eval();
        //cout<<"x_A.col("<<k<<"): "<<x_A.col(k)<<", x.col(screening_A[k]): "<<x.col(screening_A[k])<<endl;
    }
    x = x_A;
    //cout<<"xmax: "<<x.maxCoeff()<<", min: "<<x.minCoeff()<<endl;
//    for(int i=0;i<p2;i++)
//    {
//        cout<<screening_A[i]<<" ";
//    }
//    cout<<endl;

    return screening_A;
}
