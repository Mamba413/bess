#ifndef screening_H
#define screening_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen\Eigen>
#endif

#include <vector>

using namespace std;
using namespace Eigen;

vector<int> screening(Eigen::MatrixXd &x, Eigen::VectorXd &y, Eigen::VectorXd &weight, int algorithm_type, int sequence_max);

#endif