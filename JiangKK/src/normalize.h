#ifndef normlize_H
#define normlize_H
//#include <RcppEigen.h>
#ifdef R_BUILD
#include <RcppEigen.h>
#else
#include <Eigen\Eigen>
#endif

void Normalize(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXd& weights, Eigen::VectorXd& meanx, double& meany, Eigen::VectorXd& normx);
void Normalize3(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXd& meanx, Eigen::VectorXd& normx);

#endif
