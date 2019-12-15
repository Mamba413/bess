#ifndef BESS_LM_H_
#define BESS_LM_H_

#include <iostream>
#include <Eigen/Eigen>
#include "List.h"
using namespace std;

void bess_lm_pdas(Eigen::MatrixXd& X, Eigen::VectorXd& y, int T0, int max_steps, Eigen::VectorXd& beta, Eigen::VectorXi& A_out, int& l);
// [[Rcpp::export]]
List bess_lm(Eigen::MatrixXd& X, Eigen::VectorXd& y, int T0, int max_steps, Eigen::VectorXd& beta, Eigen::VectorXd& weights, bool normal=true);
// [[Rcpp::export]]
//List bess_lms(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXi& T_list, int max_steps, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, bool warm_start = false, bool normal = true);
//// [[Rcpp::export]]
//List bess_lm_gs(Eigen::MatrixXd& X, Eigen::VectorXd& y, int s_min, int s_max, int K_max, int max_steps, double epsilon, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, bool warm_start = false, bool normal = true);

void pywrap_bess_lm(double* X, int X_row, int X_col, double* y, int y_len, int T0, int max_steps, double* beta, int beta_len, double* weights, int weights_len, double* coef0, double* beta_return, int beta_return_len, double* mse, double* nullmse, double* aic,double* bic, double* gic, int* A, int A_len, bool normal=true);
//void pywrap_bess_lms(double* X, int X_row, int X_col, double* y, int y_len, int* T_list, int T_list_len, int max_steps, double* beta0, int beta0_len, double* weights, int weights_len,
//                     bool warm_start = false, bool normal = true);

#endif
