//
// Created by jiangkangkang on 2020/3/9.
//

#ifndef BESS_UTILITIES_H
#define BESS_UTILITIES_H

#include <iostream>
#include <Eigen/Eigen>
using namespace std;

Eigen::MatrixXd Pointer2MatrixXd(double* x, int x_row, int x_col);
Eigen::MatrixXi Pointer2MatrixXi(int* x, int x_row, int x_col);
Eigen::VectorXd Pointer2VectorXd(double* x, int x_len);
Eigen::VectorXi Pointer2VectorXi(int* x, int x_len);
void MatrixXd2Pointer(Eigen::MatrixXd x_matrix, double* x);
void MatrixXi2Pointer(Eigen::MatrixXi x_matrix, int* x);
void VectorXd2Pointer(Eigen::VectorXd x_vector, double*x);
void VectorXi2Pointer(Eigen::VectorXi x_vector, int* x);

#endif //BESS_UTILITIES_H
