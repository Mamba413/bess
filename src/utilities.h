//
// Created by jiangkangkang on 2020/3/9.
//

#ifndef BESS_UTILITIES_H
#define BESS_UTILITIES_H

#include <iostream>
#include <Eigen/Eigen>
#include <vector>
using namespace std;

Eigen::MatrixXd Pointer2MatrixXd(double* x, int x_row, int x_col);
Eigen::MatrixXi Pointer2MatrixXi(int* x, int x_row, int x_col);
Eigen::VectorXd Pointer2VectorXd(double* x, int x_len);
Eigen::VectorXi Pointer2VectorXi(int* x, int x_len);
void MatrixXd2Pointer(Eigen::MatrixXd x_matrix, double* x);
void MatrixXi2Pointer(Eigen::MatrixXi x_matrix, int* x);
void VectorXd2Pointer(Eigen::VectorXd x_vector, double*x);
void VectorXi2Pointer(Eigen::VectorXi x_vector, int* x);

Eigen::VectorXi find_ind(Eigen::VectorXi& L, Eigen::VectorXi& index, Eigen::VectorXi& gsize, int p, int N);
Eigen::MatrixXd X_seg(Eigen::MatrixXd& X, int n, Eigen::VectorXi& ind);
std::vector<Eigen::MatrixXd> Phi(Eigen::MatrixXd& X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, double lambda);
std::vector<Eigen::MatrixXd> invPhi(std::vector<Eigen::MatrixXd>& Phi, int N);

// Eigen::VectorXi sort_vec(Eigen::VectorXi& vec);
// void max_k(Eigen::VectorXd& nums, int k, Eigen::VectorXi& result);
// int partition(Eigen::VectorXd& nums, Eigen::VectorXi& ind,  int l, int r);
void max_k(Eigen::VectorXd& vec, int k, Eigen::VectorXi& result);

#endif //BESS_UTILITIES_H
