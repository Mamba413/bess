//
// Created by jiangkangkang on 2020/3/9.
//

#include "utilities.h"

#include <iostream>
#include <Eigen/Eigen>

using namespace std;

Eigen::MatrixXd Pointer2MatrixXd(double* x, int x_row, int x_col)
{
    Eigen::MatrixXd x_matrix(x_row, x_col);
    int i, j;
    for(i=0;i<x_row;i++)
    {
        for(j=0;j<x_col;j++)
        {
            x_matrix(i, j) = x[i * x_col + j];
        }
    }
    return x_matrix;
}

Eigen::MatrixXi Pointer2MatrixXi(int* x, int x_row, int x_col)
{
    Eigen::MatrixXi x_matrix(x_row, x_col);
    int i, j;
    for(i=0;i<x_row;i++)
    {
        for(j=0;j<x_col;j++)
        {
            x_matrix(i, j) = x[i * x_col + j];
        }
    }
    return x_matrix;
}

Eigen::VectorXd Pointer2VectorXd(double* x, int x_len)
{
    Eigen::VectorXd x_vector(x_len);
    int i;
    for(i=0;i<x_len;i++)
    {
        x_vector[i] = x[i];
    }
    return x_vector;
}

Eigen::VectorXi Pointer2VectorXi(int* x, int x_len)
{
    Eigen::VectorXi x_vector(x_len);
    int i;
    for(i=0;i<x_len;i++)
    {
        x_vector[i] = x[i];
    }
    return x_vector;
}

void MatrixXd2Pointer(Eigen::MatrixXd x_matrix, double* x)
{
    int x_matrix_row, x_matrix_col, i, j;
    x_matrix_row = x_matrix.rows();
    x_matrix_col = x_matrix.cols();
    for(i=0;i<x_matrix_row;i++)
    {
        for(j=0;j<x_matrix_col;j++)
        {
            x[i * x_matrix_col + j] = x_matrix(i,j);
        }
    }
}

void MatrixXi2Pointer(Eigen::MatrixXi x_matrix, int *x)
{
    int x_matrix_row, x_matrix_col, i, j;
    x_matrix_row = x_matrix.rows();
    x_matrix_col = x_matrix.cols();
    for(i=0;i<x_matrix_row;i++)
    {
        for(j=0;j<x_matrix_col;j++)
        {
            x[i * x_matrix_col + j] = x_matrix(i,j);
        }
    }
}

void VectorXd2Pointer(Eigen::VectorXd x_vector, double* x)
{
    int x_matrix_len, i;
    x_matrix_len = x_vector.size();

    for(i=0;i<x_matrix_len;i++)
    {
        x[i] = x_vector[i];
    }
}

void VectorXi2Pointer(Eigen::VectorXi x_vector, int *x)
{
    int x_matrix_len, i;
    x_matrix_len = x_vector.size();

    for(i=0;i<x_matrix_len;i++)
    {
        x[i] = x_vector[i];
    }
}

