//
// Created by Mamba on 2020/2/18.
//
// #define R_BUILD
#ifndef SRC_DATA_H
#define SRC_DATA_H

#ifdef R_BUILD
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#else
#include <Eigen/Eigen>
#endif
#include <iostream>

#include <vector>
#include "normalize.h"
using namespace std;
using namespace Eigen;


class Data {

public:
    Eigen::MatrixXd x;
    Eigen::VectorXd y;
    Eigen::VectorXd weight;
    Eigen::VectorXd x_mean;
    Eigen::VectorXd x_norm;
    double y_mean;
    int n;
    int p;
    int data_type;
    bool is_normal;
    int g_num;
    Eigen::VectorXi g_index;
    Eigen::VectorXi g_size;

    Data() = default;

    Data(Eigen::MatrixXd& x, Eigen::VectorXd& y, int data_type, Eigen::VectorXd weight, bool is_normal, Eigen::VectorXi g_index) {
        this->x = x;
        //cout<<"xmax: "<<x.maxCoeff()<<", min: "<<x.minCoeff()<<endl;
        this->y = y;
        this->data_type = data_type;
        this->n = x.rows();
        this->p = x.cols();

        this->weight = weight;
        this->is_normal = is_normal;
        this->x_mean = Eigen::VectorXd::Zero(this->p);
        this->x_norm = Eigen::VectorXd::Zero(this->p);

        if(is_normal){
            this->normalize();
        }

        // Eigen::VectorXd tmp(n);
        // for(int i=0;i<p;i++){
        //     tmp = this->x.col(i);
        //     tmp = tmp.array().square();
        //     cout<<sqrt(weight.dot(tmp))<<endl;
        // }
        //cout<<"g_index: "<< g_index<<endl;
        this->g_index = g_index;
        //cout<<"g_index: "<<this->g_index<<endl;
        this->g_num = (g_index).size(); 
        //cout<<"g_num: "<<this->g_num<<endl;
        Eigen::VectorXi temp = Eigen::VectorXi::Zero(g_num);
        temp.head(g_num-1) = g_index.tail(g_num-1);
        temp(g_num-1) = this->p;
        this->g_size =  temp-g_index;
    };

    void add_weight() {
        for(int i=0;i<this->n;i++){
            this->x.row(i) = this->x.row(i)*sqrt(this->weight(i));
            this->y(i) = this->y(i)*sqrt(this->weight(i));
        }
    };

    void normalize() {
//        std::cout<<"normalize"<<endl;
        if(this->data_type == 1){
            Normalize(this->x, this->y, this->weight, this->x_mean, this->y_mean, this->x_norm);
        }
        else if(this->data_type == 2){
//            std::cout<<"normalize"<<endl;
            Normalize3(this->x, this->weight, this->x_mean, this->x_norm);
        }
        else{
            // std::cout<<"norm_4"<<endl;
            Normalize4(this->x, this->weight, this->x_norm);
            //cout<<"xmax: "<<x.maxCoeff()<<", min: "<<x.minCoeff()<<endl;
            //cout<<"x: "<<this->x<<endl;
        }
//        std::cout<<"normalize end"<<endl;
        // reload this method for different data type
    };

  Eigen::VectorXi get_g_index() {
       return this->g_index;
   };

    int get_g_num() {
       return this->g_num;
    };

    Eigen::VectorXi get_g_size() {
       return this->g_size;
    };

    int get_n() {
        return this->n;
    };

    int get_p() {
        return this->p;
    };

    double get_nullloss() {
        if(this->data_type ==1){
            return this->y.squaredNorm()/double(this->n);
        } else{
            return -2*log(0.5)*this->weight.sum();
        }
    };
};
#endif //SRC_DATA_H
