// #define R_BUILD
#ifdef R_BUILD
#include <Rcpp.h>
#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace std;
#else

#include <Eigen\Eigen>
#include "List.h"

#endif

#include <iostream>
#include "Data.h"
#include "Algorithm.h"
#include "Metric.h"
#include "path.h"
#include "utilities.h"
//#include "bess.h"
#include "screening.h"
#include <vector>

#ifdef OTHER_ALGORITHM1
#include "SpliceAlgorithm.h"
#endif

#ifdef OTHER_ALGORITHM2
#include "PrincipalBallAlgorithm.h"
#endif

using namespace Eigen;
using namespace std;

// [[Rcpp::export]]
List bessCpp(Eigen::MatrixXd x, Eigen::VectorXd y, int data_type, Eigen::VectorXd weight,
             bool is_normal,
             int algorithm_type, int model_type, int max_iter, int exchange_num,
             int path_type, bool is_warm_start,
             int ic_type, bool is_cv, int K,
             Eigen::VectorXd state,
             Eigen::VectorXi sequence,
             Eigen::VectorXd lambda_seq,
             int s_min, int s_max, int K_max, double epsilon,
             double lambda_min, double lambda_max,
             bool is_screening, int powell_path,
             Eigen::VectorXi g_index) {

    //#ifndef R_BUILD
        srand(123);
    //#endif
    int p = x.cols();
    int sequence_max = sequence[sequence.size() - 1];
    vector<int> screening_A;
    if (is_screening) {
        screening_A = screening(x, y, weight, model_type, sequence_max);
    }
    //cout<<"fini"<<endl;
    Data data(x, y, data_type, weight, is_normal, g_index);
    // cout<<"data.x;"<<data.x<<endl;  


    Algorithm *algorithm;
    if (algorithm_type == 1) {
        if (model_type == 1) {
            // std::cout<<"model type "<<model_type<<endl;
            data.add_weight();
            algorithm = new PdasLm(data, max_iter);
            // cout<<"algorithm->data.x;"<<algorithm->data.x<<endl;
        } else if (model_type == 2) {
            // std::cout<<"model type "<<model_type<<endl;
            algorithm = new PdasLogistic(data, max_iter);
        } else if (model_type == 3) {
            // std::cout<<"model type "<<model_type<<endl;
            algorithm = new PdasPoisson(data, max_iter);
        } else {
            // std::cout<<"model type "<<model_type<<endl;
            algorithm = new PdasCox(data, max_iter);
        }
    }
       else if (algorithm_type == 2) {
        if (model_type == 1) {
            data.add_weight();
            algorithm = new GroupPdasLm(data, max_iter);
            algorithm->PhiG = Phi(x, g_index, data.get_g_size(), data.get_n(), data.get_p(), data.get_g_num());
            algorithm->invPhiG = invPhi(algorithm->PhiG, data.get_g_num());
        } else if (model_type == 2) {
            algorithm = new GroupPdasLogistic(data, max_iter);
        } else if (model_type == 3) {
            algorithm = new GroupPdasPoisson(data, max_iter);
        } else {
            algorithm = new GroupPdasCox(data, max_iter);
        }
    }
    else if (algorithm_type == 5) {
        if (model_type == 1) {
            data.add_weight();
            algorithm = new L0L2Lm(data, max_iter);
        }
        else if (model_type == 2) {
            algorithm = new L0L2Logistic(data, max_iter);
        } 
        else if (model_type == 3) {
            algorithm = new L0L2Poisson(data, max_iter);
        } else {
            algorithm = new L0L2Cox(data, max_iter);
        }
    }



    #ifdef OTHER_ALGORITHM1
        if (algorithm_type == 3) {
            if (model_type == 1) {
                data.add_weight();
                algorithm = new SpliceLm(data, max_iter);
            }
            algorithm->update_exchange_num(exchange_num);
        }
    #endif
    #ifdef OTHER_ALGORITHM2
        if (algorithm_type == 4) {
            if (model_type == 1) {
                data.add_weight();
                algorithm = new PrincipalBallLm(data, max_iter);
            }
        }
    #endif
    algorithm->set_warm_start(is_warm_start);

    Metric *metric;
    if (model_type == 1) {
        metric = new LmMetric(ic_type, is_cv, K);
    } else if (model_type == 2) {
        metric = new LogisticMetric(ic_type, is_cv, K);
    } else if (model_type == 3) {
        metric = new PoissonMetric(ic_type, is_cv, K);
    } else {
        metric = new CoxMetric(ic_type, is_cv, K);
    }
    if (is_cv) {
        metric->set_cv_train_test_mask(data.get_n());
        metric->set_cv_initial_model_param(K, data.get_p());
    }

    List result;
    if (path_type == 1) {
        result = sequential_path(data, algorithm, metric, sequence, lambda_seq);
    }
    else
    {
        if(algorithm_type == 5)
        {
            double log_lambda_min = log(max(lambda_min, 1e-5));
            double log_lambda_max = log(max(lambda_max, 1e-5));
            result = pgs_path(data, algorithm, metric, s_min, s_max, log_lambda_min, log_lambda_max, powell_path);
        }
        else
        {
            cout<<"gs_path"<<endl;
            result = gs_path(data, algorithm, metric, s_min, s_max, K_max, epsilon);
        }
            
    }

    // cout<<"1"<<endl;
    if (is_screening) {
        // cout<<"1"<<endl;
        Eigen::VectorXd beta_screening_A;
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    #ifndef R_BUILD
        result.get_value_by_name("beta", beta_screening_A);
        for (unsigned int i = 0; i < screening_A.size(); i++) {
            beta(screening_A[i]) = beta_screening_A(i);
        }
        result.add("beta", beta);
    #else
        beta_screening_A = result["beta"];
        for(int i=0;i<screening_A.size();i++) {
            beta(screening_A[i]) = beta_screening_A(i);
        }
        result["beta"] = beta;
    #endif
    }
    return result;
}

#ifndef R_BUILD

void pywrap_bess(double *x, int x_row, int x_col, double *y, int y_len, int data_type, double *weight, int weight_len,
                 bool is_normal,
                 int algorithm_type, int model_type, int max_iter, int exchange_num,
                 int path_type, bool is_warm_start,
                 int ic_type, bool is_cv, int K,
                 int *gindex, int gindex_len,
                 double *state, int state_len,
                 int *sequence, int sequence_len,
                 double *lambda_sequence, int lambda_sequence_len,
                 int s_min, int s_max, int K_max, double epsilon,
                 double lambda_min, double lambda_max,
                 bool is_screening, int powell_path,
                 double *beta_out, int beta_out_len, double *coef0_out, int coef0_out_len, double *train_loss_out,
                 int train_loss_out_len, double *ic_out, int ic_out_len, double *nullloss_out, double *aic_out,
                 int aic_out_len, double *bic_out, int bic_out_len, double *gic_out, int gic_out_len, int *A_out,
                 int A_out_len, int *l_out)
                 {
    Eigen::MatrixXd x_Mat;
    Eigen::VectorXd y_Vec;
    Eigen::VectorXd weight_Vec;
    Eigen::VectorXi gindex_Vec;
    Eigen::VectorXd state_Vec;
    Eigen::VectorXi sequence_Vec;
    Eigen::VectorXd lambda_sequence_Vec;

    x_Mat = Pointer2MatrixXd(x, x_row, x_col);
    y_Vec = Pointer2VectorXd(y, y_len);
    weight_Vec = Pointer2VectorXd(weight, weight_len);
    state_Vec = Pointer2VectorXd(state, state_len);
    gindex_Vec = Pointer2VectorXi(gindex, gindex_len);
    sequence_Vec = Pointer2VectorXi(sequence, sequence_len);
    lambda_sequence_Vec = Pointer2VectorXd(lambda_sequence, lambda_sequence_len);

//    std::cout<<"pywrap"<<endl;
    List mylist = bessCpp(x_Mat, y_Vec, data_type, weight_Vec, is_normal,
                          algorithm_type, model_type, max_iter, exchange_num,
                          path_type, is_warm_start, ic_type, is_cv, K, state_Vec, sequence_Vec, lambda_sequence_Vec,
                          s_min, s_max, K_max, epsilon,
                          lambda_min, lambda_max, is_screening, powell_path, gindex_Vec);
//    std::cout<<"pywrap_2"<<endl;

    // all sequence output
//    Eigen::VectorXd aic_sequence;
//    Eigen::VectorXd bic_sequence;
//    Eigen::VectorXd gic_sequence;
//
//    Eigen::MatrixXd beta_matrix;
//    Eigen::VectorXd coef0_sequence;
//    Eigen::VectorXd ic_sequence;
//    Eigen::VectorXi T_sequence;
//    double nullloss;
//    Eigen::VectorXi A;
//    int l;
//
//    mylist.get_value_by_name("beta", beta_matrix);
//    mylist.get_value_by_name("coef0", coef0_sequence);
//    mylist.get_value_by_name("ic", ic_sequence);
//    mylist.get_value_by_name("nullloss", nullloss);
//    mylist.get_value_by_name("aic", aic_sequence);
//    mylist.get_value_by_name("bic", bic_sequence);
//    mylist.get_value_by_name("gic", gic_sequence);
//    mylist.get_value_by_name("A", A);
//    mylist.get_value_by_name("l", l);
//
//    MatrixXd2Pointer(beta_matrix, beta_out);
//
//    VectorXd2Pointer(coef0_sequence, coef0_out);
//    VectorXd2Pointer(ic_sequence, ic_out);
//    *nullloss_out = nullloss;
//    VectorXd2Pointer(aic_sequence, aic_out);
//    VectorXd2Pointer(bic_sequence, bic_out);
//    VectorXd2Pointer(gic_sequence, gic_out);
//    VectorXi2Pointer(A, A_out);
//    *l_out = l;
//    std::cout<<"pywrap_end"<<endl;


    Eigen::VectorXd beta;
    double coef0;
    double train_loss;
    double ic;
    mylist.get_value_by_name("beta", beta);
    mylist.get_value_by_name("coef0", coef0);
    mylist.get_value_by_name("train_loss", train_loss);
    mylist.get_value_by_name("ic", ic);

    VectorXd2Pointer(beta, beta_out);
    *coef0_out = coef0;
    *train_loss_out = train_loss;
    *ic_out = ic;
}

#endif
//
//int main() {
//    Eigen::MatrixXd x(3,2);
//    x<<1.0,2.0,3.0,4.5,1.2,1.6;
//    Eigen::VectorXd y(3);
//    y(0) = 1;
//    y(1) = 2;
//    y(2) = 2;
//    Eigen::VectorXd state;
//    Eigen::VectorXd weight(3);
//    weight(0) = 1;
//    weight(1) = 1;
//    weight(2) = 1;
//    Eigen::VectorXi sequence(2);
//    sequence(0) = 1;
//    sequence(1) = 2;
//    bool is_weight;
//    bool is_normal;
//    bool is_warm_start;
//    int data_type;
//    unsigned int algorithm_type;
//    unsigned int max_iter=10;
//    unsigned int c_max;
//    unsigned int path_type;
//    unsigned int sparsity_level;
//
//    int s_min = 1;
//    int s_max = 10;
//    int K_max = 20;
//    double epsilon = 0.00001;
//
//    // algorithm_type: 1:PDAS 2:GROUP 3.SDAR
//    algorithm_type = 1;
//    // data_type：1:REGRESSION 2:CLASSIFICATION
//    // maybe change a name
//    data_type = 1;
//    // path_type: 1:sequential_path 2:gs_path
//    path_type = 0;
//
//    double coef0 = 0;
//
//    Data data(x, y, data_type);
////    if (data_type == 1) {
////        data = GaussData(x, y);
////    } else if (data_type == 2) {
////        data = BinomialData(x, y);
////    } else if (data_type == 4) {
////        data = CoxData(x, y, state);
////    }
//    if (is_weight) {
//        data.add_weight(weight);
//    }
//    if (is_normal) {
//        data.normalize();
//    }
//
//    Algorithm algorithm;
//    if (algorithm_type == 1) {
//        if (data_type == 1) {
//            algorithm = PDAS_LM(data, max_iter);
//        }
//        else if(data_type == 2) {
//            algorithm = PDAS_GLM(data, max_iter, coef0);
//        }
//    }
//    else if (algorithm_type == 2){
//        if (data_type == 1) {
//            algorithm = GROUP_LM(data, max_iter);
//        } else if (data_type == 2) {
//            algorithm = GROUP_GLM(data, max_iter, coef0);
//        }
//    }
////    Eigen::VectorXd beta_init = Eigen::VectorXd::Zero(2);
////	PDAS_LM algorithm(data, max_iter);
////	algorithm.update_sparsity_level(2);
////	algorithm.update_beta_init(beta_init);
////    algorithm.run();
//
//    Metric metric;
//    metric = LinearMetric();
//
//
//    List result;
//    if (path_type == 0) {
//        Eigen::VectorXd beta_init = Eigen::VectorXd::Zero(2);
//        algorithm.update_beta_init(beta_init);
//        algorithm.update_sparsity_level(sparsity_level);
//        algorithm.run();
//        metric.aic(algorithm, data);
//    } else if (path_type == 1) {
//        result = sequential_path(data, algorithm, metric, sequence, is_warm_start);
//    } else if (path_type == 2) {
//        result = gs_path(data, algorithm, metric, s_min, s_max, K_max, epsilon, is_warm_start);
//    }
//
//    return 1;
//}

