//
// Created by jk on 2020/3/18.
//
#ifndef SRC_ALGORITHM_H
#define SRC_ALGORITHM_H

#include "Data.h"
#include "utilities.h"
#include "logistic.h"
#include "poisson.h"
#include "coxph.h"
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>
#include <time.h>

using namespace std;

bool quick_sort_pair_max(std::pair<int, double> x, std::pair<int, double> y);

class Algorithm {
  public:
    Data data;
    vector<Eigen::MatrixXd> PhiG;
    vector<Eigen::MatrixXd> invPhiG;
    Eigen::VectorXd beta_init;
    int group_df;
    int sparsity_level;
    double lambda_level = 0;
    Eigen::VectorXi train_mask;
    int max_iter;
    int exchange_num;
    bool warm_start;
    Eigen::VectorXd beta;
    double coef0_init;
    double coef0;
    double loss;
    Eigen::VectorXi A_out;
    int l;
    int model_fit_max;
    int model_type;
    // int ag_type;
    int algorithm_type;

    Algorithm() = default;

    Algorithm(Data &data, int model_type, int algorithm_type, int max_iter = 100) {
        this->data = data;
        this->max_iter = max_iter;
        this->A_out = Eigen::VectorXi::Zero(data.get_p());
        this->model_type = model_type;
        this->coef0 = 0.0;
        this->beta = Eigen::VectorXd::Zero(data.get_p());
        this->coef0_init = 0.0;
        this->beta_init = Eigen::VectorXd::Zero(data.get_p());
        this->warm_start = true;
        this->exchange_num = 5;
        this->algorithm_type = algorithm_type;
    };

    void update_PhiG(vector<Eigen::MatrixXd>& PhiG) {
      this->PhiG = PhiG;
    }

    void update_invPhiG(vector<Eigen::MatrixXd>& invPhiG) {
      this->invPhiG = invPhiG;
    }

    void set_warm_start(bool warm_start) {
        this->warm_start = warm_start;
    };

    void update_beta_init(Eigen::VectorXd beta_init) {
        // std::cout << "update beta init"<<endl;

        this->beta_init = beta_init;
    };

    void update_coef0_init(double coef0){
        this->coef0_init = coef0;
    };

    void update_group_df(int group_df) {
         this->group_df = group_df;
    };

    void update_sparsity_level(int sparsity_level) {
        // std::cout << "update sparsity level" << endl;
        this->sparsity_level = sparsity_level;
        //  to ensure
        this->group_df = sparsity_level;
    }

    void update_lambda_level(double lambda_level) {
        // std::cout << "update lambda level" << endl;
        this->lambda_level = lambda_level;
    }

    void update_train_mask(Eigen::VectorXi train_mask) {
        // std::cout << "update train mask" << endl;
        this->train_mask = train_mask;
    }

    void update_exchange_num(int exchange_num) {
        this->exchange_num = exchange_num;
    };

    bool get_warm_start() {
        return this->warm_start;
    }

    double get_loss() {
        return this->loss;
    }

    int get_group_df() {
        return this->group_df;
    };

    int get_sparsity_level() {
        return this->sparsity_level;
    }

    Eigen::VectorXd get_beta() {
        return this->beta;
    }

    double get_coef0() {
        return this->coef0;
    }

    Eigen::VectorXi  get_A_out() {
        return this->A_out;
    };

    int get_l() {
        return this->l;
    }

    void fit() {
      int train_n = this->train_mask.size();
      int p = data.get_p();
      // cout<<"train_n: "<<train_n<<", p: "<<p<<endl;
      Eigen::MatrixXd train_x(train_n, p);
      Eigen::VectorXd train_y(train_n);
      Eigen::VectorXd train_weight(train_n);
      int T0 = this->sparsity_level;

      int N = data.get_g_num();
      // cout<<"g_num: "<<N<<endl;
      Eigen::VectorXi g_index = data.get_g_index();
      Eigen::VectorXi g_size = data.get_g_size();
      //cout<<"g_index: "<<g_index<<endl;
      //cout<<"g_size: "<<g_size<<endl;
      //cout<<"train_init"<<endl;

      if(train_n == data.get_n())
      {
        train_x = data.x;
        train_y = data.y;
        train_weight = data.weight;
      }
      else
      {
        for (int i = 0; i < train_n; i++) {
      
          train_x.row(i) = data.x.row(this->train_mask(i)).eval();
          train_y(i) = data.y(this->train_mask(i));
          train_weight(i) = data.weight(this->train_mask(i));
          //  cout<<"i"<<i<<", train_mask: "<<this->train_mask(i)<<", train_x"<<train_x.row(i)<<", train_y(i):"<<train_y(i)<<", weight:"<<train_weight(i)<<endl;
        }
      }

      // cout<<"train_x[29]: "<<train_x.col(29)<<endl;
      // cout<<"train_n: "<<train_n<<endl;
      // cout<<"p: "<<p<<endl;
      // cout<<"N: "<<N<<endl;
      // cout<<"train_mask: "<<this->train_mask<<endl;
      // train_x normalize

      //cout<<"T0: "<<T0<<endl;
      Eigen::VectorXi A = Eigen::VectorXi::Zero(T0);
      // Eigen::VectorXi I = Eigen::VectorXi::Zero(p - T0);
      Eigen::MatrixXi A_list(T0, max_iter +2);
      A_list.col(0) = Eigen::VectorXi::Zero(T0);

      // this->get_A(train_x,train_y,this->beta_init,this->coef0_init,T0,Eigen::VectorXi::Zero(p),train_weight, g_index, g_size, N, A);


      Eigen::MatrixXd X_A = Eigen::MatrixXd::Zero(train_n, T0);
      Eigen::VectorXd beta_A = Eigen::VectorXd::Zero(T0);
      this->beta = this->beta_init;
      this->coef0 = this->coef0_init;
      if(this->algorithm_type == 1 || this->algorithm_type == 5)
      {
         //cout<<"=============pdas fit"<<endl;
        for(this->l=1;this->l<=this->max_iter;l++) {
           //cout<<":::::pdas iter: "<<this->l<<endl;
          // clock_t t1 = clock();
          this->get_A(train_x,train_y,this->beta,this->coef0,T0,train_weight, g_index, g_size, N, A);
          // clock_t t2 = clock();
          // printf("get A time=%f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
          A_list.col(this->l) = A;
          //cout<<"A: "<<A<<endl;
          for(int mm = 0;mm < T0; mm++) {
            X_A.col(mm)=train_x.col(A[mm]);
          }
          // t1 = clock();
          this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0);
          // t2 = clock();
          // printf("fit time=%f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
          //std::cout<<"primary fit end"<<endl;
          this->beta = Eigen::VectorXd::Zero(p);
          // cout<<"betaA: ";
          for(int mm=0;mm<T0;mm++) {
              this->beta(A[mm]) = beta_A(mm);
              // std::cout<<beta_A(mm)<<" ";
          }
          for(int ll=0;ll<this->l;ll++)
          {
            if(A==A_list.col(ll))
            {
              // cout<<"A: "<<A<<endl;
              // cout<<"betaA: "<<beta_A<<endl;
              // cout<<"pdas fit time: "<<this->l<<endl;
              return;
            }

          }
        }
      }
      else
      {
        // cout<<"Group pdas fit"<<endl;
        Eigen::VectorXi ind;
        for(this->l=1;this->l<=this->max_iter;l++) {
          // cout<<"pdas iter: "<<this->l<<endl;
          // cout<<"beta"<<endl;
          // clock_t t1 = clock();
          this->get_A(train_x, train_y, this->beta, this->coef0, T0, train_weight, g_index, g_size, N, A);
          // clock_t t2 = clock();
          // printf("get A time=%f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
          A_list.col(this->l) = A;

          ind = find_ind(A, g_index, g_size, p, N);
          // cout<<"ind"<<endl;
          X_A = X_seg(train_x, train_n, ind);
          // cout<<"X_A"<<endl;
          beta_A = Eigen::VectorXd::Zero(ind.size());
          // t1 = clock();
          this->primary_model_fit(X_A, train_y, train_weight, beta_A, this->coef0);
          // t2 = clock();
          // printf("fit time=%f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
          // cout<<"primary fit end"<<endl;

          // cout<<"A: "<<A<<endl;
          // cout<<"betaA: "<<beta_A<<endl;
          // cout<<"ind: "<<ind<<endl;
          this->beta = Eigen::VectorXd::Zero(p);
          for(int mm=0;mm<ind.size();mm++) {
              this->beta(ind(mm)) = beta_A(mm);
          }
          // cout<<"pdas iter 1"<<endl;
          for(int ll=0;ll<this->l;ll++)
          {
            if(A==A_list.col(ll))
            {
              // cout<<"A: "<<A<<endl;
              // cout<<"betaA: "<<beta_A<<endl;
              // cout<<"A: "<<A<<endl;
              return;
            }
          }



        }
      }
    };

    virtual void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)=0;


    virtual void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)=0;
};

class PdasLm : public Algorithm {
public:
    PdasLm(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 1, algorithm_type, max_iter) {
      this->algorithm_type = algorithm_type;
    };

    void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
    {
        // cout<<"pdas_lm_get_A"<<endl;
        int n=X.rows();
        vector<int>A(T0);

        Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
        Eigen::VectorXd d=(X.transpose()*(y-X*beta-coef)) /double(n);
        Eigen::VectorXd bd=beta+d;
        bd=bd.cwiseAbs();
        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            bd(A[k])=-1.0;
        }
        sort (A.begin(),A.end());
        for(int i=0;i<T0;i++)
            A_out(i) = A[i];
    };

    void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {
        beta = (X.adjoint()*X+this->lambda_level*Eigen::MatrixXd::Identity(X.cols(),X.cols())).colPivHouseholderQr().solve(X.adjoint()*y);
    };
};

class PdasLogistic : public Algorithm {
public:
    PdasLogistic(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 2, algorithm_type, max_iter) {
    };

    void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {

      int n = x.rows();
      int p = x.cols();
      if (n <= p)
      {
        Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, n);
        Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
        X.rightCols(n-1) = x.leftCols(n-1);
        Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd Pi = pi(X, y, beta0, n);
        Eigen::VectorXd log_Pi = Pi.array().log();
        Eigen::VectorXd log_1_Pi = (one-Pi).array().log();
        double loglik0 = (y.cwiseProduct(log_Pi)+(one-y).cwiseProduct(log_1_Pi)).dot(weights);
        Eigen::VectorXd W = Pi.cwiseProduct(one-Pi);
        Eigen::VectorXd Z = X*beta0+(y-Pi).cwiseQuotient(W);
        W = W.cwiseProduct(weights);
        for (int i=0;i<n;i++)
        {
          X_new.row(i) = X.row(i)*W(i);
        }
        beta1 = (X_new.transpose()*X).ldlt().solve(X_new.transpose()*Z);

        double loglik1;

        int j;
        for(j=0;j<30;j++)
        {
          Pi = pi(X, y, beta1, n);
          log_Pi = Pi.array().log();
          log_1_Pi = (one-Pi).array().log();
          loglik1 = (y.cwiseProduct(log_Pi)+(one-y).cwiseProduct(log_1_Pi)).dot(weights);
          if (abs(loglik0-loglik1)/(0.1+abs(loglik1)) < 1e-6)
          {
            break;
          }
          beta0 = beta1;
          loglik0 = loglik1;
          W = Pi.cwiseProduct(one-Pi);
          for(int i=0; i<n; i++){
            if(W(i) <0.001) W(i) = 0.001;
          }
          Z = X*beta0+(y-Pi).cwiseQuotient(W);
          W = W.cwiseProduct(weights);
          for (int i=0;i<n;i++)
          {
            X_new.row(i) = X.row(i)*W(i);
          }
          beta1 = (X_new.transpose()*X).ldlt().solve(X_new.transpose()*Z);

        }
        for(int i=0;i<p;i++){
            if(i<n) beta(i) = beta0(i+1);
            else  beta(i)=0;
        }
        coef0 = beta0(0);
        }
      else
      {
        Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p+1);
        X.rightCols(p) = x;
        Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p+1);
        Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p+1);
        Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p+1);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
        Eigen::VectorXd Pi = pi(X, y, beta0, n);
        Eigen::VectorXd log_Pi = Pi.array().log();
        Eigen::VectorXd log_1_Pi = (one-Pi).array().log();
        double loglik0 = (y.cwiseProduct(log_Pi)+(one-y).cwiseProduct(log_1_Pi)).dot(weights);
        Eigen::VectorXd W = Pi.cwiseProduct(one-Pi);
        Eigen::VectorXd Z = X*beta0+(y-Pi).cwiseQuotient(W);
        W = W.cwiseProduct(weights);
        for (int i=0;i<n;i++)
        {
          X_new.row(i) = X.row(i)*W(i);
        }
        beta1 = (X_new.transpose()*X).ldlt().solve(X_new.transpose()*Z);
        double loglik1;

        int j;
        for(j=0;j<30;j++)
        {
          Pi = pi(X, y, beta1, n);
          log_Pi = Pi.array().log();
          log_1_Pi = (one-Pi).array().log();
          loglik1 = (y.cwiseProduct(log_Pi)+(one-y).cwiseProduct(log_1_Pi)).dot(weights);
          if (abs(loglik0-loglik1)/(0.1+abs(loglik1)) < 1e-6)
          {
            break;
          }
          beta0 = beta1;
          loglik0 = loglik1;
          W = Pi.cwiseProduct(one-Pi);
          for(int i=0; i<n; i++){
            if(W(i) <0.001) W(i) = 0.001;
          }
          Z = X*beta0+(y-Pi).cwiseQuotient(W);
          W = W.cwiseProduct(weights);
          for (int i=0;i<n;i++)
          {
            X_new.row(i) = X.row(i)*W(i);
          }
          beta1 = (X_new.transpose()*X).ldlt().solve(X_new.transpose()*Z);
        }
        for(int i=0;i<p;i++)
            beta(i) = beta0(i+1);
        coef0 = beta0(0);
      }
    }

    void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
    {
        int n=X.rows();
        int p=X.cols();
        Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
        vector<int>A(T0);
        Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;;
        Eigen::VectorXd xbeta_exp = X*beta+coef;
        //
        for(int i=0;i<=n-1;i++) {
            if(xbeta_exp(i)>30.0) xbeta_exp(i) = 30.0;
            if(xbeta_exp(i)<-30.0) xbeta_exp(i) = -30.0;
        }
        xbeta_exp = xbeta_exp.array().exp();
        Eigen::VectorXd pr = xbeta_exp.array()/(xbeta_exp+one).array();
        Eigen::VectorXd l1=-X.adjoint()*((y-pr).cwiseProduct(weights));
        X=X.array().square();
        Eigen::VectorXd l2=(X.adjoint())*((pr.cwiseProduct(one-pr)).cwiseProduct(weights));
        Eigen::VectorXd d=-l1.cwiseQuotient(l2);
        bd = (beta+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
        for(int k=0;k<T0;k++) {
            bd.maxCoeff(&A[k]);
            bd(A[k])=-1.0;
        }
        sort(A.begin(),A.end());
        for(int i=0;i<T0;i++)
          A_out(i) = A[i];
    };

};

class PdasPoisson : public Algorithm {
public:
    PdasPoisson(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 3, algorithm_type, max_iter) {
      this->algorithm_type = algorithm_type;
    };

    void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {
      int n = x.rows();
      int p = x.cols();
      Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p+1);
      X.rightCols(p) = x;
      Eigen::MatrixXd X_trans = X.transpose();
      Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(p+1, n);
      Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p+1);
      beta0.tail(p) = beta;
      beta0(0) = coef0;
      Eigen::VectorXd eta = X*beta0;
      Eigen::VectorXd expeta = eta.array().exp();
      Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
      double loglik0 = 1e5;
      double loglik1;

      for (int j=0;j<50;j++)
      {
        for (int i=0;i<n;i++) {
          temp.col(i) = X_trans.col(i)*expeta(i)*weights(i);
        }
        z = eta+(y-expeta).cwiseQuotient(expeta);
        beta0 = (temp*X).ldlt().solve(temp*z);
        eta = X*beta0;
        for(int i=0;i<=n-1;i++)
        {
          if(eta(i)<-30.0) eta(i) = -30.0;
          if(eta(i)>30.0) eta(i) = 30.0;
        }
        expeta = eta.array().exp();
        loglik1 = (y.cwiseProduct(eta)-expeta).dot(weights);
        if (abs(loglik0-loglik1)/abs(0.1+loglik0) < 1e-6)  break;
        loglik0 = loglik1;
      }
      for(int i=0;i<p;i++)
          beta(i) = beta0(i+1);
      coef0 = beta0(0);
    }

    void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
    {
        // cout<<"poisson_get_A"<<endl;
        int n=X.rows();
        int p=X.cols();
        int i;
        vector<int>A(T0);

        Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
        Eigen::VectorXd xbeta_exp = X*beta+coef;
        for(i=0;i<=n-1;i++) {
            if(xbeta_exp(i)>30.0) xbeta_exp(i) = 30.0;
            if(xbeta_exp(i)<-30.0) xbeta_exp(i) = -30.0;
        }
        xbeta_exp = xbeta_exp.array().exp();

        Eigen::VectorXd res = y-xbeta_exp;
        Eigen::VectorXd g(p);
        Eigen::VectorXd bd;
        Eigen::MatrixXd Xsquare;
        for(i=0;i<p;i++){
            g(i) = -res.dot(X.col(i));
        }

        // std::cout<<"Poisson fit 3"<<endl;

        Xsquare = X.array().square();

        Eigen::VectorXd h(p);
        for(i=0;i<p;i++){
            h(i) = xbeta_exp.dot(Xsquare.col(i));
        }
        // std::cout<<"Poisson fit 4"<<endl;
        bd = h.cwiseProduct((beta - g.cwiseQuotient(h)).cwiseAbs2());
        // std::cout<<"Poisson fit 5"<<endl;
        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            bd(A[k]) = 0.0;
        }
        sort(A.begin(),A.end());
        for(int i=0;i<T0;i++)
            A_out(i) = A[i];

        // cout<<"poisson_get_A end"<<endl;
    }

};

class PdasCox : public Algorithm {
public:
    PdasCox(Data &data, int algorithm_type, int max_iter) : Algorithm(data, 4, algorithm_type, max_iter) {
      this->algorithm_type = algorithm_type;
    };

    void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd status, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {
      int n = X.rows();
      int p = X.cols();
      // cout<<"cox_fit"<<endl;
      Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd theta(n);
      Eigen::MatrixXd one = (Eigen::MatrixXd::Ones(n, n)).triangularView<Eigen::Upper>();
      Eigen::MatrixXd x_theta(n, p);
      Eigen::VectorXd xij_theta(n);
      Eigen::VectorXd cum_theta(n);
      Eigen::VectorXd g(p);
      Eigen::MatrixXd h(p, p);
      Eigen::VectorXd d(p);
      double loglik0 = 1e5;
      double loglik1;

      double step;
      int m;
      int l;
      for (l=1;l<=30;l++)
      {
        step = 0.5;
        m = 1;
        theta = X*beta0;
        for (int i=0;i<n;i++)
        {
          if (theta(i) > 30) theta(i) = 30;
          else if (theta(i) < -30) theta(i) = -30;
        }
        theta = theta.array().exp();
        cum_theta = one*theta;
        x_theta = X.array().colwise()*theta.array();
        x_theta = one*x_theta;
        x_theta = x_theta.array().colwise()/cum_theta.array();
        g = (X-x_theta).transpose()*(weights.cwiseProduct(status));

        for (int k1=0;k1<p;k1++)
        {
          for (int k2=k1;k2<p;k2++)
          {
            xij_theta = (theta.cwiseProduct(X.col(k1))).cwiseProduct(X.col(k2));
            for(int j=n-2;j>=0;j--)
            {
              xij_theta(j) = xij_theta(j+1) + xij_theta(j);
            }
            h(k1, k2) = -(xij_theta.cwiseQuotient(cum_theta) - x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(status));
            h(k2, k1) = h(k1, k2);
          }
        }
        d = h.ldlt().solve(g);
        beta1 = beta0-pow(step, m)*d;
        loglik1 = loglik_cox(X, status, beta1, weights);
        while ((loglik0 > loglik1) && (m<5))
        {
          m = m+1;
          beta1 = beta0-pow(step, m)*d;
          loglik1 = loglik_cox(X, status, beta1, weights);
        }
        if (abs(loglik0-loglik1)/abs(0.1+loglik0) < 1e-5)
        {
          break;
        }
        beta0 = beta1;
        loglik0 = loglik1;
      }
      // cout<<"cox_fit end"<<endl;
      beta = beta0;
    }

    void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
    {
        int n=X.rows();
        int p=X.cols();
        Eigen::VectorXd l1 = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd l2 = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd cum_theta=Eigen::VectorXd::Zero(n);
        Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
        Eigen::MatrixXd xtheta(n,p);
        Eigen::MatrixXd x2theta(n,p);
        vector<int>A(T0);
        Eigen::VectorXd theta=X*beta;
        // for(int i=0;i<=n-1;i++) {
        //     if(theta(i)>30) theta(i) = 30;
        //     if(theta(i)<-30) theta(i) = -30;
        // }
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
        for(int k=0;k<y.size();k++){
          xtheta.row(k)=xtheta.row(k) * y[k];
          x2theta.row(k)=x2theta.row(k) * y[k];
        }
        l1=-xtheta.adjoint()*weights;
        // cout<<"l1: "<<l1<<endl;
        l2=x2theta.adjoint()*weights;
        // cout<<"l2: "<<l2<<endl;
        d=-l1.cwiseQuotient(l2);
        bd=beta+d;
        bd=bd.cwiseAbs();
        bd=bd.cwiseProduct(l2.cwiseSqrt());
        bd=bd.array().square();
        // cout<<"bd: "<<bd<<endl;
        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            bd(A[k])=-1.0;
        }
        sort (A.begin(),A.end());
        for(int i=0;i<T0;i++)
            A_out(i) = A[i];
    }

};

class L0L2Lm : public Algorithm {
public:
     L0L2Lm(Data &data, int algorithm_type, int max_iter) : Algorithm(data, 1, algorithm_type, max_iter) {
       //cout<<"L0L2Lm"<<endl;
      }

   void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
    {
        int n=X.rows();
        vector<int>A(T0);
        Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
        // cout<<"coef0: "<<coef0<<endl;
        Eigen::VectorXd d=(X.adjoint()*(y-X*beta-coef)/double(n) - 2*this->lambda_level*beta) / sqrt(1 + 2*this->lambda_level);
        Eigen::VectorXd bd=sqrt(1 + 2 * this-> lambda_level) * beta + d;
        bd=bd.cwiseAbs2();
        // cout<<"this->lambda_level: "<<this->lambda_level<<endl;
        // cout<<"bd:"<<bd<<endl;
        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            // cout<<"max00 "<<k<<" "<<A[k]<<" bd :"<<bd(A[k])<<"d :"<<d(A[k])<<"beta: "<<sqrt(1 + 2 * this-> lambda_level) * beta(A[k])<<endl;
            bd(A[k])=-1.0;
        }
        sort (A.begin(),A.end());
        for(int i=0;i<T0;i++)
            A_out(i) = A[i];
    };

    void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {
      // cout<<"lambda"<<this->lambda_level<<endl;
      // cout<<"X:"<<X<<endl;
      // cout<<"X.adjoint()"<<X.adjoint()<<endl;
      // cout<<"X:"<<X<<endl;
      // cout<<"X.transpose"<<X.transpose()<<endl;
      // cout<<"X:"<<X<<endl;
      beta = (X.adjoint()*X+this->lambda_level*Eigen::MatrixXd::Identity(X.cols(),X.cols())).colPivHouseholderQr().solve(X.adjoint()*y);
    }
};

class L0L2Logistic : public Algorithm {
public:
    L0L2Logistic(Data &data, int algorithm_type, int max_iter) : Algorithm(data, 2, algorithm_type, max_iter) {
    };

    void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {

      int n = x.rows();
      int p = x.cols();
      Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p+1);
      X.rightCols(p) = x;
      Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p+1);
      Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p+1);
      Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p+1);
      Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
      Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p+1,p+1);
      lambdamat(0,0)=0;
      Eigen::VectorXd Pi = pi(X, y, beta0, n);
      Eigen::VectorXd log_Pi = Pi.array().log();
      Eigen::VectorXd log_1_Pi = (one-Pi).array().log();
      double loglik0 = (y.cwiseProduct(log_Pi)+(one-y).cwiseProduct(log_1_Pi)).dot(weights);
      Eigen::VectorXd W = Pi.cwiseProduct(one-Pi);
      for(int i=0; i<n; i++){
        if(W(i) <0.001) W(i) = 0.001;
      }
      Eigen::VectorXd Z = X*beta0+(y-Pi).cwiseQuotient(W);
      W = W.cwiseProduct(weights);
      //cout<<"W: "<<W.topLeftCorner(3,3)<<endl;
      for (int i=0;i<n;i++)
      {
        X_new.row(i) = X.row(i)*W(i);
      }
      beta1 = (2*this->lambda_level*lambdamat+X_new.transpose()*X).ldlt().solve(X_new.transpose()*Z);
      double loglik1;

      int j;
      for(j=0;j<30;j++)
      {
        Pi = pi(X, y, beta1, n);
        // cout<<"PI:"<<Pi.head(5)<<endl;
        log_Pi = Pi.array().log();
        log_1_Pi = (one-Pi).array().log();
        loglik1 = (y.cwiseProduct(log_Pi)+(one-y).cwiseProduct(log_1_Pi)).dot(weights);
        if (abs(loglik0-loglik1)/(0.1+abs(loglik1)) < 1e-6)
        {
          break;
        }
        beta0 = beta1;
        loglik0 = loglik1;
        W = Pi.cwiseProduct(one-Pi);
        for(int i=0; i<n; i++){
          if(W(i) <0.001) W(i) = 0.001;
        }
        Z = X*beta0+(y-Pi).cwiseQuotient(W);
        W = W.cwiseProduct(weights);
        //cout<<"W: "<<W.topLeftCorner(3,3)<<endl;
        for (int i=0;i<n;i++)
        {
          X_new.row(i) = X.row(i)*W(i);
        }
        beta1 = (2*this->lambda_level*lambdamat+X_new.transpose()*X).ldlt().solve(X_new.transpose()*Z);
      }
      for(int i=0;i<p;i++)
          beta(i) = beta0(i+1);
      coef0 = beta0(0);
      }

     void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
     {
        int n=X.rows();
        int p=X.cols();
        Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
        vector<int>A(T0);
        Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
        Eigen::VectorXd xbeta_exp = X*beta+coef;
        for(int i=0;i<=n-1;i++) {
            if(xbeta_exp(i)>25.0) xbeta_exp(i) = 25.0;
            if(xbeta_exp(i)<-25.0) xbeta_exp(i) = -25.0;
        }
        xbeta_exp = xbeta_exp.array().exp();
        Eigen::VectorXd pr = xbeta_exp.array()/(xbeta_exp+one).array();
        Eigen::VectorXd l1=-X.adjoint()*((y-pr).cwiseProduct(weights))+2*this->lambda_level*beta;
        X=X.array().square();
        Eigen::VectorXd l2=(X.adjoint())*((pr.cwiseProduct(one-pr)).cwiseProduct(weights))+2*this->lambda_level*Eigen::MatrixXd::Ones(p,1);
        Eigen::VectorXd d=-l1.cwiseQuotient(l2);
        bd = (beta+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            bd(A[k])=-1.0;
        }
        sort (A.begin(),A.end());
        for(int i=0;i<T0;i++)
          A_out(i) = A[i];
        }
};

class L0L2Poisson : public Algorithm {
public:
    L0L2Poisson(Data &data, int algorithm_type, unsigned int max_iter = 20) : Algorithm(data, 3, algorithm_type, max_iter) {
        this->model_fit_max = model_fit_max;
    };

    // void get_GroupA(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, Eigen::VectorXd weights, Eigen::VectorXi &A_out){}

    void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {
      int n = x.rows();
      int p = x.cols();
      Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p+1);
      X.rightCols(p) = x;
      Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p+1,p+1);
      lambdamat(0,0)=0;
      Eigen::MatrixXd X_trans = X.transpose();
      Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(p+1, n);
      Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p+1);
      beta0.tail(p) = beta;
      beta0(0) = coef0;
      Eigen::VectorXd eta = X*beta0;
      Eigen::VectorXd expeta = eta.array().exp();
      Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
      double loglik0 = 1e5;
      double loglik1;

      for (int j=0;j<50;j++)
      {
        for (int i=0;i<n;i++) {
          temp.col(i) = X_trans.col(i)*expeta(i)*weights(i);
        }
        z = eta+(y-expeta).cwiseQuotient(expeta);
        beta0 = (2*this->lambda_level*lambdamat + temp*X).ldlt().solve(temp*z);
        eta = X*beta0;
        for(int i=0;i<=n-1;i++)
        {
          if(eta(i)<-30.0) eta(i) = -30.0;
          if(eta(i)>30.0) eta(i) = 30.0;
        }
        expeta = eta.array().exp();
        loglik1 = (y.cwiseProduct(eta)-expeta).dot(weights);
        if (abs(loglik0-loglik1)/abs(0.1+loglik0) < 1e-6)  break;
        loglik0 = loglik1;
      }
      for(int i=0;i<p;i++)
          beta(i) = beta0(i+1);
      coef0 = beta0(0);
    }
    // void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    // {
    //   int n = x.rows();
    //   int p = x.cols();
    //   if (n <= p)
    //   {
    //     // cout<<"poisson_fit 1"<<endl;
    //     Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, n);
    //     Eigen::MatrixXd h(n, n);
    //     Eigen::VectorXd d = Eigen::VectorXd::Zero(n);
    //     Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
    //     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(n);
    //     Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(n);
    //     Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(n, n);
    //     double loglik0;
    //     double loglik1;
    //     X = x.leftCols(n);
    //     Eigen::VectorXd eta = Eigen::VectorXd::Zero(n);
    //     Eigen::VectorXd expeta = Eigen::VectorXd::Zero(n);
    //     Eigen::VectorXd expeta_w = Eigen::VectorXd::Zero(n);
    //     int j;
    //     for(j=0;j<100;j++)
    //     {
    //     //   cout<<"j = "<< j<<endl;
    //       double step = 0.2;
    //       int m = 0;
    //       eta = X*beta0;
    //       for(int i=0;i<=n-1;i++)
    //       {
    //         if(eta(i)<-30.0) eta(i) = -30.0;
    //         if(eta(i)>30.0) eta(i) = 30.0;
    //       }
    //       expeta = eta.array().exp();
    //       expeta_w = expeta.cwiseProduct(weights);
    //       for (int i=0;i<n;i++)
    //       {
    //         temp.col(i) = X.col(i)*expeta_w;
    //       }
    //       g = X.transpose()*(y-expeta).cwiseProduct(weights);
    //       h = X.transpose()*temp;
    //       d = h.ldlt().solve(g);
    //       beta1 = beta0+pow(step, m)*d;
    //       loglik0 = loglik_poiss(X, y, beta0, n, weights);
    //       loglik1 = loglik_poiss(X, y, beta1, n, weights);
    //       while ((loglik0 >= loglik1) && (m<10))
    //       {
    //         m = m+1;
    //         beta1 = beta0+pow(step, m)*d;
    //         loglik1 = loglik_poiss(X, y, beta1, n, weights);
    //       }
    //       beta0 = beta1;
    //       if (abs(loglik0-loglik1)/abs(loglik0) < 1e-8)
    //       {
    //         break;
    //       }
    //     }
    //     for(int i=0;i<p;i++)
    //         beta(i) = beta0(i+1);
    //     coef0 = beta0(0);

    //   }

    //   else {
    //     // cout<<"poisson_fit 2"<<endl;
    //     Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p+1);
    //     X.rightCols(p) = x;
    //     Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p+1,p+1);
    //     lambdamat(0,0)=0;
    //     Eigen::MatrixXd h(p+1, p+1);
    //     Eigen::VectorXd d = Eigen::VectorXd::Zero(p+1);
    //     Eigen::VectorXd g = Eigen::VectorXd::Zero(p+1);
    //     Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p+1);
    //     Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p+1);
    //     Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(n, p+1);
    //     Eigen::VectorXd eta = Eigen::VectorXd::Zero(n);
    //     Eigen::VectorXd expeta = Eigen::VectorXd::Zero(n);
    //     Eigen::VectorXd expeta_w = Eigen::VectorXd::Zero(n);
    //     double loglik0;
    //     double loglik1;

    //     int j;
    //     for(j=0;j<100;j++)
    //     {
    //     //   cout<<"j= "<<j<<endl;
    //       double step = 0.2;
    //       int m = 0;
    //       eta = X*beta0;
    //       for(int i=0;i<=n-1;i++)
    //       {
    //         if(eta(i)<-30.0) eta(i) = -30.0;
    //         if(eta(i)>30.0) eta(i) = 30.0;
    //       }
    //       expeta = eta.array().exp();
    //       expeta_w = expeta.cwiseProduct(weights);
    //       for (int i=0; i<p+1; i++)
    //       {
    //         temp.col(i) = X.col(i)*expeta_w;
    //       }
    //       g = X.transpose()*(y-expeta).cwiseProduct(weights);
    //       h = X.transpose()*temp + 2 * this->lambda_level * lambdamat;
    //       d = h.ldlt().solve(g);
    //       beta1 = beta0+pow(step, m)*d;
    //       loglik0 = loglik_poiss(x, y, beta0, n, weights);
    //       loglik1 = loglik_poiss(x, y, beta1, n, weights);
    //       while ((loglik0 >= loglik1) && (m<10))
    //       {
    //         m = m+1;
    //         beta1 = beta0+pow(step, m)*d;
    //         loglik1 = loglik_poiss(x, y, beta1, n, weights);
    //       }
    //     //   cout<<"m: "<<m<<endl;
    //       beta0 = beta1;
    //       if (abs(loglik0-loglik1)/abs(loglik0) < 1e-8)
    //       {
    //         break;
    //       }
    //     }
    //     // cout<<"poisson_fit end"<<endl;
    //     for(int i=0;i<p;i++)
    //         beta(i) = beta0(i+1);
    //     coef0 = beta0(0);
    //     }
    // }

    void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
    {
        // cout<<"poisson_get_A"<<endl;
        int n=X.rows();
        int p=X.cols();
        int i;

        // vector<int>E(p);
        // for(int k=0;k<=p-1;k++) {
        //     E[k]=k;
        // }
        // vector<int>I(p-T0);
        vector<int>A(T0);

        Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
        Eigen::VectorXd xbeta_exp = X*beta+coef;
        for(i=0;i<=n-1;i++) {
            if(xbeta_exp(i)>30.0) xbeta_exp(i) = 30.0;
            if(xbeta_exp(i)<-30.0) xbeta_exp(i) = -30.0;
        }
        xbeta_exp = xbeta_exp.array().exp();

        Eigen::VectorXd res = y-xbeta_exp;
        Eigen::VectorXd l1(p);
        Eigen::VectorXd bd;
        Eigen::MatrixXd Xsquare;
        for(i=0;i<p;i++){
            l1(i) = -res.dot(X.col(i)) + 2*this->lambda_level*beta(i);
        }

        // std::cout<<"Poisson fit 3"<<endl;

        Xsquare = X.array().square();

        Eigen::VectorXd l2(p);
        for(i=0;i<p;i++){
            l2(i) = xbeta_exp.dot(Xsquare.col(i)) + 2*this->lambda_level;
        }
        // std::cout<<"Poisson fit 4"<<endl;
        Eigen::VectorXd d=-l1.cwiseQuotient(l2);
        // if(B.size()<p) {
        //     for(int k=0;k<=B.size()-1;k++) {
        //       d(B(k))=0.0;
        //     }
        // }
        bd = (beta+d).cwiseAbs().cwiseProduct(l2.cwiseSqrt());
        //cout<<"l2.cwiseSqrt(): "<<l2.cwiseSqrt()<<", bd: "<<bd<<endl;
        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            //cout<<"bd(A["<<k<<"]): "<<bd(A[k])<<", A["<<k<<"]: "<<A[k]<<endl;
            bd(A[k])=-1.0;
        }
        sort(A.begin(),A.end());
        // set_difference(E.begin(),E.end(), A.begin(),A.end(),I.begin());
        for(int i=0;i<T0;i++)
            A_out(i) = A[i];
        // for(int i=0;i<p-T0;i++)
        //     I_out(i) = I[i];

        // cout<<"poisson_get_A end"<<endl;
    }
};

class L0L2Cox : public Algorithm {
public:
    L0L2Cox(Data &data, int algorithm_type, unsigned int max_iter = 20) : Algorithm(data, 4, algorithm_type, max_iter) {
        this->model_fit_max = model_fit_max;
    };

    // void get_GroupA(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, Eigen::VectorXd weights, Eigen::VectorXi &A_out){}


    void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd status, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {
      int n = X.rows();
      int p = X.cols();

      Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p);

      Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p,p);
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
        g = (X-x_theta).transpose()*(weights.cwiseProduct(status)) + 2 * this->lambda_level * beta;
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
        h = h + this->lambda_level * lambdamat;
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
        // cout<<"m: "<<m<<endl;
        beta0 = beta1;
        if (abs(loglik0-loglik1)/abs(loglik0) < 1e-8)
        {
          break;
        }
      }

      beta = beta0;
    }

    void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out)
    {
        int n=X.rows();
        int p=X.cols();

        Eigen::VectorXd l1 = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd l2 = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd cum_theta=Eigen::VectorXd::Zero(n);
        Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
        Eigen::MatrixXd xtheta(n,p);
        Eigen::MatrixXd x2theta(n,p);
        // vector<int>E(p);
        // for(int k=0;k<=p-1;k++) {
        //     E[k]=k;
        // }
        vector<int>A(T0);
        // vector<int>I(p-T0);
        Eigen::VectorXd theta=X*beta;
        //cout<<",  theta: "<<theta;
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
       //cout<<",   x2theta "<<x2theta;
        xtheta=X.array()-xtheta.array();
        //cout<<",  xtheta "<<xtheta;
        for(unsigned int k=0;k<y.size();k++) {
            if(y[k] == 0.0)
            {
                xtheta.row(k)=Eigen::VectorXd::Zero(p);
                x2theta.row(k)=Eigen::VectorXd::Zero(p);
            }
        }
        l1=-xtheta.adjoint()*weights + 2*this->lambda_level * beta;
        l2=x2theta.adjoint()*weights + 2*this->lambda_level * Eigen::MatrixXd::Ones(p,1);
        d=-l1.cwiseQuotient(l2);
        //cout<<", l1: "<<l1;
        //cout<<", l2: "<<l2;
        //cout<<", d: "<<d;
        // if(B.size()<p) {
        //     for(int k=0;k<=B.size()-1;k++) {
        //         d(B(k))=0.0;
        //     }
        // }
        bd=beta+d;
        //cout<<", beta: "<<beta;
        bd=bd.cwiseAbs();
        bd=bd.cwiseProduct(l2.cwiseSqrt());
        //cout<<", l2.cwiseSqrt(): "<<l2.cwiseSqrt();
        //cout<<", bd: "<<bd<<endl;
        for(int k=0;k<=T0-1;k++) {
            bd.maxCoeff(&A[k]);
            bd(A[k])=0.0;
        }
        sort (A.begin(),A.end());
        // set_difference(E.begin(),E.end(), A.begin(),A.end(),I.begin());
        for(int i=0;i<T0;i++)
            A_out(i) = A[i];
        // for(int i=0;i<p-T0;i++)
        //     I_out(i) = I[i];
    }
};

class GroupPdasLm : public Algorithm {
  public:
    GroupPdasLm(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 1, algorithm_type, max_iter) {
    };

    void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                        Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out){
      int n=X.rows();
      int p=X.cols();
      // cout<<"n: "<<n<<", p:"<<p<<endl;
      // cout<<"coef0: "<<coef0<<endl;
      // vector<Eigen::MatrixXd> PhiG = (this->algorithm_type == 5) ? this->PhiG : Phi(X, index, gsize, n, p, N, this->lambda_level);
      // vector<Eigen::MatrixXd> invPhiG = (this->algorithm_type == 5) ? this->invPhiG : invPhi(PhiG, N);
      vector<Eigen::MatrixXd> PhiG = Phi(X, index, gsize, n, p, N, this->lambda_level);
      vector<Eigen::MatrixXd> invPhiG = invPhi(PhiG, N);
      Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd bd = Eigen::VectorXd::Zero(N);
      Eigen::VectorXd coef = Eigen::VectorXd::Ones(n) * coef0;
      Eigen::VectorXd d = X.adjoint()*(y-X*beta-coef)/double(n) - 2*this->lambda_level*beta;
      vector<int> A(T0, -1);

      for(int i=0;i<N;i++) {
        // cout<<"i: "<<i<<", ";
        Eigen::MatrixXd phiG = PhiG[i];
        Eigen::MatrixXd invphiG = invPhiG[i];
        betabar.segment(index(i), gsize(i)) = phiG*beta.segment(index(i), gsize(i));
        dbar.segment(index(i), gsize(i)) = invphiG*d.segment(index(i), gsize(i));
      }
      Eigen::VectorXd temp = betabar + dbar;
       //cout<<"temp: "<<temp<<"N: "<<N<<endl;
      // cout<<"lambda_level: "<<this->lambda_level<<endl;
      for(int i=0;i<N;i++) {
        bd(i) = (temp.segment(index(i), gsize(i))).squaredNorm()/gsize(i);
         //cout<<"i: "<<i<<",bd("<<i<<"): "<<bd(i)<<", ";
      }
      // cout<<"bd: "<<bd<<endl;
      // for(int k=0;k<T0;k++) {
      //   bd.maxCoeff(&A[k]);
      //   // cout<<"max00 "<<k<<" "<<A[k]<<" bd :"<<bd(A[k])<<" d :"<<dbar(A[k])<<" beta: "<<betabar(A[k])<<" phiG: "<<PhiG[A[k]]<<" invphiG: "<<invPhiG[A[k]]<<endl;
      //   bd(A[k]) = -1.0;
      // }
      // sort(A.begin(), A.end());
      // for(int i=0;i<T0;i++) {
      //   A_out(i) = A[i];

      
      // Eigen::VectorXd bd_tmp_1 = bd, bd_tmp_2=bd;
      // Eigen::VectorXi A_out_1(T0);
      // Eigen::VectorXi A_out_2(T0);
      // clock_t t1 = clock();
      // max_k_2(bd_tmp_1, T0, A_out_1);
      // clock_t t2 = clock();
      // printf("max k new time=%f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
      // cout<<"A_out_1: "<<A_out_1<<endl;

      // t1 = clock();
      // for(int k=0;k<T0;k++) {
      //   bd_tmp_2.maxCoeff(&A[k]);
      //   // cout<<"max00 "<<k<<" "<<A[k]<<" bd :"<<bd(A[k])<<" dbar :"<<dbar(A[k])<<" beta: "<<betabar(A[k])<<"d: "<<d(A[k])<<endl;
      //   bd_tmp_2(A[k])=0;
      // }
      // sort(A.begin(), A.end());
      // for(int i=0;i<T0;i++) {
      // A_out_2(i) = A[i];
      // }
      // t2 = clock();
      // printf("max k old time=%f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
      // cout<<"A_out_2: "<<A_out_2<<endl;

      max_k(bd, T0, A_out);
  };

    void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {
      // cout<<"lambda"<<this->lambda_level<<endl;
      // Matrix<double, 3, 3> A;
      // A << 1, 2, 3, 4, 5, 6, 7, 8, 9;
      // // cout<<"lambda"<<this->lambda_level<<endl;
      // cout<<"A:"<<A<<endl;
      // cout<<"A.adjoint()"<<A.adjoint()<<endl;
      // cout<<"A:"<<A<<endl;
      // cout<<"A.transpose"<<A.transpose()<<endl;
      // cout<<"A:"<<A<<endl;
      // beta = X.colPivHouseholderQr().solve(y);
      beta = (X.adjoint()*X+this->lambda_level*Eigen::MatrixXd::Identity(X.cols(),X.cols())).colPivHouseholderQr().solve(X.adjoint()*y);
    };
};

class GroupPdasLogistic : public Algorithm {
public:
    GroupPdasLogistic(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 2, algorithm_type, max_iter) {
        this->algorithm_type = algorithm_type;
    };

    void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {

      int n = x.rows();
      int p = x.cols();

      Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p+1);
      X.rightCols(p) = x;
      Eigen::MatrixXd X_new = Eigen::MatrixXd::Zero(n, p+1);
      Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p+1);
      Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p+1);
      Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
      Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p+1,p+1);
      lambdamat(0,0)=0;
      Eigen::VectorXd Pi = pi(X, y, beta0, n);
      Eigen::VectorXd log_Pi = Pi.array().log();
      Eigen::VectorXd log_1_Pi = (one-Pi).array().log();
      double loglik0 = (y.cwiseProduct(log_Pi)+(one-y).cwiseProduct(log_1_Pi)).dot(weights);
      Eigen::VectorXd W = Pi.cwiseProduct(one-Pi);
      Eigen::VectorXd Z = X*beta0+(y-Pi).cwiseQuotient(W);
      W = W.cwiseProduct(weights);
      for (int i=0;i<n;i++)
      {
        X_new.row(i) = X.row(i)*W(i);
      }
      beta1 = (2*this->lambda_level*lambdamat+X_new.transpose()*X).ldlt().solve(X_new.transpose()*Z);
      double loglik1;

      int j;
      for(j=0;j<30;j++)
      {
        Pi = pi(X, y, beta1, n);
        log_Pi = Pi.array().log();
        log_1_Pi = (one-Pi).array().log();
        loglik1 = (y.cwiseProduct(log_Pi)+(one-y).cwiseProduct(log_1_Pi)).dot(weights);
        if (abs(loglik0-loglik1)/(0.1+abs(loglik1)) < 1e-6)
        {
          break;
        }
        beta0 = beta1;
        loglik0 = loglik1;
        W = Pi.cwiseProduct(one-Pi);
        for(int i=0; i<n; i++){
          if(W(i) < 0.001) W(i) = 0.001;
        }
        Z = X*beta0+(y-Pi).cwiseQuotient(W);
        W = W.cwiseProduct(weights);
        for (int i=0;i<n;i++)
        {
          X_new.row(i) = X.row(i)*W(i);
        }
        beta1 = (2*this->lambda_level*lambdamat+X_new.transpose()*X).ldlt().solve(X_new.transpose()*Z);
      }
      for(int i=0;i<p;i++)
          beta(i) = beta0(i+1);
      coef0 = beta0(0);
    }

    void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out){

      int n=X.rows();
      int p=X.cols();
      //cout<<"n: "<<n<<", p: "<<p<<endl;
      Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd one = Eigen::VectorXd::Ones(n);
      Eigen::VectorXd bd = Eigen::VectorXd::Zero(N);
      Eigen::VectorXd eta(n);
      Eigen::VectorXd expeta(n);
      Eigen::VectorXd pr(n);
      Eigen::VectorXd g(n);
      Eigen::VectorXd h(n);
      Eigen::VectorXd d(n);
      vector<int> A(T0, -1);

      eta = X*beta+one*coef0;
      for (int i=0;i<n;i++) {
        //cout<<"i :"<<i<<", ";
        if (eta(i) > 30) eta(i) = 30;
        else if (eta(i) < -30) eta(i) = -30;
      }
      expeta = eta.array().exp();
      pr = expeta.array()/(expeta+one).array();
      g = weights.array()*(y-pr).array();
      h = weights.array()*pr.array()*(one-pr).array();
      d = X.transpose()*g - 2*this->lambda_level*beta;
      // h = pr.array()*(one-pr).array();
      //cout<<"d: "<<d<<endl;
      for(int i=0;i<N;i++) {
        //cout<<"i: "<<i<<", "<<"gsize(i): "<<gsize(i)<<"index(i): "<<index(i)<<", ";
        Eigen::MatrixXd XG = X.middleCols(index(i), gsize(i));
        Eigen::MatrixXd XG_new = XG;
        for (int j=0;j<n;j++) {
          XG_new.row(j) = XG.row(j)*h(j);
        }
        Eigen::MatrixXd XGbar = XG_new.transpose()*XG + 2*this->lambda_level*Eigen::MatrixXd::Identity(gsize(i), gsize(i));
        Eigen::MatrixXd phiG(gsize(i), gsize(i));
        XGbar.sqrt().evalTo(phiG);
        Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(gsize(i), gsize(i)));
        betabar.segment(index(i), gsize(i)) = phiG*beta.segment(index(i), gsize(i));
        dbar.segment(index(i), gsize(i)) = invphiG*d.segment(index(i), gsize(i));
      }
      Eigen::VectorXd temp = betabar + dbar;
      for(int i=0;i<N;i++) {
        bd(i) = (temp.segment(index(i), gsize(i))).squaredNorm()/gsize(i);
      }
      // cout<<"bd"<<bd<<endl;
      // for(int k=0;k<T0;k++) {
      //   bd.maxCoeff(&A[k]);
      //   //cout<<"k: "<<k<<"A[k]"<<A[k]<<", ";
      //   bd(A[k]) = -1.0;
      // }
      // sort(A.begin(), A.end());
      // for(int i=0;i<T0;i++) {
      //   A_out(i) = A[i];
      // }
      max_k(bd, T0, A_out);
  };
};

class GroupPdasPoisson : public Algorithm {
public:
    GroupPdasPoisson(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 3, algorithm_type, max_iter) {
    };

    void primary_model_fit(Eigen::MatrixXd x, Eigen::VectorXd y, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {
      int n = x.rows();
      int p = x.cols();
      Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, p+1);
      X.rightCols(p) = x;
      Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p+1,p+1);
      lambdamat(0,0)=0;
      Eigen::MatrixXd X_trans = X.transpose();
      Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(p+1, n);
      Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p+1);
      beta0.tail(p) = beta;
      beta0(0) = coef0;
      Eigen::VectorXd eta = X*beta0;
      Eigen::VectorXd expeta = eta.array().exp();
      Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
      double loglik0 = 1e5;
      double loglik1;

      for (int j=0;j<50;j++)
      {
        for (int i=0;i<n;i++) {
          temp.col(i) = X_trans.col(i)*expeta(i)*weights(i);
        }
        z = eta+(y-expeta).cwiseQuotient(expeta);
        beta0 = (2*this->lambda_level*lambdamat + temp*X).ldlt().solve(temp*z);
        eta = X*beta0;
        for(int i=0;i<=n-1;i++)
        {
          if(eta(i)<-30.0) eta(i) = -30.0;
          if(eta(i)>30.0) eta(i) = 30.0;
        }
        expeta = eta.array().exp();
        loglik1 = (y.cwiseProduct(eta)-expeta).dot(weights);
        if (abs(loglik0-loglik1)/abs(0.1+loglik0) < 1e-6)  break;
        loglik0 = loglik1;
      }
      for(int i=0;i<p;i++)
          beta(i) = beta0(i+1);
      coef0 = beta0(0);
    }

    void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out) {

      int n=X.rows();
      int p=X.cols();
      Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd bd = Eigen::VectorXd::Zero(N);
      Eigen::VectorXd eta(n);
      Eigen::VectorXd expeta(n);
      Eigen::VectorXd g(n);
      Eigen::VectorXd d(n);
      vector<int> A(T0, -1);

      eta = X*beta+Eigen::VectorXd::Ones(n)*coef0;
      expeta = eta.array().exp();
      g = (y-expeta).cwiseProduct(weights);
      d = X.transpose()*g - 2*this->lambda_level*beta;
      for(int i=0;i<N;i++) {
        Eigen::MatrixXd XG = X.middleCols(index(i), gsize(i));
        Eigen::MatrixXd XG_new = XG;
        for (int j=0;j<n;j++) {
          XG_new.row(j) = XG.row(j)*expeta(j)*weights(j);
        }
        Eigen::MatrixXd XGbar = XG_new.transpose()*XG + 2*this->lambda_level*Eigen::MatrixXd::Identity(gsize(i), gsize(i));
        Eigen::MatrixXd phiG(gsize(i), gsize(i));
        XGbar.sqrt().evalTo(phiG);
        Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(gsize(i), gsize(i)));
        betabar.segment(index(i), gsize(i)) = phiG*beta.segment(index(i), gsize(i));
        dbar.segment(index(i), gsize(i)) = invphiG*d.segment(index(i), gsize(i));
      }
      Eigen::VectorXd temp = betabar+dbar;
      for(int i=0;i<N;i++) {
        bd(i) = (temp.segment(index(i), gsize(i)).squaredNorm())/gsize(i);
      }
      // for(int k=0;k<T0;k++) {
      //   bd.maxCoeff(&A[k]);
      //   bd(A[k])=0;
      // }
      // sort(A.begin(), A.end());
      // for(int i=0;i<T0;i++) {
      //   A_out(i) = A[i];
      // }

      max_k(bd, T0, A_out);
  };

};

class GroupPdasCox : public Algorithm {
public:
    GroupPdasCox(Data &data, int algorithm_type, unsigned int max_iter) : Algorithm(data, 4, algorithm_type, max_iter) {
    };

    void primary_model_fit(Eigen::MatrixXd X, Eigen::VectorXd status, Eigen::VectorXd weights, Eigen::VectorXd &beta, double &coef0)
    {
      // to be ensure
      int n = X.rows();
      int p = X.cols();
      Eigen::MatrixXd lambdamat = Eigen::MatrixXd::Identity(p,p);
      Eigen::VectorXd beta0 = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd beta1 = Eigen::VectorXd::Zero(p);
      Eigen::VectorXd theta(n);
      Eigen::MatrixXd one = (Eigen::MatrixXd::Ones(n, n)).triangularView<Eigen::Upper>();
      Eigen::MatrixXd x_theta(n, p);
      Eigen::VectorXd xij_theta(n);
      Eigen::VectorXd cum_theta(n);
      Eigen::VectorXd g(p);
      Eigen::MatrixXd h(p, p);
      Eigen::VectorXd d(p);
      double loglik0 = 1e5;
      double loglik1;

      double step;
      int m;
      int l;
      for (l=1;l<=30;l++)
      {
        step = 0.5;
        m = 1;
        theta = X*beta0;
        for (int i=0;i<n;i++)
        {
          if (theta(i) > 30) theta(i) = 30;
          else if (theta(i) < -30) theta(i) = -30;
        }
        theta = theta.array().exp();
        //cout<<"theta: "<<theta.head(3)<<endl;
        cum_theta = one*theta;
        x_theta = X.array().colwise()*theta.array();
        x_theta = one*x_theta;
        x_theta = x_theta.array().colwise()/cum_theta.array();
        //cout<<"(X-x_theta).transpose()*(weights.cwiseProduct(status)) : "<<(X-x_theta).transpose()*(weights.cwiseProduct(status)) <<endl;
        g = (X-x_theta).transpose()*(weights.cwiseProduct(status)) + 2 * this->lambda_level * beta0;
        //cout<<"g: "<<g<<endl;
        for (int k1=0;k1<p;k1++)
        {
          for (int k2=k1;k2<p;k2++)
          {
            xij_theta = (theta.cwiseProduct(X.col(k1))).cwiseProduct(X.col(k2));
            for(int j=n-2;j>=0;j--)
            {
              xij_theta(j) = xij_theta(j+1) + xij_theta(j);
            }
            h(k1, k2) = -(xij_theta.cwiseQuotient(cum_theta) - x_theta.col(k1).cwiseProduct(x_theta.col(k2))).dot(weights.cwiseProduct(status));
            h(k2, k1) = h(k1, k2);
          }
        }
        h = h + 2*this->lambda_level * lambdamat;
        d = h.ldlt().solve(g);
        //cout<<"d"<<d<<endl;
        beta1 = beta0-pow(step, m)*d;
        //cout<<"step: "<<step<<", m"<<m<<", pow(step, m)*d: "<<pow(step, m)*d<<endl;
        loglik1 = loglik_cox(X, status, beta1, weights);
        while ((loglik0 > loglik1) && (m<5))
        {
          m = m+1;
          beta1 = beta0-pow(step, m)*d;
          loglik1 = loglik_cox(X, status, beta1, weights);
        }
        if (abs(loglik0-loglik1)/abs(0.1+loglik0) < 1e-5)
        {
          break;
        }
        beta0 = beta1;
        loglik0 = loglik1;
      }
      beta = beta0;
      //cout<<"beta: "<<beta<<endl;
    }

    void get_A(Eigen::MatrixXd X, Eigen::VectorXd y, Eigen::VectorXd beta, double coef0, int T0, Eigen::VectorXd weights,
                       Eigen::VectorXi index, Eigen::VectorXi gsize, int N, Eigen::VectorXi &A_out){
      int n=X.rows();
      int p=X.cols();
      if(this->algorithm_type == 2 || this->algorithm_type == 3)
      {
        Eigen::VectorXd betabar = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd dbar = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
        Eigen::MatrixXd h(n, n);
        Eigen::VectorXd cum_theta(n);
        Eigen::VectorXd cum_theta2(n);
        Eigen::VectorXd cum_theta3(n);
        Eigen::VectorXd theta(n);
        Eigen::VectorXd d(p);
        Eigen::VectorXd g(n);
        vector<int> A(T0, -1);

        theta = X*beta;
        for(int i=0;i<=n-1;i++) {
          if(theta(i)>30.0) theta(i) = 30.0;
          if(theta(i)<-30.0) theta(i) = -30.0;
        }
        theta = weights.array()*theta.array().exp();
        cum_theta(n-1) = theta(n-1);
        for(int k=n-2;k>=0;k--) {
          cum_theta(k) = cum_theta(k+1)+theta(k);
        }
        cum_theta2(0) = (y(0)*weights(0))/cum_theta(0);
        for(int k=1;k<=n-1;k++) {
          cum_theta2(k) = (y(k)*weights(k))/cum_theta(k)+cum_theta2(k-1);
        }
        cum_theta3(0) = (y(0)*weights(0))/pow(cum_theta(0),2);
        for(int k=1;k<=n-1;k++) {
          cum_theta3(k) = (y(k)*weights(k))/pow(cum_theta(k),2)+cum_theta3(k-1);
        }
        h = -cum_theta3.replicate(1, n);
        h = h.cwiseProduct(theta.replicate(1, n));
        h = h.cwiseProduct(theta.replicate(1, n).transpose());
        for(int i=0;i<n;i++) {
          for(int j=i+1;j<n;j++) {
            h(j, i) = h(i, j);
          }
        }
        h.diagonal() = cum_theta2.cwiseProduct(theta) + h.diagonal();
        g = weights.cwiseProduct(y) - cum_theta2.cwiseProduct(theta);
        d = X.transpose()*g - 2*this->lambda_level*beta;
        //cout<<"d: "<<d<<endl;
        for(int i=0;i<N;i++) {
          Eigen::MatrixXd XG =X.middleCols(index(i), gsize(i));
          Eigen::MatrixXd XGbar = XG.transpose()*h*XG + 2*this->lambda_level*Eigen::MatrixXd::Identity(gsize(i), gsize(i));
          Eigen::MatrixXd phiG(gsize(i), gsize(i));
          XGbar.sqrt().evalTo(phiG);
          Eigen::MatrixXd invphiG = phiG.ldlt().solve(Eigen::MatrixXd::Identity(gsize(i), gsize(i)));
          betabar.segment(index(i), gsize(i)) = phiG*beta.segment(index(i), gsize(i));
          dbar.segment(index(i), gsize(i)) = invphiG*d.segment(index(i), gsize(i));
        }
        Eigen::VectorXd temp = betabar+dbar;
          for(int i=0;i<N;i++) {
          bd(i) = (temp.segment(index(i), gsize(i))).squaredNorm()/(gsize(i));
        }
        // cout<<"d[9993]"<<d(9993)<<endl;
        // cout<<"dbar[9993]"<<dbar(9993)<<endl;
        // cout<<"d[9993]"<<d(20906)<<endl;
        // cout<<"dbar[9993]"<<dbar(20906)<<endl;
        // cout<<"theta: "<<theta<<endl;
        // cout<<"cum_theta2: "<<cum_theta2<<endl;
        // cout<<"g: "<<g<<endl;
        // cout<<"lambda_level"<<this->lambda_level<<endl;
        // cout<<"N"<<N<<endl;
        // cout<<"bd: "<<bd<<endl;

        max_k(bd, T0, A_out);
      }
      else if(this->algorithm_type == 1 || this->algorithm_type == 5)
      {
        Eigen::VectorXd l1 = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd l2 = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd cum_theta=Eigen::VectorXd::Zero(n);
        Eigen::VectorXd d = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd bd = Eigen::VectorXd::Zero(p);
        Eigen::MatrixXd xtheta(n,p);
        Eigen::MatrixXd x2theta(n,p);
        vector<int>A(T0);
        Eigen::VectorXd theta=X*beta;
        for(int i=0;i<=n-1;i++) {
            if(theta(i)>30.0) theta(i) = 30.0;
            if(theta(i)<-30.0) theta(i) = -30.0;
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
       //cout<<",   x2theta "<<x2theta;
        xtheta=X.array()-xtheta.array();
        //cout<<",  xtheta "<<xtheta;
        for(unsigned int k=0;k<y.size();k++) {
            if(y[k] == 0.0)
            {
                xtheta.row(k)=Eigen::VectorXd::Zero(p);
                x2theta.row(k)=Eigen::VectorXd::Zero(p);
            }
        }
        l1=-xtheta.adjoint()*weights + 2*this->lambda_level * beta;
        l2=x2theta.adjoint()*weights + 2*this->lambda_level * Eigen::MatrixXd::Ones(p,1);
        d=-l1.cwiseQuotient(l2);
        bd=beta+d;
        bd=bd.cwiseAbs();
        bd=bd.cwiseProduct(l2.cwiseSqrt());
        // for(int k=0;k<=T0-1;k++) {
        //     bd.maxCoeff(&A[k]);
        //     // cout<<"max01 "<<k<<" "<<A[k]<<" bd :"<<bd(A[k])<<" d :"<<d(A[k])<<" beta: "<<beta(A[k])<<endl;
        //     bd(A[k])=0.0;
        // }
        // sort (A.begin(),A.end());
        // for(int i=0;i<T0;i++){
        //   A_out(i) = A[i];
        // }
        max_k(bd, T0, A_out);
        //cout<<endl;
      }
      else
      {
        cout<<"algorithm can not be "<<this->algorithm_type<<endl;
      }
    };
};

#endif //SRC_ALGORITHM_H

