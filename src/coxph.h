#ifndef coxph_H
#define coxph_H

Eigen::VectorXd coxPH(Eigen::MatrixXd& X, Eigen::VectorXd& status, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, int max_steps = 10, double ita = 0.5, double err = 10e-6);

#endif
