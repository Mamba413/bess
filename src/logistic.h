#ifndef logistic_H
#define logistic_H

Eigen::VectorXd logistic(Eigen::MatrixXd& X, Eigen::VectorXd& y, Eigen::VectorXd& beta0, Eigen::VectorXd& weights, int max_steps = 20, double err = 10e-7);

#endif
