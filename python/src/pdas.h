#ifndef pdas_H
#define pdas_H
#include <Eigen/Eigen>
#include <vector>

using namespace std;
Eigen::VectorXi find_ind(Eigen::VectorXi& L, Eigen::VectorXi& index, Eigen::VectorXi& gsize, int p, int N);
Eigen::MatrixXd X_seg(Eigen::MatrixXd& X, int n, Eigen::VectorXi& ind);
std::vector<Eigen::MatrixXd> Phi(Eigen::MatrixXd& X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N, double lambda);
std::vector<Eigen::MatrixXd> invPhi(std::vector<Eigen::MatrixXd>& Phi, int N);

#endif
