#include <Eigen/Eigen>
#include <unsupported//Eigen/MatrixFunctions>
#include <algorithm>
#include <vector>
using namespace std;

Eigen::VectorXi find_ind(Eigen::VectorXi& L, Eigen::VectorXi& index, Eigen::VectorXi& gsize, int p, int N)
{
  if (L.size() == N) {
    return Eigen::VectorXi::LinSpaced(p, 0, p-1);
  }
  else 
  {
    int mark = 0;
    Eigen::VectorXi ind = Eigen::VectorXi::Zero(p);
    for (int i=0;i<L.size();i++) {
        ind.segment(mark, gsize(L[i])) = Eigen::VectorXi::LinSpaced(gsize(L[i]), index(L[i]), index(L[i])+gsize(L[i])-1);
        mark = mark + gsize(L[i]);
    }
    return ind.head(mark);
  }
}

Eigen::MatrixXd X_seg(Eigen::MatrixXd& X, int n, Eigen::VectorXi& ind) {
  Eigen::MatrixXd X_new(n, ind.size());
  for (int k=0;k<ind.size();k++) {
    X_new.col(k) = X.col(ind[k]);
  }
  return X_new;
}


std::vector<Eigen::MatrixXd> Phi(Eigen::MatrixXd& X, Eigen::VectorXi index, Eigen::VectorXi gsize, int n, int p, int N){
  std::vector<Eigen::MatrixXd> Phi(N);
  for (int i=0;i<N;i++) {
    Eigen::MatrixXd X_ind = X.block(0, index(i), n, gsize(i));
    Eigen::MatrixXd XtX = (X_ind.transpose()*X_ind)/double(n);
    XtX.sqrt().evalTo(Phi[i]);
  }
  return Phi;
}

std::vector<Eigen::MatrixXd> invPhi(std::vector<Eigen::MatrixXd>& Phi, int N){
  std::vector<Eigen::MatrixXd> invPhi(N);
  int row;
  for (int i=0;i<N;i++){
    row = (Phi[i]).rows();
    invPhi[i] = (Phi[i]).ldlt().solve(Eigen::MatrixXd::Identity(row, row));
  }
  return invPhi;
}
