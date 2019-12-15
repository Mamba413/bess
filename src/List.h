#ifndef List_H
#define List_H
 
#include <iostream>
#include <Eigen/Eigen>
#include <vector>
using namespace std;
using namespace Eigen;

class List
{
    public:
        List(){};
        ~List(){};
//        template<typename T> void add(string name, T value);
//		template<typename T> T get_value_by_name(string name, T type);
		void add(string name, double value);
		double get_value_by_name(string name, double type);
		void add(string name, MatrixXd value);
		MatrixXd get_value_by_name(string name, MatrixXd type);
		void add(string name, VectorXd value);
		VectorXd get_value_by_name(string name, VectorXd type);
		void add(string name, VectorXi value);
		VectorXi get_value_by_name(string name, VectorXi type);
    private:
    	vector<double> vector_double;
    	vector<string> vector_double_name;
    	vector<Eigen::MatrixXd> vector_MatrixXd;
    	vector<string> vector_MatrixXd_name;
    	vector<Eigen::VectorXd> vector_VectorXd;
    	vector<string> vector_VectorXd_name;
    	vector<Eigen::VectorXi> vector_VectorXi;
    	vector<string> vector_VectorXi_name;
};

#endif //List_H
