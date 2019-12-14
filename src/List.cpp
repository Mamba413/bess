#include <iostream>
#include <typeinfo>
#include <Eigen/Eigen> 
#include <vector>
#include "List.h"

using namespace std;
//using namespace Eigen;

//template<typename T> void List::add(string name, T value)
//{
//	double temp_double;
//	Eigen::MatrixXd temp_MatrixXd;
//	Eigen::VectorXd temp_VectorXd;
//	cout<<typeid(temp_double).name()<<endl;
//	cout<<typeid(temp_MatrixXd).name()<<endl;
//	cout<<typeid(temp_VectorXd).name()<<endl;
////	cout<<typeid(typeid(value).name()).name()<<endl;
////	bool a;
////	a = typeid(value).name() == "d";
////	cout<<a<<endl;
//	if(typeid(value).name() == typeid(temp_double).name())
//	{
//		cout<<"value in double add"<<endl;
//		vector_double.push_back(value);
//		vector_double_name.push_back(name);
//	}
////	else if(typeid(value).name() == typeid(temp_MatrixXd).name())
////	{
////		cout<<"value in MatrixXd add"<<endl;
////		vector_MatrixXd.push_back(value);
////		vector_MatrixXd_name.push_back(name);	
////	}
////	else if(typeid(value).name() == typeid(temp_VectorXd).name())
////	{
////		cout<<"value in VectorXd add"<<endl;
////		vector_VectorXd.push_back(value);
////		vector_VectorXd_name.push_back(name);	
////	}
//}
//
//template<typename T> T List::get_value_by_name(string name, T type)
//{
//	double temp_double;
//	Eigen::MatrixXd temp_MatrixXd;
//	Eigen::VectorXd temp_VectorXd;
//	T value;
//	int i;
//	if(typeid(value).name() == typeid(temp_double).name())
//	{
//		for(i=0;i<vector_double_name.size();i++)
//		{
//			cout<<"value in get double"<<endl;
//			if(vector_double_name[i] == name)
//			{
//				return vector_double[i];
//			}
//		}
//	}
//	else if(typeid(value).name() == typeid(temp_MatrixXd).name())
//	{
//		for(i=0;i<vector_MatrixXd_name.size();i++)
//		{
//			cout<<"value in get MatrixXd"<<endl;
//			if(vector_MatrixXd_name[i] == name)
//			{
//				return vector_MatrixXd[i];
//			}
//		}
//	}
//	else if(typeid(value).name() == typeid(temp_VectorXd).name())
//	{
//		for(i=0;i<vector_VectorXd_name.size();i++)
//		{
//			cout<<"value in get VectorXd"<<endl;
//			if(vector_VectorXd_name[i] == name)
//			{
//				return vector_VectorXd[i];
//			}
//		}
//	}
//}

void List::add(string name, double value)
{
	cout<<"value in double add"<<endl;
	vector_double.push_back(value);
	vector_double_name.push_back(name);
}

void List::add(string name, MatrixXd value)
{
	cout<<"value in MatrixXd add"<<endl;
	vector_MatrixXd.push_back(value);
	vector_MatrixXd_name.push_back(name);	
}

void List::add(string name, VectorXd value)
{
	cout<<"value in VectorXd add"<<endl;
	vector_VectorXd.push_back(value);
	vector_VectorXd_name.push_back(name);	
}

void List::add(string name, VectorXi value)
{
	cout<<"value in VectorXi add"<<endl;
	vector_VectorXi.push_back(value);
	vector_VectorXi_name.push_back(name);
}


double List::get_value_by_name(string name, double type)
{
	std::size_t i;
	for(i=0;i<vector_double_name.size();i++)
	{
		cout<<"value in get double"<<endl;
		if(vector_double_name[i] == name)
		{
			return vector_double[i];
		}
	}
}

MatrixXd List::get_value_by_name(string name, MatrixXd type)
{
	std::size_t i;
	for(i=0;i<vector_MatrixXd_name.size();i++)
	{
		cout<<"value in get MatrixXd"<<endl;
		if(vector_MatrixXd_name[i] == name)
		{
			return vector_MatrixXd[i];
		}
	}
}

VectorXd List::get_value_by_name(string name, VectorXd type)
{
	std::size_t i;
	for(i=0;i<vector_VectorXd_name.size();i++)
	{
		cout<<"value in get VectorXd"<<endl;
		if(vector_VectorXd_name[i] == name)
		{
			return vector_VectorXd[i];
		}
	}
}

VectorXi List::get_value_by_name(string name, VectorXi type)
{
	std::size_t i;
	for(i=0;i<vector_VectorXi_name.size();i++)
	{
		cout<<"value in get VectorXi"<<endl;
		if(vector_VectorXi_name[i] == name)
		{
			return vector_VectorXi[i];
		}
	}
}
//
//int main()
//{
//	int i;
//	List mylist;
//
//	mylist.add("alpha", 0.5);
//	double alpha;
//	alpha = mylist.get_value_by_name("alpha", alpha);
//	cout<<alpha<<endl;
//
//	int m=2;
//	int n=2;
//	Eigen::MatrixXd mymatrix(m,n);
//	mymatrix<<2.3,4.0,5.0,6.1;
//	mylist.add("beta", mymatrix);
//	Eigen::MatrixXd beta;
//	beta = mylist.get_value_by_name("beta", beta);
//		cout<<beta(0, 0)<<endl;
//
//	int len=5;
//	Eigen::VectorXd myvector(len);
//	Eigen::VectorXd gamma;
//	myvector<<5.2,6.3,1.0,5.8,6.5;
//	mylist.add("gamma", myvector);
//	gamma = mylist.get_value_by_name("gamma", gamma);
//
//	cout<<gamma[1]<<endl;
//}
