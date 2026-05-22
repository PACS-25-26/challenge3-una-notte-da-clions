#ifndef LAPLACIAN_SOLVERS_HPP
#define LAPLACIAN_SOLVERS_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <Eigen/Dense>

enum class SolverType {
    JACOBI,
    SCHWARZ
};  

enum class BoundaryCondition {
    DIRICHLET,
    NEUMANN,
    ROBIN
};

enum class ExecutionMode {
    SEQUENTIAL,
    PARALLEL
};

template <typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
struct Data_Struct {
    unsigned n; 
    double x1, x2; 
    Func1 f1;  
    Func2 f2; 
    Func3 f3;
    Func4 f4;
    u_ex u_exact;
};

struct Result_Struct {
    Eigen::MatrixXd MESH;
    Eigen::MatrixXd u_h; 
    unsigned iterations;

};

struct Convergence_Struct {
    std::vector<double> errors_L2_serial;
    std::vector<double> errors_L2_parallel; 
    std::vector<unsigned> n;
};

namespace laplacian_solvers {
    template<SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode>
    class LaplacianSolver {
        private:
            Data_Struct data;
        public:
            LaplacianSolver(const Data_Struct& data) : data(data){};
            Result_Struct solve();
            Convergence_Struct convergence_test(); 
    };



} 

#include "laplacian_solvers_implementation.hpp"

#endif 