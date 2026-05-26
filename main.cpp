#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <functional>
#include <cmath>

#include "laplacian_solvers.hpp"


int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    using funcType = std::function<double(double, double)>;

    // Initial test
    Data_Struct<funcType> data;

    data.n = 10;
    data.x1 = 0.;
    data.x2 = 1.;
    data.f0 = [](double x, double y) { return 8*M_PI*M_PI*std::sin(2*M_PI*x)*std::sin(2*M_PI*y); }; // Source term
    data.f1 = [](double x, double y) { return std::sin(2*M_PI*x)*std::sin(2*M_PI*y); }; // BC bottom edge
    data.f2 = [](double x, double y) { return std::sin(2*M_PI*x)*std::sin(2*M_PI*y); }; // BC right edge
    data.f3 = [](double x, double y) { return std::sin(2*M_PI*x)*std::sin(2*M_PI*y); }; // BC top edge
    data.f4 = [](double x, double y) { return std::sin(2*M_PI*x)*std::sin(2*M_PI*y); }; // BC left edge
    data.u_exact_lambda = [](double x, double y) { return std::sin(2*M_PI*x)*std::sin(2*M_PI*y); }; // Exact solution
    data.tolerance = 1e-6;
    data.max_iterations = 10000;
    data.alpha = 1.0; // Parameter for Robin boundary condition

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::DIRICHLET, ExecutionMode::SEQUENTIAL, funcType> solver(data);
    solver.build_mesh();

    auto result = solver.solve();

    solver.print();

    MPI_Finalize();
    return 0;
}