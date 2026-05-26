#ifndef laplacian_solvers_test_hpp
#define laplacian_solvers_test_hpp

#include <iostream>
#include <cmath>
#include <functional>
#include <mpi.h>
#include "laplacian_solvers.hpp"

// Define the function type alias for readability
using funcType = std::function<double(double, double)>;

// Base data configuration setup helper
inline Data_Struct<funcType> create_default_data() {
    Data_Struct<funcType> data;
    data.n = 10;
    data.x1 = 0.0;
    data.x2 = 1.0;
    data.tolerance = 1e-6;
    data.max_iterations = 100000;
    data.alpha = 1.0;
    return data;
}

// =========================================================================
// TEST 1: DIRICHLET SEQUENTIAL
// =========================================================================
inline void run_test_1_dirichlet_sequential(int mpi_rank) {
    if (mpi_rank == 0) {
        std::cout << "\n=== Running Test 1: DIRICHLET SEQUENTIAL ===" << std::endl;
    }

    auto data = create_default_data();
    data.f0 = [](double x, double y) { return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y); };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 0.0; };
    data.f3 = [](double x, double y) { return 0.0; };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::DIRICHLET, ExecutionMode::SEQUENTIAL, funcType> solver(data);
    solver.build_mesh();
    solver.solve();
    solver.print();
}

// =========================================================================
// TEST 2: NEUMANN SEQUENTIAL
// =========================================================================
inline void run_test_2_neumann_sequential(int mpi_rank) {
    if (mpi_rank == 0) {
        std::cout << "\n=== Running Test 2: NEUMANN SEQUENTIAL ===" << std::endl;
    }

    auto data = create_default_data();
    data.f0 = [](double x, double y) { return 2.0 * M_PI * M_PI * std::cos(M_PI * x) * std::cos(M_PI * y); };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 0.0; };
    data.f3 = [](double x, double y) { return 0.0; };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) { return std::cos(M_PI * x) * std::cos(M_PI * y); };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::NEUMANN, ExecutionMode::SEQUENTIAL, funcType> solver(data);
    solver.build_mesh();
    solver.solve();
    solver.print();
}

// =========================================================================
// TEST 3: ROBIN SEQUENTIAL
// =========================================================================
inline void run_test_3_robin_sequential(int mpi_rank) {
    if (mpi_rank == 0) {
        std::cout << "\n=== Running Test 3: ROBIN SEQUENTIAL ===" << std::endl;
    }

    auto data = create_default_data();
    data.f0 = [](double x, double y) { return -2.0 * std::exp(x + y); };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 2.0 * std::exp(1.0 + y); };
    data.f3 = [](double x, double y) { return 2.0 * std::exp(x + 1.0); };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) { return std::exp(x + y); };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::ROBIN, ExecutionMode::SEQUENTIAL, funcType> solver(data);
    solver.build_mesh();
    solver.solve();
    solver.print();
}

// =========================================================================
// TEST 4: DIRICHLET PARALLEL
// =========================================================================
inline void run_test_4_dirichlet_parallel(int mpi_rank) {
    if (mpi_rank == 0) {
        std::cout << "\n=== Running Test 4: DIRICHLET PARALLEL ===" << std::endl;
    }

    auto data = create_default_data();
    data.f0 = [](double x, double y) { return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y); };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 0.0; };
    data.f3 = [](double x, double y) { return 0.0; };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::DIRICHLET, ExecutionMode::PARALLEL, funcType> solver(data);
    solver.build_mesh();
    solver.solve();
    solver.print();
    
    // Puoi anche testare l'esportazione VTK che abbiamo scritto prima:
    // solver.export_to_vtk("test_4_output.vtk");
}

// =========================================================================
// TEST 5: NEUMANN PARALLEL
// =========================================================================
inline void run_test_5_neumann_parallel(int mpi_rank) {
    if (mpi_rank == 0) {
        std::cout << "\n=== Running Test 5: NEUMANN PARALLEL ===" << std::endl;
    }

    auto data = create_default_data();
    data.f0 = [](double x, double y) { return 2.0 * M_PI * M_PI * std::cos(M_PI * x) * std::cos(M_PI * y); };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 0.0; };
    data.f3 = [](double x, double y) { return 0.0; };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) { return std::cos(M_PI * x) * std::cos(M_PI * y); };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::NEUMANN, ExecutionMode::PARALLEL, funcType> solver(data);
    solver.build_mesh();
    solver.solve();
    solver.print();
}

// =========================================================================
// TEST 6: ROBIN PARALLEL
// =========================================================================
inline void run_test_6_robin_parallel(int mpi_rank) {
    if (mpi_rank == 0) {
        std::cout << "\n=== Running Test 6: ROBIN PARALLEL ===" << std::endl;
    }

    auto data = create_default_data();
    data.f0 = [](double x, double y) { return -2.0 * std::exp(x + y); };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 2.0 * std::exp(1.0 + y); };
    data.f3 = [](double x, double y) { return 2.0 * std::exp(x + 1.0); };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) { return std::exp(x + y); };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::ROBIN, ExecutionMode::PARALLEL, funcType> solver(data);
    solver.build_mesh();
    solver.solve();
    solver.print();
}

#endif // TESTS_HPP