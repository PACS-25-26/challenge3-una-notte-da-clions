#ifndef LAPLACIAN_SOLVERS_TEST_1_HPP
#define LAPLACIAN_SOLVERS_TEST_1_HPP

#include <iostream>
#include <cmath>
#include <functional>
#include <mpi.h>
#include "laplacian_solvers.hpp"
#include "laplacian_convergence_test.hpp"
#include "laplacian_solvers_postprocessing.hpp"

namespace laplacian_solvers {
    // Define the function type alias for readability
    using funcType = std::function<double(double, double)>;

    // Base data configuration setup helper
    inline Data_Struct<funcType> create_default_data() {
        Data_Struct<funcType> data;
        data.n = 8;
        data.x1 = 0.0;
        data.x2 = 1.0;
        data.tolerance = 1e-6;
        data.max_iterations = 100000;
        data.alpha = 1.0;
        return data;
    }

    // =========================================================================
    // TEST 1: DIRICHLET SEQUENTIAL (Jacobi)
    // =========================================================================
    inline void run_test_1_dirichlet_sequential(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 1: DIRICHLET SEQUENTIAL (Jacobi) ===" << std::endl;
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
        solver.export_to_vtk("test_1_dirichlet_sequential.vtk");
    }

    // =========================================================================
    // TEST 2: NEUMANN SEQUENTIAL (Jacobi)
    // =========================================================================
    inline void run_test_2_neumann_sequential(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 2: NEUMANN SEQUENTIAL (Jacobi) ===" << std::endl;
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
        solver.export_to_vtk("test_2_neumann_sequential.vtk");
    }

    // =========================================================================
    // TEST 3: ROBIN SEQUENTIAL (Jacobi)
    // =========================================================================
    inline void run_test_3_robin_sequential(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 3: ROBIN SEQUENTIAL (Jacobi) ===" << std::endl;
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
        solver.export_to_vtk("test_3_robin_sequential.vtk");
    }

    // =========================================================================
    // TEST 4: DIRICHLET PARALLEL (Jacobi)
    // =========================================================================
    inline void run_test_4_dirichlet_parallel(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 4: DIRICHLET PARALLEL (Jacobi) ===" << std::endl;
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
        solver.export_to_vtk("test_4_dirichlet_parallel_jacobi.vtk");
    }

    // =========================================================================
    // TEST 5: NEUMANN PARALLEL (Jacobi)
    // =========================================================================
    inline void run_test_5_neumann_parallel(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 5: NEUMANN PARALLEL (Jacobi) ===" << std::endl;
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
        solver.export_to_vtk("test_5_neumann_parallel_jacobi.vtk");
    }

    // =========================================================================
    // TEST 6: ROBIN PARALLEL (Jacobi)
    // =========================================================================
    inline void run_test_6_robin_parallel(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 6: ROBIN PARALLEL (Jacobi) ===" << std::endl;
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
        solver.export_to_vtk("test_6_robin_parallel_jacobi.vtk");
    }

    // =========================================================================
    // TEST 7: DIRICHLET PARALLEL (Schwarz)
    // =========================================================================
    inline void run_test_7_dirichlet_parallel_schwarz(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 7: DIRICHLET PARALLEL (Schwarz) ===" << std::endl;
        }

        auto data = create_default_data();
        data.f0 = [](double x, double y) { return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y); };
        data.f1 = [](double x, double y) { return 0.0; };
        data.f2 = [](double x, double y) { return 0.0; };
        data.f3 = [](double x, double y) { return 0.0; };
        data.f4 = [](double x, double y) { return 0.0; };
        data.u_exact_lambda = [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); };

        laplacian_solvers::Laplacian_Solver<SolverType::SCHWARZ, BoundaryCondition::DIRICHLET, ExecutionMode::PARALLEL, funcType> solver(data);
        solver.build_mesh();
        solver.solve();
        solver.print();
        solver.export_to_vtk("test_7_dirichlet_parallel_schwarz.vtk");
    }

        // =========================================================================
        // TEST 8: NEUMANN PARALLEL (Schwarz)
        // =========================================================================
    inline void run_test_8_neumann_parallel_schwarz(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 8: NEUMANN PARALLEL (Schwarz) ===" << std::endl;
        }

        auto data = create_default_data();
        data.f0 = [](double x, double y) { return 2.0 * M_PI * M_PI * std::cos(M_PI * x) * std::cos(M_PI * y); };
        data.f1 = [](double x, double y) { return 0.0; };
        data.f2 = [](double x, double y) { return 0.0; };
        data.f3 = [](double x, double y) { return 0.0; };
        data.f4 = [](double x, double y) { return 0.0; };
        data.u_exact_lambda = [](double x, double y) { return std::cos(M_PI * x) * std::cos(M_PI * y); };

        laplacian_solvers::Laplacian_Solver<SolverType::SCHWARZ, BoundaryCondition::NEUMANN, ExecutionMode::PARALLEL, funcType> solver(data);
        solver.build_mesh();
        solver.solve();
        solver.print();
        solver.export_to_vtk("test_8_neumann_parallel_schwarz.vtk");
    }

    // =========================================================================
    // TEST 9: ROBIN PARALLEL (Schwarz)
    // =========================================================================
    inline void run_test_9_robin_parallel_schwarz(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 9: ROBIN PARALLEL (Schwarz) ===" << std::endl;
        }

        auto data = create_default_data();
        data.f0 = [](double x, double y) { return -2.0 * std::exp(x + y); };
        data.f1 = [](double x, double y) { return 0.0; };
        data.f2 = [](double x, double y) { return 2.0 * std::exp(1.0 + y); };
        data.f3 = [](double x, double y) { return 2.0 * std::exp(x + 1.0); };
        data.f4 = [](double x, double y) { return 0.0; };
        data.u_exact_lambda = [](double x, double y) { return std::exp(x + y); };

        laplacian_solvers::Laplacian_Solver<SolverType::SCHWARZ, BoundaryCondition::ROBIN, ExecutionMode::PARALLEL, funcType> solver(data);
        solver.build_mesh();
        solver.solve();
        solver.print();
        solver.export_to_vtk("test_9_robin_parallel_schwarz.vtk");
    }

    // =========================================================================
    // TEST 10: DIRICHLET SEQUENTIAL vs PARALLEL (Jacobi)
    // =========================================================================
    inline void run_test_10_dirichlet_sequential_vs_parallel_jacobi(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 10: DIRICHLET SEQUENTIAL vs PARALLEL (Jacobi) ===" << std::endl;
        }

        auto data = create_default_data();
        data.f0 = [](double x, double y) { return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y); };
        data.f1 = [](double x, double y) { return 0.0; };
        data.f2 = [](double x, double y) { return 0.0; };
        data.f3 = [](double x, double y) { return 0.0; };
        data.f4 = [](double x, double y) { return 0.0; };
        data.u_exact_lambda = [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); };

        laplacian_solvers::Convergence_Test<SolverType::JACOBI, BoundaryCondition::DIRICHLET, funcType> convergence_test(data);
        auto convergence_results = convergence_test.run_convergence_test();
        convergence_test.print_convergence_results();
    }

    // =========================================================================
    // TEST 11: NEUMANN SEQUENTIAL vs PARALLEL (Jacobi)
    // =========================================================================
    inline void run_test_11_neumann_sequential_vs_parallel_jacobi(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 11: NEUMANN SEQUENTIAL vs PARALLEL (Jacobi) ===" << std::endl;
        }

        auto data = create_default_data();
        data.f0 = [](double x, double y) { return 2.0 * M_PI * M_PI * std::cos(M_PI * x) * std::cos(M_PI * y); };
        data.f1 = [](double x, double y) { return 0.0; };
        data.f2 = [](double x, double y) { return 0.0; };
        data.f3 = [](double x, double y) { return 0.0; };
        data.f4 = [](double x, double y) { return 0.0; };
        data.u_exact_lambda = [](double x, double y) { return std::cos(M_PI * x) * std::cos(M_PI * y); };

        laplacian_solvers::Convergence_Test<SolverType::JACOBI, BoundaryCondition::NEUMANN, funcType> convergence_test(data);
        auto convergence_results = convergence_test.run_convergence_test();
        convergence_test.print_convergence_results();
    }

    // =========================================================================
    // TEST 12: ROBIN SEQUENTIAL vs PARALLEL (Jacobi)
    // =========================================================================
    inline void run_test_12_robin_sequential_vs_parallel_jacobi(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 12: ROBIN SEQUENTIAL vs PARALLEL (Jacobi) ===" << std::endl;
        }

        auto data = create_default_data();
        data.f0 = [](double x, double y) { return -2.0 * std::exp(x + y); };
        data.f1 = [](double x, double y) { return 0.0; };
        data.f2 = [](double x, double y) { return 2.0 * std::exp(1.0 + y); };
        data.f3 = [](double x, double y) { return 2.0 * std::exp(x + 1.0); };
        data.f4 = [](double x, double y) { return 0.0; };
        data.u_exact_lambda = [](double x, double y) { return std::exp(x + y); };

        laplacian_solvers::Convergence_Test<SolverType::JACOBI, BoundaryCondition::ROBIN, funcType> convergence_test(data);
        auto convergence_results = convergence_test.run_convergence_test();
        convergence_test.print_convergence_results();
    }

    // =========================================================================
    // TEST 13: DIRICHLET SEQUENTIAL vs PARALLEL (Schwarz)
    // =========================================================================
    inline void run_test_13_dirichlet_sequential_vs_parallel_schwarz(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 13: DIRICHLET SEQUENTIAL vs PARALLEL (Schwarz) ===" << std::endl;
        }

        auto data = create_default_data();
        data.f0 = [](double x, double y) { return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y); };
        data.f1 = [](double x, double y) { return 0.0; };
        data.f2 = [](double x, double y) { return 0.0; };
        data.f3 = [](double x, double y) { return 0.0; };
        data.f4 = [](double x, double y) { return 0.0; };
        data.u_exact_lambda = [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); };

        laplacian_solvers::Convergence_Test<SolverType::SCHWARZ, BoundaryCondition::DIRICHLET, funcType> convergence_test(data);
        auto convergence_results = convergence_test.run_convergence_test();
        convergence_test.print_convergence_results();
    }

    // =========================================================================
    // TEST 14: NEUMANN SEQUENTIAL vs PARALLEL (Schwarz)
    // =========================================================================
    inline void run_test_14_neumann_sequential_vs_parallel_schwarz(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 14: NEUMANN SEQUENTIAL vs PARALLEL (Schwarz) ===" << std::endl;
        }

        auto data = create_default_data();
        data.f0 = [](double x, double y) { return 2.0 * M_PI * M_PI * std::cos(M_PI * x) * std::cos(M_PI * y); };
        data.f1 = [](double x, double y) { return 0.0; };
        data.f2 = [](double x, double y) { return 0.0; };
        data.f3 = [](double x, double y) { return 0.0; };
        data.f4 = [](double x, double y) { return 0.0; };
        data.u_exact_lambda = [](double x, double y) { return std::cos(M_PI * x) * std::cos(M_PI * y); };

        laplacian_solvers::Convergence_Test<SolverType::SCHWARZ, BoundaryCondition::NEUMANN, funcType> convergence_test(data);
        auto convergence_results = convergence_test.run_convergence_test();
        convergence_test.print_convergence_results();
    }

    // =========================================================================
    // TEST 15: ROBIN SEQUENTIAL vs PARALLEL (Schwarz)
    // =========================================================================
    inline void run_test_15_robin_sequential_vs_parallel_schwarz(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 15: ROBIN SEQUENTIAL vs PARALLEL (Schwarz) ===" << std::endl;
        }

        auto data = create_default_data();
        data.f0 = [](double x, double y) { return -2.0 * std::exp(x + y); };
        data.f1 = [](double x, double y) { return 0.0; };
        data.f2 = [](double x, double y) { return 2.0 * std::exp(1.0 + y); };
        data.f3 = [](double x, double y) { return 2.0 * std::exp(x + 1.0); };
        data.f4 = [](double x, double y) { return 0.0; };
        data.u_exact_lambda = [](double x, double y) { return std::exp(x + y); };

        laplacian_solvers::Convergence_Test<SolverType::SCHWARZ, BoundaryCondition::ROBIN, funcType> convergence_test(data);
        auto convergence_results = convergence_test.run_convergence_test();
        convergence_test.print_convergence_results();
    }
    
}

#endif // TESTS_HPP