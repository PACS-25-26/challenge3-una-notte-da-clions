#ifndef LAPLACIAN_SOLVERS_TEST_2_HPP
#define LAPLACIAN_SOLVERS_TEST_2_HPP

#include "laplacian_solvers_test_1.hpp"

/**
 * @file laplacian_solvers_test_2.hpp
 * @brief Extended integration test suite evaluating non-homogeneous boundary test cases.
 * * Contains advanced validation tests using multi-dimensional polynomial and 
 * trigonometric exact solutions to verify convergence under complex boundary configurations.
 */

namespace laplacian_solvers {
    
    // =========================================================================
    // TEST 16: DIRICHLET SEQUENTIAL (Jacobi) - Non-homogeneous Polynomial
    // =========================================================================
    /**
     * @brief Test 16: Sequential Jacobi solver with a non-homogeneous polynomial solution under Dirichlet boundaries.
     * * Validates analytical solution: \f$ u(x,y) = x^2y + xy^2 + x + y + 1 \f$
     * @param mpi_rank Rank of the calling MPI process.
     */
    inline void run_test_16_dirichlet_sequential(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 16: DIRICHLET SEQUENTIAL (Jacobi) ===" << std::endl;
        }

        auto data = create_default_data();
        
        // Exact solution: u(x,y) = x^2*y + x*y^2 + x + y + 1
        data.u_exact_lambda = [](double x, double y) { return x*x*y + x*y*y + x + y + 1.0; };
        
        // Forcing term: f0 = -Laplacian(u) = -2(x+y)
        data.f0 = [](double x, double y) { return -2.0 * (x + y); };
        
        // Dirichlet boundaries: u evaluated exactly at the edges
        data.f1 = [](double x, double y) { return x + 1.0; };                           // Bottom (y=0)
        data.f2 = [](double x, double y) { return y*y + 2.0*y + 2.0; };                 // Right  (x=1)
        data.f3 = [](double x, double y) { return x*x + 2.0*x + 2.0; };                 // Top    (y=1)
        data.f4 = [](double x, double y) { return y + 1.0; };                           // Left   (x=0)

        laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::DIRICHLET, ExecutionMode::SEQUENTIAL, funcType> solver(data);
        solver.build_mesh();
        solver.solve();
        solver.print();
        solver.export_to_vtk("test_16_dirichlet_sequential.vtk");
    }

    // =========================================================================
    // TEST 17: NEUMANN SEQUENTIAL (Jacobi) - Non-homogeneous Polynomial
    // =========================================================================
    /**
     * @brief Test 17: Sequential Jacobi solver with a non-homogeneous polynomial solution under Neumann boundaries.
     * * Verifies normal derivative flux mappings: \f$ \frac{\partial u}{\partial n} = \nabla u \cdot \mathbf{n} \f$
     * @param mpi_rank Rank of the calling MPI process.
     */
    inline void run_test_17_neumann_sequential(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 17: NEUMANN SEQUENTIAL (Jacobi) ===" << std::endl;
        }

        auto data = create_default_data();
        
        // Exact solution: u(x,y) = x^2*y + x*y^2 + x + y + 1
        data.u_exact_lambda = [](double x, double y) { return x*x*y + x*y*y + x + y + 1.0; };
        
        // Forcing term: f0 = -Laplacian(u) = -2(x+y)
        data.f0 = [](double x, double y) { return -2.0 * (x + y); };
        
        // Neumann boundaries: du/dn = grad(u) * n
        data.f1 = [](double x, double y) { return -(x*x + 1.0); };                      // Bottom (n =  0, -1) -> -uy
        data.f2 = [](double x, double y) { return y*y + 2.0*y + 1.0; };                 // Right  (n =  1,  0) ->  ux
        data.f3 = [](double x, double y) { return x*x + 2.0*x + 1.0; };                 // Top    (n =  0,  1) ->  uy
        data.f4 = [](double x, double y) { return -(y*y + 1.0); };                      // Left   (n = -1,  0) -> -ux

        laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::NEUMANN, ExecutionMode::SEQUENTIAL, funcType> solver(data);
        solver.build_mesh();
        solver.solve();
        solver.print();
        solver.export_to_vtk("test_17_neumann_sequential.vtk");
    }
    
    // =========================================================================
    // TEST 18: ROBIN SEQUENTIAL (Jacobi) - Non-homogeneous Trigonometric
    // =========================================================================
    /**
     * @brief Test 18: Sequential Jacobi solver with a non-homogeneous trigonometric solution under Robin boundaries.
     * * Evaluates linear combination constraints: \f$ u + \frac{\partial u}{\partial n} = g \f$
     * @param mpi_rank Rank of the calling MPI process.
     */
    inline void run_test_18_robin_sequential(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 18: ROBIN SEQUENTIAL (Jacobi) ===" << std::endl;
        }

        auto data = create_default_data();
        
        // Exact solution: u(x,y) = sin(x+y) + 2x + 3y
        data.u_exact_lambda = [](double x, double y) { return std::sin(x + y) + 2.0*x + 3.0*y; };
        
        // Forcing term: f0 = -Laplacian(u) = 2*sin(x+y)
        data.f0 = [](double x, double y) { return 2.0 * std::sin(x + y); };
        
        // Robin boundaries: u + du/dn = g
        // Note: ux = cos(x+y) + 2, uy = cos(x+y) + 3
        data.f1 = [](double x, double y) { return std::sin(x) - std::cos(x) + 2.0*x - 3.0; };            // Bottom: u - uy
        data.f2 = [](double x, double y) { return std::sin(1.0+y) + std::cos(1.0+y) + 3.0*y + 4.0; };    // Right:  u + ux
        data.f3 = [](double x, double y) { return std::sin(x+1.0) + std::cos(x+1.0) + 2.0*x + 6.0; };    // Top:    u + uy
        data.f4 = [](double x, double y) { return std::sin(y) - std::cos(y) + 3.0*y - 2.0; };            // Left:   u - ux

        laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::ROBIN, ExecutionMode::SEQUENTIAL, funcType> solver(data);
        solver.build_mesh();
        solver.solve();
        solver.print();
        solver.export_to_vtk("test_18_robin_sequential.vtk");
    }

    // =========================================================================
    // TEST 19: DIRICHLET PARALLEL (Jacobi) - Non-homogeneous Polynomial
    // =========================================================================
    /**
     * @brief Test 19: Parallel MPI-distributed Jacobi solver with a non-homogeneous polynomial solution under Dirichlet boundaries.
     * @param mpi_rank Rank of the calling MPI process.
     */
    inline void run_test_19_dirichlet_parallel(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 19: DIRICHLET PARALLEL (Jacobi) ===" << std::endl;
        }

        auto data = create_default_data();
        
        // Exact solution: u(x,y) = x^2*y + x*y^2 + x + y + 1
        data.u_exact_lambda = [](double x, double y) { return x*x*y + x*y*y + x + y + 1.0; };
        
        // Forcing term: f0 = -Laplacian(u) = -2(x+y)
        data.f0 = [](double x, double y) { return -2.0 * (x + y); };
        
        // Dirichlet boundaries: u evaluated exactly at the edges
        data.f1 = [](double x, double y) { return x + 1.0; };                           // Bottom (y=0)
        data.f2 = [](double x, double y) { return y*y + 2.0*y + 2.0; };                 // Right  (x=1)
        data.f3 = [](double x, double y) { return x*x + 2.0*x + 2.0; };                 // Top    (y=1)
        data.f4 = [](double x, double y) { return y + 1.0; };                           // Left   (x=0)

        laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::DIRICHLET, ExecutionMode::PARALLEL, funcType> solver(data);
        solver.build_mesh();
        solver.solve();
        solver.print();
        solver.export_to_vtk("test_19_dirichlet_parallel.vtk");
    }

    // =========================================================================
    // TEST 20: NEUMANN PARALLEL (Jacobi) - Non-homogeneous Polynomial
    // =========================================================================
    /**
     * @brief Test 20: Parallel MPI-distributed Jacobi solver with a non-homogeneous polynomial solution under Neumann boundaries.
     * @param mpi_rank Rank of the calling MPI process.
     */
    inline void run_test_20_neumann_parallel(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 20: NEUMANN PARALLEL (Jacobi) ===" << std::endl;
        }

        auto data = create_default_data();
        
        // Exact solution: u(x,y) = x^2*y + x*y^2 + x + y + 1
        data.u_exact_lambda = [](double x, double y) { return x*x*y + x*y*y + x + y + 1.0; };
        
        // Forcing term: f0 = -Laplacian(u) = -2(x+y)
        data.f0 = [](double x, double y) { return -2.0 * (x + y); };
        
        // Neumann boundaries: du/dn = grad(u) * n
        data.f1 = [](double x, double y) { return -(x*x + 1.0); };                      // Bottom (n =  0, -1) -> -uy
        data.f2 = [](double x, double y) { return y*y + 2.0*y + 1.0; };                 // Right  (n =  1,  0) ->  ux
        data.f3 = [](double x, double y) { return x*x + 2.0*x + 1.0; };                 // Top    (n =  0,  1) ->  uy
        data.f4 = [](double x, double y) { return -(y*y + 1.0); };                      // Left   (n = -1,  0) -> -ux

        laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::NEUMANN, ExecutionMode::PARALLEL, funcType> solver(data);
        solver.build_mesh();
        solver.solve();
        solver.print();
        solver.export_to_vtk("test_20_neumann_parallel.vtk");
    }
    
    // =========================================================================
    // TEST 21: ROBIN PARALLEL (Jacobi) - Non-homogeneous Trigonometric
    // =========================================================================
    /**
     * @brief Test 21: Parallel MPI-distributed Jacobi solver with a non-homogeneous trigonometric solution under Robin boundaries.
     * @param mpi_rank Rank of the calling MPI process.
     */
    inline void run_test_21_robin_parallel(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 21: ROBIN PARALLEL (Jacobi) ===" << std::endl;
        }

        auto data = create_default_data();
        
        // Exact solution: u(x,y) = sin(x+y) + 2x + 3y
        data.u_exact_lambda = [](double x, double y) { return std::sin(x + y) + 2.0*x + 3.0*y; };
        
        // Forcing term: f0 = -Laplacian(u) = 2*sin(x+y)
        data.f0 = [](double x, double y) { return 2.0 * std::sin(x + y); };
        
        // Robin boundaries: u + du/dn = g
        // Note: ux = cos(x+y) + 2, uy = cos(x+y) + 3
        data.f1 = [](double x, double y) { return std::sin(x) - std::cos(x) + 2.0*x - 3.0; };            // Bottom: u - uy
        data.f2 = [](double x, double y) { return std::sin(1.0+y) + std::cos(1.0+y) + 3.0*y + 4.0; };    // Right:  u + ux
        data.f3 = [](double x, double y) { return std::sin(x+1.0) + std::cos(x+1.0) + 2.0*x + 6.0; };    // Top:    u + uy
        data.f4 = [](double x, double y) { return std::sin(y) - std::cos(y) + 3.0*y - 2.0; };            // Left:   u - ux

        laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::ROBIN, ExecutionMode::PARALLEL, funcType> solver(data);
        solver.build_mesh();
        solver.solve();
        solver.print();
        solver.export_to_vtk("test_21_robin_parallel.vtk");
    }

    // =========================================================================
    // TEST 22: NEUMANN SEQUENTIAL vs PARALLEL (Schwarz)
    // =========================================================================
    /**
     * @brief Test 22: Performance convergence benchmark for a polynomial solution under Neumann conditions using the Schwarz solver.
     * @param mpi_rank Rank of the calling MPI process.
     */
    inline void run_test_22_neumann_sequential_vs_parallel_schwarz(int mpi_rank) {
        if (mpi_rank == 0) {
            std::cout << "\n=== Running Test 22: NEUMANN SEQUENTIAL vs PARALLEL (Schwarz) ===" << std::endl;
        }

        auto data = create_default_data();
        
        // Exact solution: u(x,y) = x^2*y + x*y^2 + x + y + 1
        data.u_exact_lambda = [](double x, double y) { return x*x*y + x*y*y + x + y + 1.0; };
        
        // Forcing term: f0 = -Laplacian(u) = -2(x+y)
        data.f0 = [](double x, double y) { return -2.0 * (x + y); };
        
        // Neumann boundaries: du/dn = grad(u) * n
        data.f1 = [](double x, double y) { return -(x*x + 1.0); };                      // Bottom (n =  0, -1) -> -uy
        data.f2 = [](double x, double y) { return y*y + 2.0*y + 1.0; };                 // Right  (n =  1,  0) ->  ux
        data.f3 = [](double x, double y) { return x*x + 2.0*x + 1.0; };                 // Top    (n =  0,  1) ->  uy
        data.f4 = [](double x, double y) { return -(y*y + 1.0); };  

        laplacian_solvers::Convergence_Test<SolverType::SCHWARZ, BoundaryCondition::NEUMANN, funcType> convergence_test(data);
        auto convergence_results = convergence_test.run_convergence_test();
        convergence_test.print_convergence_results();
    }

} // namespace laplacian_solvers

#endif // LAPLACIAN_SOLVERS_TEST_2_HPP