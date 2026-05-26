#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <functional>
#include <cmath>

#include "laplacian_solvers.hpp"

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    using funcType = std::function<double(double, double)>;

    Data_Struct<funcType> data;
    data.n = 10;
    data.x1 = 0.0;
    data.x2 = 1.0;
    data.tolerance = 1e-6;
    data.max_iterations = 100000;
    data.alpha = 1.0;

    // =========================================================================
    // TEST 1: DIRICHLET
    // Exact solution: u(x,y) = sin(pi*x) * sin(pi*y)
    // -Laplacian(u) = 2*pi^2 * sin(pi*x)*sin(pi*y)
    // BC: u = 0 on all edges (sin vanishes at 0 and 1 -> exact Dirichlet)
    // Low frequency: Jacobi converges in ~10000 iterations on a 20x20 grid
    // =========================================================================
    std::cout << "\n=== Test 1: DIRICHLET ===" << std::endl;
    data.f0 = [](double x, double y) {
        return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
    };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 0.0; };
    data.f3 = [](double x, double y) { return 0.0; };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) {
        return std::sin(M_PI * x) * std::sin(M_PI * y);
    };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::DIRICHLET, ExecutionMode::SEQUENTIAL, funcType> solver_dirichlet(data);
    solver_dirichlet.build_mesh();
    //solver_dirichlet.solve();
    //solver_dirichlet.print();

    // =========================================================================
    // TEST 2: NEUMANN
    // Exact solution: u(x,y) = cos(pi*x) * cos(pi*y)
    // -Laplacian(u) = 2*pi^2 * cos(pi*x)*cos(pi*y)
    // Neumann BC: du/dn = 0 on all edges
    //   Bottom (y=0, n=[0,-1]): -du/dy = pi*cos(pi*x)*sin(0) = 0       ok
    //   Top    (y=1, n=[0,+1]): +du/dy = -pi*cos(pi*x)*sin(pi) = 0     ok
    //   Left   (x=0, n=[-1,0]): -du/dx = pi*sin(0)*cos(pi*y) = 0       ok
    //   Right  (x=1, n=[+1,0]): +du/dx = -pi*sin(pi)*cos(pi*y) = 0     ok
    // Compatibility: integral(f) over [0,1]^2 = 2*pi^2 * 0 = 0         ok
    // Normalization: mean of cos(pi*x)*cos(pi*y) over [0,1]^2 = 0      ok
    // =========================================================================
    std::cout << "\n=== Test 2: NEUMANN ===" << std::endl;
    data.f0 = [](double x, double y) {
        return 2.0 * M_PI * M_PI * std::cos(M_PI * x) * std::cos(M_PI * y);
    };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 0.0; };
    data.f3 = [](double x, double y) { return 0.0; };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) {
        return std::cos(M_PI * x) * std::cos(M_PI * y);
    };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::NEUMANN, ExecutionMode::SEQUENTIAL, funcType> solver_neumann(data);
    solver_neumann.build_mesh();
    solver_neumann.solve();
    solver_neumann.print();

    // =========================================================================
    // TEST 3: ROBIN
    // Exact solution: u(x,y) = x^2 + y^2
    // -Laplacian(u) = -4
    // Robin BC: du/dn + alpha*u = g  (alpha = 1)
    //   Bottom (y=0, n=[0,-1]): du/dn = -du/dy = 0  -> g = 0 + u(x,0) = x^2
    //   Right  (x=1, n=[+1,0]): du/dn = +du/dx = 2  -> g = 2 + u(1,y) = 3 + y^2
    //   Top    (y=1, n=[0,+1]): du/dn = +du/dy = 2  -> g = 2 + u(x,1) = 3 + x^2
    //   Left   (x=0, n=[-1,0]): du/dn = -du/dx = 0  -> g = 0 + u(0,y) = y^2
    // Well-posed: Robin term alpha*u breaks the null-space -> unique solution
    // =========================================================================
    std::cout << "\n=== Test 3: ROBIN ===" << std::endl;
    data.f0 = [](double x, double y) { return -2.0 * std::exp(x + y); };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 2.0 * std::exp(1.0 + y); };
    data.f3 = [](double x, double y) { return 2.0 * std::exp(x + 1.0); };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) { return std::exp(x + y); };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::ROBIN, ExecutionMode::SEQUENTIAL, funcType> solver_robin(data);
    solver_robin.build_mesh();
    //solver_robin.solve();
    //solver_robin.print();

    // =========================================================================
    // TEST 4: DIRICHLET PARALLEL
    // Exact solution: u(x,y) = sin(pi*x) * sin(pi*y)
    // -Laplacian(u) = 2*pi^2 * sin(pi*x)*sin(pi*y)
    // BC: u = 0 on all edges (sin vanishes at 0 and 1 -> exact Dirichlet)
    // Low frequency: Jacobi converges in ~10000 iterations on a 20x20 grid
    // =========================================================================
    std::cout << "\n=== Test 4: DIRICHLET PARALLEL ===" << std::endl;
    data.f0 = [](double x, double y) {
        return 2.0 * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
    };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 0.0; };
    data.f3 = [](double x, double y) { return 0.0; };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) {
        return std::sin(M_PI * x) * std::sin(M_PI * y);
    };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::DIRICHLET, ExecutionMode::PARALLEL, funcType> solver_dirichlet_parallel(data);
    //solver_dirichlet_parallel.build_mesh();
    //solver_dirichlet_parallel.solve();
    //solver_dirichlet_parallel.print();

    // =========================================================================
    // TEST 5: NEUMANN
    // Exact solution: u(x,y) = cos(pi*x) * cos(pi*y)
    // -Laplacian(u) = 2*pi^2 * cos(pi*x)*cos(pi*y)
    // Neumann BC: du/dn = 0 on all edges
    //   Bottom (y=0, n=[0,-1]): -du/dy = pi*cos(pi*x)*sin(0) = 0       ok
    //   Top    (y=1, n=[0,+1]): +du/dy = -pi*cos(pi*x)*sin(pi) = 0     ok
    //   Left   (x=0, n=[-1,0]): -du/dx = pi*sin(0)*cos(pi*y) = 0       ok
    //   Right  (x=1, n=[+1,0]): +du/dx = -pi*sin(pi)*cos(pi*y) = 0     ok
    // Compatibility: integral(f) over [0,1]^2 = 2*pi^2 * 0 = 0         ok
    // Normalization: mean of cos(pi*x)*cos(pi*y) over [0,1]^2 = 0      ok
    // =========================================================================
    std::cout << "\n=== Test 5: NEUMANN ===" << std::endl;
    data.f0 = [](double x, double y) {
        return 2.0 * M_PI * M_PI * std::cos(M_PI * x) * std::cos(M_PI * y);
    };
    data.f1 = [](double x, double y) { return 0.0; };
    data.f2 = [](double x, double y) { return 0.0; };
    data.f3 = [](double x, double y) { return 0.0; };
    data.f4 = [](double x, double y) { return 0.0; };
    data.u_exact_lambda = [](double x, double y) {
        return std::cos(M_PI * x) * std::cos(M_PI * y);
    };
    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::NEUMANN, ExecutionMode::PARALLEL, funcType> solver_neumann_parallel(data);
    solver_neumann_parallel.build_mesh();
    solver_neumann_parallel.solve();
    solver_neumann_parallel.print();

    MPI_Finalize();
    return 0;
}