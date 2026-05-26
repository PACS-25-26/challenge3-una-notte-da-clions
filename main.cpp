#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <functional>
#include <cmath>

#include "laplacian_solvers.hpp"

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    using funcType = std::function<double(double, double)>;

    // Configurazione base della struttura dati
    Data_Struct<funcType> data;
    data.n = 20; 
    data.x1 = 0.0; 
    data.x2 = 1.0;
    data.tolerance = 1e-6;
    data.max_iterations = 10000;
    data.alpha = 1.0; // Parametro per Robin (du/dn + alpha*u = g)

    // =========================================================================
    // CASO 1: DIRICHLET
    // Soluzione esatta: u(x,y) = sin(2*pi*x) * sin(2*pi*y)
    // -Laplaciano(u) = 8 * pi^2 * u
    // Condizioni ai bordi: u = 0 ovunque
    // =========================================================================
    std::cout << "\n=== Test 1: DIRICHLET ===" << std::endl;
    data.f0 = [](double x, double y) { return 8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y); };
    data.f1 = [](double x, double y) { return 0.0; }; // Bottom
    data.f2 = [](double x, double y) { return 0.0; }; // Right
    data.f3 = [](double x, double y) { return 0.0; }; // Top
    data.f4 = [](double x, double y) { return 0.0; }; // Left
    data.u_exact_lambda = [](double x, double y) { return std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y); };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::DIRICHLET, ExecutionMode::SEQUENTIAL, funcType> solver_dirichlet(data);
    solver_dirichlet.build_mesh();
    solver_dirichlet.solve();
    solver_dirichlet.print();

    // =========================================================================
    // CASO 2: NEUMANN
    // Soluzione esatta: u(x,y) = cos(2*pi*x) * cos(2*pi*y)
    // -Laplaciano(u) = 8 * pi^2 * u
    // Condizioni ai bordi: grad(u) * n = 0 (flusso nullo sui bordi per questa funzione)
    // =========================================================================
    std::cout << "\n=== Test 2: NEUMANN ===" << std::endl;
    data.f0 = [](double x, double y) { return 8.0 * M_PI * M_PI * std::cos(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y); };
    data.f1 = [](double x, double y) { return 0.0; }; 
    data.f2 = [](double x, double y) { return 0.0; }; 
    data.f3 = [](double x, double y) { return 0.0; }; 
    data.f4 = [](double x, double y) { return 0.0; }; 
    data.u_exact_lambda = [](double x, double y) { return std::cos(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y); };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::NEUMANN, ExecutionMode::SEQUENTIAL, funcType> solver_neumann(data);
    solver_neumann.build_mesh();
    solver_neumann.solve();
    solver_neumann.print();

    // =========================================================================
    // CASO 3: ROBIN
    // Soluzione esatta: u(x,y) = x^2 + y^2
    // -Laplaciano(u) = - (2 + 2) = -4.0
    // Condizioni ai bordi: g = du/dn + alpha * u (con alpha = 1.0 e n normale uscente)
    // =========================================================================
    std::cout << "\n=== Test 3: ROBIN ===" << std::endl;
    data.f0 = [](double x, double y) { return -4.0; };
    
    // Bottom (y=0, n=[0,-1]) -> du/dn = -dy = 0   -> g = 0 + u(x,0) = x^2
    data.f1 = [](double x, double y) { return x * x; }; 
    
    // Right (x=1, n=[1,0])   -> du/dn = dx = 2(1) -> g = 2 + u(1,y) = 2 + 1 + y^2 = 3 + y^2
    data.f2 = [](double x, double y) { return 3.0 + y * y; }; 
    
    // Top (y=1, n=[0,1])     -> du/dn = dy = 2(1) -> g = 2 + u(x,1) = 2 + x^2 + 1 = 3 + x^2
    data.f3 = [](double x, double y) { return 3.0 + x * x; }; 
    
    // Left (x=0, n=[-1,0])   -> du/dn = -dx = 0   -> g = 0 + u(0,y) = y^2
    data.f4 = [](double x, double y) { return y * y; }; 
    
    data.u_exact_lambda = [](double x, double y) { return x * x + y * y; };

    laplacian_solvers::Laplacian_Solver<SolverType::JACOBI, BoundaryCondition::ROBIN, ExecutionMode::SEQUENTIAL, funcType> solver_robin(data);
    solver_robin.build_mesh();
    solver_robin.solve();
    solver_robin.print();

    MPI_Finalize();
    return 0;
}