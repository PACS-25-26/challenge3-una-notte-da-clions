/**
 * @file laplacian_solvers_implementation.hpp
 * @brief Template-based implementation of the 2D Laplace and Poisson solvers.
 * @details Contains the concrete method definitions for the Laplacian_Solver class,
 *          organizing the compile-time dispatching chain (execution mode -> solver type -> boundary conditions)
 *          and implementing Jacobi and Schwarz methods.
 */
#ifndef LAPLACIAN_SOLVERS_IMPLEMENTATION_HPP
#define LAPLACIAN_SOLVERS_IMPLEMENTATION_HPP
#include "laplacian_solvers.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

namespace laplacian_solvers{

    /* --- CONSTRUCTOR --- */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::Laplacian_Solver(const Data_Struct<Func0, Func1, Func2, Func3, Func4, u_ex>& d) : data(d) {
        meshX = Eigen::MatrixXd::Zero(data.n, data.n);
        meshY = Eigen::MatrixXd::Zero(data.n, data.n);
        h = (data.x2 - data.x1) / (data.n - 1);
        u_h = Eigen::MatrixXd::Zero(data.n, data.n);
        u_exact = Eigen::MatrixXd::Zero(data.n, data.n);

        for (unsigned i = 0; i < data.n; ++i) {
            for (unsigned j = 0; j < data.n; ++j) {
                meshX(i, j) = data.x1 + j * h;
                meshY(i, j) = data.x2 - i * h;
            }
        }

        for (unsigned i = 0; i < data.n; ++i) {
            for (unsigned j = 0; j < data.n; ++j) {
                u_exact(i, j) = data.u_exact_lambda(meshX(i, j), meshY(i, j));
            }
        }
        
        if constexpr (boundary_condition == BoundaryCondition::DIRICHLET){
            for (unsigned i = 0; i < data.n; ++i) {
                u_h(i, 0) = data.f4(meshX(i, 0), meshY(i, 0)); // Left edge
                u_h(i, data.n - 1) = data.f2(meshX(i, data.n - 1), meshY(i, data.n - 1)); // Right edge
                u_h(0, i) = data.f3(meshX(0, i), meshY(0, i)); // Top edge
                u_h(data.n - 1, i) = data.f1(meshX(data.n - 1, i), meshY(data.n - 1, i)); // Bottom edge
            }
            
        }

    };

    /* --- SOLVE METHOD --- */

    //sequential or parallel?
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::solve(){

        if constexpr (execution_mode == ExecutionMode::SEQUENTIAL) {
            return sequential_solve();
        } else {
            return parallel_solve();
        }
        
    }

    // Jacobi or Schwarz?
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::sequential_solve(){
        if constexpr (solver_type == SolverType::JACOBI) {
            return jacobi_sequential();
        } else {
            throw std::runtime_error("Schwarz is a parallel solver.");
        }
    }

    
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::parallel_solve(){
        if constexpr (solver_type == SolverType::JACOBI) {
            return jacobi_parallel();
        } else {
            return schwarz_parallel();
        }
    }

    // DIRICHLET or NEUMANN or ROBIN?
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_sequential(){
        if constexpr (boundary_condition == BoundaryCondition::DIRICHLET) {
            return jacobi_sequential_dirichlet();
        } else if constexpr (boundary_condition == BoundaryCondition::NEUMANN) {
            return jacobi_sequential_neumann();
        } else {
            return jacobi_sequential_robin();
        }
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_parallel(){
        if constexpr (boundary_condition == BoundaryCondition::DIRICHLET) {
            return jacobi_parallel_dirichlet();
        } else if constexpr (boundary_condition == BoundaryCondition::NEUMANN) {
            return jacobi_parallel_neumann();
        } else {
            return jacobi_parallel_robin();
        }
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::schwarz_parallel(){
        if constexpr (boundary_condition == BoundaryCondition::DIRICHLET) {
            return schwarz_parallel_dirichlet();
        } else if constexpr (boundary_condition == BoundaryCondition::NEUMANN) {
            return schwarz_parallel_neumann();
        } else {
            return schwarz_parallel_robin();
        }
    }

    /* --- ACTUAL SOLVERS JACOBI --- */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_sequential_dirichlet(){

        Eigen::MatrixXd u_new = u_h;
        unsigned iter = 0;
        double err = data.tolerance + 1.0; 
        const double h2 = h * h;    

        while (err > data.tolerance && iter < data.max_iterations) {
            err = 0.0;
        for (unsigned i = 1; i < data.n - 1; ++i) {
            for (unsigned j = 1; j < data.n - 1; ++j) {
                
                u_new(i, j) = 0.25 * (u_h(i-1, j) + u_h(i+1, j) + u_h(i, j-1) + u_h(i, j+1) + h2 * data.f0(meshX(i, j), meshY(i, j)));
                
                double local_err = std::abs(u_new(i, j) - u_h(i, j));
                if (local_err > err) {
                    err = local_err;
                }
            }
        }

        u_h = u_new; 
        iter++;
    }

    Result_Struct result;
    result.u_h = u_h;
    result.iterations = iter;
    result.X = meshX;
    result.Y = meshY;
    return result;
    
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_sequential_neumann(){
        // Implementazione del metodo di Jacobi sequenziale con condizioni al contorno di Neumann
        Eigen::MatrixXd u_new = u_h; // Parte da zeri (inizializzati nel costruttore)
        unsigned iter = 0;
        double error = data.tolerance + 1.0;
        const double h2 = h * h;
        const unsigned last = data.n - 1;

        while (error > data.tolerance && iter < data.max_iterations) {
            error = 0.0;

            for (unsigned i = 1; i < last; ++i) {
                for (unsigned j = 1; j < last; ++j) {
                    u_new(i, j) = 0.25 * (u_h(i-1, j) + u_h(i+1, j) + u_h(i, j-1) + u_h(i, j+1) + h2 * data.f0(meshX(i, j), meshY(i, j)));
                    
                    double diff = std::abs(u_new(i, j) - u_h(i, j));
                    if (diff > error) error = diff;
                }
            }

            //BC NEUMANN
            for (unsigned i = 0; i < data.n; ++i) {
                // left edge
                u_new(i, 0) = u_new(i, 1) + h * data.f4(meshX(i, 0), meshY(i, 0));
                
                // right edge
                u_new(i, last) = u_new(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last));
                
                // top edge 
                u_new(0, i) = u_new(1, i) + h * data.f3(meshX(0, i), meshY(0, i));
                
                // bottom edge 
                u_new(last, i) = u_new(last - 1, i) + h * data.f1(meshX(last, i), meshY(last, i));
            }

            for (unsigned i = 0; i < data.n; ++i) {
                error = std::max({error, std::abs(u_new(i, 0) - u_h(i, 0)), 
                                        std::abs(u_new(i, last) - u_h(i, last)),
                                        std::abs(u_new(0, i) - u_h(0, i)), 
                                        std::abs(u_new(last, i) - u_h(last, i))});
            }

            u_h = u_new;
            iter++;
        }

        Result_Struct result;
        result.u_h = u_h;
        result.iterations = iter;
        result.X = meshX;
        result.Y = meshY;
        return result;
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_sequential_robin(){
        // Implementazione del metodo di Jacobi sequenziale con condizioni al contorno di Robin
        Eigen::MatrixXd u_new = u_h; // Inizializzata a zeri dal costruttore
        unsigned iter = 0;
        double error = data.tolerance + 1.0;
        const double h2 = h * h;
        const unsigned last = data.n - 1;

        const double den = 1.0 + data.gamma * h;

        while (error > data.tolerance && iter < data.max_iterations) {
            error = 0.0;

            for (unsigned i = 1; i < last; ++i) {
                for (unsigned j = 1; j < last; ++j) {
                    u_new(i, j) = 0.25 * (u_h(i-1, j) + u_h(i+1, j) + 
                                        u_h(i, j-1) + u_h(i, j+1) + 
                                        h2 * data.f0(meshX(i, j), meshY(i, j)));
                    
                    double diff = std::abs(u_new(i, j) - u_h(i, j));
                    if (diff > error) error = diff;
                }
            }

            for (unsigned i = 0; i < data.n; ++i) {
                // Left edge
                u_new(i, 0) = (u_new(i, 1) + h * data.f4(meshX(i, 0), meshY(i, 0))) / den;
                
                // Right edge 
                u_new(i, last) = (u_new(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last))) / den;
                
                // Top edge 
                u_new(0, i) = (u_new(1, i) + h * data.f3(meshX(0, i), meshY(0, i))) / den;
                
                // Bottom edge 
                u_new(last, i) = (u_new(last - 1, i) + h * data.f1(meshX(last, i), meshY(last, i))) / den;
            }

            for (unsigned i = 0; i < data.n; ++i) {
                error = std::max({error, std::abs(u_new(i, 0) - u_h(i, 0)), 
                                        std::abs(u_new(i, last) - u_h(i, last)),
                                        std::abs(u_new(0, i) - u_h(0, i)), 
                                        std::abs(u_new(last, i) - u_h(last, i))});
            }

            u_h = u_new;
            iter++;
        }

        Result_Struct result;
        result.u_h = u_h;
        result.iterations = iter;
        result.X = meshX;
        result.Y = meshY;
        return result;
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_parallel_dirichlet(){
        // Implementazione del metodo di Jacobi parallelo con condizioni al contorno di Dirichlet
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_parallel_neumann(){
        // Implementazione del metodo di Jacobi parallelo con condizioni al contorno di Neumann
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_parallel_robin(){
        // Implementazione del metodo di Jacobi parallelo con condizioni al contorno di Robin
    }

    /* --- ACTUAL SOLVERS SCHWARZ --- */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::schwarz_parallel_dirichlet(){
        // Implementazione del metodo di Schwarz parallelo con condizioni al contorno di Dirichlet
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::schwarz_parallel_neumann(){
        // Implementazione del metodo di Schwarz parallelo con condizioni al contorno di Neumann
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::schwarz_parallel_robin(){
        // Implementazione del metodo di Schwarz parallelo con condizioni al contorno di Robin
    }


    /* --- CONVERGENCE TEST METHOD --- */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Convergence_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::convergence_test(){
        
    }

    /* --- PRINT AND EXPORT METHODS --- */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::print(){
        // STAMPA GRIGLIA, SOLUZIONE DISCRETA, SOLUZIONE ESATTA SULLA GRIGLIA
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::export_to_vtk(const Eigen::MatrixXd& meshX, const Eigen::MatrixXd& meshY, const Eigen::MatrixXd& u_h, const std::string& filename){
        // Esporta i dati in formato VTK per la visualizzazione con Paraview
    }

}

#endif