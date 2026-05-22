#ifndef LAPLACIAN_SOLVERS_IMPLEMENTATION_HPP
#define LAPLACIAN_SOLVERS_IMPLEMENTATION_HPP
#include "laplacian_solvers.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

namespace laplacian_solvers{

    // CONSTRUCTOR
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

    // SOLVE METHOD //

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
            return schwarz_sequential();
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
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::schwarz_sequential(){
        if constexpr (boundary_condition == BoundaryCondition::DIRICHLET) {
            return schwarz_sequential_dirichlet();
        } else if constexpr (boundary_condition == BoundaryCondition::NEUMANN) {
            return schwarz_sequential_neumann();
        } else {
            return schwarz_sequential_robin();
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

    // ACTUAL SOLVERS JACOBI

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_sequential_dirichlet(){
        // Implementazione del metodo di Jacobi sequenziale con condizioni al contorno di Dirichlet
    
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_sequential_neumann(){
        // Implementazione del metodo di Jacobi sequenziale con condizioni al contorno di Neumann
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::jacobi_sequential_robin(){
        // Implementazione del metodo di Jacobi sequenziale con condizioni al contorno di Robin
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

    //ACTUAL SOLVERS SCHWARZ

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::schwarz_sequential_dirichlet(){
        // Implementazione del metodo di Schwarz sequenziale con condizioni al contorno di Dirichlet
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::schwarz_sequential_neumann(){
        // Implementazione del metodo di Schwarz sequenziale con condizioni al contorno di Neumann
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::schwarz_sequential_robin(){
        // Implementazione del metodo di Schwarz sequenziale con condizioni al contorno di Robin
    }

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


    // CONVERGENCE TEST METHOD
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    Convergence_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::convergence_test(){
        
    }

    // PRINT AND EXPORT METHODS
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::print(){
        // STAMPA GRIGLIA, SOLUZIONE DISCRETA, SOLUZIONE ESATTA
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename Func0, typename Func1, typename Func2, typename Func3, typename Func4, typename u_ex>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, Func0, Func1, Func2, Func3, Func4, u_ex>::export_to_vtk(const Eigen::MatrixXd& meshX, const Eigen::MatrixXd& meshY, const Eigen::MatrixXd& u_h, const std::string& filename){
        // Esporta i dati in formato VTK per la visualizzazione con Paraview
    }

}

#endif