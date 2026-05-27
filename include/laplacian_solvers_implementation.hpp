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
#include "laplacian_solvers_boundary_conditions.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>


namespace laplacian_solvers{

    /* --- SOLVE METHOD --- */

    //sequential or parallel?
    /**
     * @brief Primary dispatcher for the Laplacian solver.
     * Routes the execution flow to either sequential or parallel solvers based on the compile-time 
     * 'execution_mode' parameter using constexpr if-branching.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::solve(){

        if constexpr (execution_mode == ExecutionMode::SEQUENTIAL) {
            return sequential_solve();
        } else {
            return parallel_solve();
        }
        
    }

    // Jacobi or Schwarz?

    /**
     * @brief Dispatcher for sequential solver logic.
     * Routes to the Jacobi sequential implementation. Throws a std::runtime_error if Schwarz 
     * is selected, as it requires distributed-memory parallelism (MPI).
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::sequential_solve(){
       return jacobi_sequential();
    }

    /**
     * @brief Dispatcher for parallel solver logic.
     * Routes to either Jacobi parallel or Schwarz parallel implementations based on 
     * the compile-time 'solver_type' parameter.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::parallel_solve(){
        if constexpr (solver_type == SolverType::JACOBI) {
            return jacobi_parallel();
        } else {
            return schwarz_parallel();
        }
    }

    // DIRICHLET or NEUMANN or ROBIN?
    
    /**
     * @brief Dispatches the sequential Jacobi solver to the specific boundary condition implementation.
     * Maps the 'boundary_condition' template parameter to Dirichlet, Neumann, or Robin methods.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_sequential(){
        /*if constexpr (boundary_condition == BoundaryCondition::DIRICHLET) {
            return jacobi_sequential_dirichlet();
        } else if constexpr (boundary_condition == BoundaryCondition::NEUMANN) {
            return jacobi_sequential_neumann();
        } else {
            return jacobi_sequential_robin();
        }*/
        return jacobi_sequential_merged();
    }

    /**
     * @brief Dispatches the parallel Jacobi solver to the specific boundary condition implementation.
     * Maps the 'boundary_condition' template parameter to Dirichlet, Neumann, or Robin methods.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_parallel(){
        if constexpr (boundary_condition == BoundaryCondition::DIRICHLET) {
            return jacobi_parallel_dirichlet();
        } else if constexpr (boundary_condition == BoundaryCondition::NEUMANN) {
            return jacobi_parallel_neumann();
        } else {
            return jacobi_parallel_robin();
        }
    }

    /**
     * @brief Dispatches the parallel Schwarz solver to the specific boundary condition implementation.
     * Routes the execution flow to the appropriate domain decomposition implementation.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::schwarz_parallel(){
        if constexpr (boundary_condition == BoundaryCondition::DIRICHLET) {
            return schwarz_parallel_dirichlet();
        } else if constexpr (boundary_condition == BoundaryCondition::NEUMANN) {
            return schwarz_parallel_neumann();
        } else {
            return schwarz_parallel_robin();
        }
    } 

}

#endif