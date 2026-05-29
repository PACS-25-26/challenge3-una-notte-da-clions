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

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::solve(){

        if(mpi_rank >= mpi_size) return result;
        
        if constexpr (execution_mode == ExecutionMode::SEQUENTIAL) {
            if(mpi_rank != 0) return result; // Only rank 0 will run the sequential solver
            return jacobi_sequential();
        } else {
            return parallel_solve();
        }
        
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::parallel_solve(){
        if constexpr (solver_type == SolverType::JACOBI) {
            return jacobi_parallel();
        } else {
            return schwarz_parallel();
        }
    }

}

#endif