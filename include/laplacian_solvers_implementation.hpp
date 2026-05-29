#ifndef LAPLACIAN_SOLVERS_IMPLEMENTATION_HPP
#define LAPLACIAN_SOLVERS_IMPLEMENTATION_HPP

#include "laplacian_solvers.hpp"
#include "laplacian_solvers_boundary_conditions.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

/**
 * @file laplacian_solvers_implementation.hpp
 * @brief Dispatching logic for the sequential and parallel Laplacian solvers.
 */

namespace laplacian_solvers{

    /**
     * @brief Method that triggers the appropriate linear solver backend.
     * * Checks process validity and uses compile-time template routing (`if constexpr`) to 
     * dispatch execution based on the configured @ref ExecutionMode.
     * * @tparam solver_type Iterative algorithm selector.
     * @tparam boundary_condition Boundary condition policy.
     * @tparam execution_mode Parallelism execution backend policy.
     * @tparam funcType Callable type for boundary and source terms.
     * * @return A @ref Result_Struct containing the final solution state. 
     * * @note Non-participating MPI processes or non-zero ranks under sequential mode will immediately 
     * return an uninitialized or incomplete result object.
     */
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

    /**
     * @brief Dispatches the parallel execution loop to the selected numerical algorithm.
     * * Performs compile-time static routing between a standard point-to-point 
     * Jacobi method or an overlapping/non-overlapping alternating Schwarz domain decomposition method.
     * * @tparam solver_type Iterative algorithm selector.
     * @tparam boundary_condition Boundary condition policy.
     * @tparam execution_mode Parallelism execution backend policy.
     * @tparam funcType Callable type for boundary and source terms.
     * * @return A @ref Result_Struct containing local or gathered compute states.
     */
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