#ifndef LAPLACIAN_SOLVERS_PARALLEL_SCHWARZ_IMPLEMENTATION_HPP
#define LAPLACIAN_SOLVERS_PARALLEL_SCHWARZ_IMPLEMENTATION_HPP

#include "laplacian_solvers.hpp"
#include "laplacian_solvers_boundary_conditions.hpp"
#include <iostream>
#include <cmath>
#include <vector>

namespace laplacian_solvers{

 
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::schwarz_parallel_dirichlet(){

        return Result_Struct(); // Placeholder
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::schwarz_parallel_neumann(){
        
        return Result_Struct(); // Placeholder
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::schwarz_parallel_robin(){
        
        return Result_Struct(); // Placeholder
    }
        
}
#endif
