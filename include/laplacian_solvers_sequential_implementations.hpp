#ifndef LAPLACIAN_SOLVERS_SEQUENTIAL_IMPLEMENTATIONS_HPP
#define LAPLACIAN_SOLVERS_SEQUENTIAL_IMPLEMENTATIONS_HPP

#include "laplacian_solvers.hpp"
#include "laplacian_solvers_boundary_conditions.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>


namespace laplacian_solvers{
     
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_sequential(){
        
        // Initialize the solver
        eigenMatrix u_h = eigenMatrix::Zero(data.n, data.n);

        // Dirichlet and Robin BCs must be one the initial guess
        if constexpr(boundary_condition == BoundaryCondition::DIRICHLET || boundary_condition == BoundaryCondition::ROBIN) 
            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h, data, meshX, meshY, u_h); 

        eigenMatrix u_new(u_h); 
        unsigned iter = 0;
        const double tol_squared = data.tolerance * data.tolerance;
        double err = tol_squared + 1.0;
        const double h2 = h * h;
        const unsigned last = data.n - 1;
        

        // Main iteration loop
        while (err > tol_squared && iter < data.max_iterations) {
            
            // Compute next iteration
            for (unsigned i = 1; i < last; ++i) {
                for (unsigned j = 1; j < last; ++j) {
                    u_new(i, j) = 0.25 * (u_h(i-1, j) + u_h(i+1, j) + u_h(i, j-1) + u_h(i, j+1) + h2 * data.f0(meshX(i, j), meshY(i, j)));
                }
            }   

            // Neumann and Robin BCs must be applied at each iteration
            if constexpr(boundary_condition == BoundaryCondition::NEUMANN || boundary_condition == BoundaryCondition::ROBIN) 
                apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_new, data, meshX, meshY, u_h);
            
            // Neumann requires normalization to ensure unicity of the solution
            if constexpr(boundary_condition == BoundaryCondition::NEUMANN){
                const double mean = u_h.sum() / (double)(data.n * data.n);
                u_h.array() -= mean;
            }
            
            // Compute error using modified l2 norm and prepare for next iteration
            err = h * (u_new - u_h).squaredNorm();
            u_h = u_new;
            iter++;
        }

        // Update result struct with final solution and iteration info
        result.u_h = u_h;
        result.iterations = iter;
        result.X = meshX;
        result.Y = meshY;
        result.iterartion_residue = std::sqrt(err); 
        result.valid = true;
        return result;
    }
}

#endif