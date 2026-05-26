#ifndef LAPLACIAN_SOLVERS_SEQUENTIAL_IMPLEMENTATIONS_HPP
#define LAPLACIAN_SOLVERS_SEQUENTIAL_IMPLEMENTATIONS_HPP

#include "laplacian_solvers.hpp"
#include "laplacian_solvers_boundary_conditions.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>


namespace laplacian_solvers{
     /**
     * @brief Solves the Laplacian using the sequential Jacobi iterative method with Dirichlet BCs.
     * The method updates the entire domain grid iteratively by averaging neighbors until 
     * the residual error falls below the specified tolerance or max iterations are reached.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_sequential_dirichlet(){

        eigenMatrix u_new = u_h;
        unsigned iter = 0;
        double err = data.tolerance + 1.0; 
        const double h2 = h * h;    

        apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h, data, meshX, meshY);

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

    result.u_h = u_h;
    result.iterations = iter;
    result.X = meshX;
    result.Y = meshY;
    result.iterartion_residue = err; 
    return result;
    
    }

    /**
     * @brief Solves the Laplacian using the sequential Jacobi iterative method with Neumann BCs.
     * Implements finite difference approximations for the derivative boundary conditions.
     * The edge values are updated based on the gradient function (f1-f4) and the nearest inner node.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_sequential_neumann(){
        
        eigenMatrix u_new = u_h; // Parte da zero (inizializzati nel costruttore)
        unsigned iter = 0;
        double err = data.tolerance + 1.0;
        const double h2 = h * h;
        const unsigned last = data.n - 1;

        while (err > data.tolerance && iter < data.max_iterations) {
            err = 0.0;

            for (unsigned i = 1; i < last; ++i) {
                for (unsigned j = 1; j < last; ++j) {
                    u_new(i, j) = 0.25 * (u_h(i-1, j) + u_h(i+1, j) + u_h(i, j-1) + u_h(i, j+1) + h2 * data.f0(meshX(i, j), meshY(i, j)));
                    
                    double diff = std::abs(u_new(i, j) - u_h(i, j));
                    if (diff > err) err = diff;
                }
            }

            /*
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
            */

            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_new, data, meshX, meshY, u_h);

            for (unsigned i = 0; i < data.n; ++i) {
                err = std::max({err, std::abs(u_new(i, 0) - u_h(i, 0)), 
                                       std::abs(u_new(i, last) - u_h(i, last)),
                                       std::abs(u_new(0, i) - u_h(0, i)), 
                                       std::abs(u_new(last, i) - u_h(last, i))});
            }

            u_h = u_new;

            // Normalizing for unicity
            double mean = u_h.sum() / (double)(data.n * data.n);
            u_h.array() -= mean;

            iter++;
        }

        result.u_h = u_h;
        result.iterations = iter;
        result.X = meshX;
        result.Y = meshY;
        result.iterartion_residue = err; 
        return result;
    }

    /**
     * @brief Solves the Laplacian using the sequential Jacobi iterative method with Robin BCs.
     * Implements mixed boundary conditions. The update rule involves a weighted average 
     * regulated by the 'alpha' parameter to represent the Robin boundary constraint.
     */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_sequential_robin(){
        apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h, data, meshX, meshY, u_h); //Warm start

        eigenMatrix u_new = u_h; // Inizializzata a zeri dal costruttore
        unsigned iter = 0;
        double err = data.tolerance + 1.0;
        const double h2 = h * h;
        const unsigned last = data.n - 1;

        while (err > data.tolerance && iter < data.max_iterations) {
            err = 0.0;

            for (unsigned i = 1; i < last; ++i) {
                for (unsigned j = 1; j < last; ++j) {
                    u_new(i, j) = 0.25 * (u_h(i-1, j) + u_h(i+1, j) + 
                                        u_h(i, j-1) + u_h(i, j+1) + 
                                        h2 * data.f0(meshX(i, j), meshY(i, j)));
                    
                    double diff = std::abs(u_new(i, j) - u_h(i, j));
                    if (diff > err) err = diff;
                }
            }

            /*
            for (unsigned i = 0; i < data.n; ++i) {
                // Left edge
                u_new(i, 0) = (u_new(i, 1) + h * data.f4(meshX(i, 0), meshY(i, 0))) / den;
                
                // Right edge 
                u_new(i, last) = (u_new(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last))) / den;
                
                // Top edge 
                u_new(0, i) = (u_new(1, i) + h * data.f3(meshX(0, i), meshY(0, i))) / den;
                
                // Bottom edge 
                u_new(last, i) = (u_new(last - 1, i) + h * data.f1(meshX(last, i), meshY(last, i))) / den;
            }*/

            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_new, data, meshX, meshY, u_h);

            for (unsigned i = 0; i < data.n; ++i) {
                err = std::max({err, std::abs(u_new(i, 0) - u_h(i, 0)), 
                                       std::abs(u_new(i, last) - u_h(i, last)),
                                       std::abs(u_new(0, i) - u_h(0, i)), 
                                       std::abs(u_new(last, i) - u_h(last, i))});
            }

            u_h = u_new;
            iter++;
        }

        result.u_h = u_h;
        result.iterations = iter;
        result.X = meshX;
        result.Y = meshY;
        result.iterartion_residue = err; 
        return result;
    }

}

#endif