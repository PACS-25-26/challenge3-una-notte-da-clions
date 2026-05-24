/**
 * @file laplacian_solvers_boundary_conditions.hpp
 * @brief Template-based implementation of boundary condition methods for the Laplacian solver.
 * @details Contains a template function for the implementation of a requested boundary condition. Since Neumann and Robin call their functions at
 * each iteration, a template function is used for optimization.
 */


#ifndef LAPLACIAN_SOLVERS_BOUNDARY_CONDITIONS_HPP
#define LAPLACIAN_SOLVERS_BOUNDARY_CONDITIONS_HPP
#include "laplacian_solvers.hpp"

/**
     * @brief General function for Dirichlet boundary conditions. This covers both the homogeneous and non-homogeneus cases
     * The template function can select the execution policy (sequential or parallel)
     */
template <ExecutionMode execution_mode, typename funcType>
void apply_dirichlet_condition(eigenMatrix & u_h, const Data_Struct<funcType>& data, const eigenMatrix & meshX, const eigenMatrix & meshY, const unsigned mpi_rank, const unsigned mpi_size){
    
    const unsigned last = data.n - 1;
    
    if constexpr(execution_mode == ExecutionMode::SEQUENTIAL){
        // Like in the contructor
        for(unsigned i = 0; i < data.n; i++){
            u_h(i, 0) = data.f4(meshX(i, 0), meshY(i, 0)); // Left edge
            u_h(i, last) = data.f2(meshX(i, last), meshY(i, last)); // Right edge
            u_h(0, i) = data.f3(meshX(0, i), meshY(0, i)); // Top edge
            u_h(last, i) = data.f1(meshX(last, i), meshY(last, i)); // Bottom edge
        }

    } else if constexpr(execution_mode == ExecutionMode::PARALLEL){
        // Each process will update the boundary conditions of its own subdomain -> maybe the work is not well balanced?

        // Bottom edge
        if(mpi_rank == 0) for(unsigned i = 0; i < data.n; i++) u_h(last, i) = data.f1(meshX(last, i), meshY(last, i));
        
        // Top edge
        if(mpi_rank == mpi_size - 1) for(unsigned i = 0; i < data.n; i++) u_h(0, i) = data.f3(meshX(0, i), meshY(0, i));

        // Left and right edges - might rewrite the corners
        for(unsigned i = 0; i < u_h.rows(); i++){
            u_h(i, 0) = data.f4(meshX(i, 0), meshY(i, 0));
            u_h(i, last) = data.f2(meshX(i, last), meshY(i, last));
        }
    }
}

/**
     * @brief General function for Neumann boundary conditions. This covers both the homogeneous and non-homogeneus cases
     * The template function can select the execution policy (sequential or parallel). Notice that the compatibility condition is not checked
     */
template <ExecutionMode execution_mode, typename funcType>
void apply_neumann_condition(eigenMatrix & u_h, const Data_Struct<funcType>& data, const eigenMatrix & meshX, const eigenMatrix & meshY, const unsigned mpi_rank, const unsigned mpi_size){
    
    const unsigned last = data.n - 1;
    const double h = abs(data.x2 - data.x1) / (data.n - 1);

    if constexpr(execution_mode == ExecutionMode::SEQUENTIAL){

         for (unsigned i = 0; i < data.n; i++) {
                u_h(i, 0) = u_h(i, 1) + h * data.f4(meshX(i, 0), meshY(i, 0)); // left edge
                u_h(i, last) = u_h(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last)); // right edge
                u_h(0, i) = u_h(1, i) + h * data.f3(meshX(0, i), meshY(0, i)); // top edge
                u_h(last, i) = u_h(last - 1, i) + h * data.f1(meshX(last, i), meshY(last, i)); // bottom edge 
            }

    } else if constexpr(execution_mode == ExecutionMode::PARALLEL){

        // As with Dirichlet, each process will update the boundary conditions of its own subdomain -> maybe the work is not well balanced?

        // Bottom edge
        if(mpi_rank == 0) for(unsigned i = 0; i < data.n; i++) u_h(last, i) = u_h(last - 1, i) + h * data.f1(meshX(last, i), meshY(last, i));
        
        // Top edge
        if(mpi_rank == mpi_size - 1) for(unsigned i = 0; i < data.n; i++) u_h(0, i) = u_h(1, i) + h * data.f3(meshX(0, i), meshY(0, i));

        // Left and right edges
        for(unsigned i = 0; i < u_h.rows(); i++){
            u_h(i, 0) = u_h(i, 1) + h * data.f4(meshX(i, 0), meshY(i, 0));
            u_h(i, last) = u_h(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last));
        }
    }
}

/**
     * @brief General function for Robin boundary conditions. This covers both the homogeneous and non-homogeneus cases
     * The template function can select the execution policy (sequential or parallel)
     */

template <ExecutionMode execution_mode, typename funcType>
void apply_robin_condition(eigenMatrix & u_h, const Data_Struct<funcType>& data, const eigenMatrix & meshX, const eigenMatrix& meshY, const unsigned mpi_rank, const unsigned mpi_size){
    const unsigned last = data.n - 1;
    const double h = abs(data.x2 - data.x1) / (data.n - 1);
    const double den = 1 + h*data.gamma; // Denominator for the Robin condition update formula

    if constexpr(execution_mode == ExecutionMode::SEQUENTIAL){
         for (unsigned i = 0; i < data.n; ++i) {
                u_h(i, 0) = (u_h(i, 1) + h * data.f4(meshX(i, 0), meshY(i, 0))) / den; // Left edge
                u_h(i, last) = (u_h(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last))) / den;  // Right edge 
                u_h(0, i) = (u_h(1, i) + h * data.f3(meshX(0, i), meshY(0, i))) / den; // Top edge 
                u_h(last, i) = (u_h(last - 1, i) + h * data.f1(meshX(last, i), meshY(last, i))) / den; // Bottom edge 
        }
    }

    if constexpr(execution_mode == ExecutionMode::PARALLEL){
        // As with Dirichlet, each process will update the boundary conditions of its own subdomain -> maybe the work is not well balanced?

        // Bottom edge
        if(mpi_rank == 0) for(unsigned i = 0; i < data.n; i++) u_h(last, i) = (u_h(last - 1, i) + h * data.f1(meshX(last, i), meshY(last, i))) / den;
        
        // Top edge
        if(mpi_rank == mpi_size - 1) for(unsigned i = 0; i < data.n; i++) u_h(0, i) = (u_h(1, i) + h * data.f3(meshX(0, i), meshY(0, i))) / den;

        // Left and right edges
        for(unsigned i = 0; i < u_h.rows(); i++){
            u_h(i, 0) = (u_h(i, 1) + h * data.f4(meshX(i, 0), meshY(i, 0))) / den;
            u_h(i, last) = (u_h(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last))) / den;
        }
    }
}

    /**
     * @brief General function for applying boundary conditions.
     * Available boundary conditions are Dirichlet, Neumann and Robin. They are available for both sequential and parallel execution modes, and the function will dispatch to the correct implementation based on the template parameters. 
     */

template <BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
void apply_boundary_condition(eigenMatrix & u_h, const Data_Struct<funcType>& data, const eigenMatrix & meshX, const eigenMatrix &meshY, unsigned mpi_rank = -1, unsigned mpi_size = -1){
    if constexpr(boundary_condition == BoundaryCondition::DIRICHLET) apply_dirichlet_condition<execution_mode, funcType>(u_h, data, meshX, meshY, mpi_rank, mpi_size);
    if constexpr(boundary_condition == BoundaryCondition::NEUMANN) apply_neumann_condition<execution_mode, funcType>(u_h, data, meshX, meshY, mpi_rank, mpi_size);
    if constexpr(boundary_condition == BoundaryCondition::ROBIN) apply_robin_condition<execution_mode, funcType>(u_h, data, meshX, meshY, mpi_rank, mpi_size);
}

#endif
