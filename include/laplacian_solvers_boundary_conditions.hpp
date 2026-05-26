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
        // Sequential execution: mesh and u_h share the exact same global dimensions
        for(unsigned i = 0; i < data.n; i++){
            u_h(i, 0) = data.f4(meshX(i, 0), meshY(i, 0)); // Left edge
            u_h(i, last) = data.f2(meshX(i, last), meshY(i, last)); // Right edge
            u_h(0, i) = data.f1(meshX(0, i), meshY(0, i)); // Bottom edge
            u_h(last, i) = data.f3(meshX(last, i), meshY(last, i)); // Top edge
        }

    } else if constexpr(execution_mode == ExecutionMode::PARALLEL){
        // Parallel execution: mesh spatial variables represent the local subdomain rows
        const unsigned local_rows = u_h.rows() - 2;

        // 1. Bottom physical boundary (Only managed by Rank 0)
        if(mpi_rank == 0) {
            for(unsigned i = 0; i < data.n; i++) {
                u_h(0, i) = data.f1(meshX(0, i), meshY(0, i));
            }
        }
        
        // 2. Top physical boundary (Only managed by the last Rank)
        if(mpi_rank == mpi_size - 1) {
            const unsigned local_last_row = u_h.rows() - 1; // Ghost layer index in u_h
            const unsigned mesh_last_row  = local_rows - 1; // Last available row in local mesh
            for(unsigned i = 0; i < data.n; i++) {
                u_h(local_last_row, i) = data.f3(meshX(mesh_last_row, i), meshY(mesh_last_row, i));
            }
        }
        
        // 3. Left and Right physical boundaries
        // Applied only to real local fluid rows (from 1 to local_rows)
        // Shifting index by -1 to properly sample the local mesh matrices
        for(unsigned i = 1; i <= local_rows; i++){
            u_h(i, 0)    = data.f4(meshX(i - 1, 0), meshY(i - 1, 0));
            u_h(i, last) = data.f2(meshX(i - 1, last), meshY(i - 1, last));
        }
    }
}

/**
     * @brief General function for Neumann boundary conditions. This covers both the homogeneous and non-homogeneus cases
     * The template function can select the execution policy (sequential or parallel). Notice that the compatibility condition is not checked
     */
template <ExecutionMode execution_mode, typename funcType>
void apply_neumann_condition(eigenMatrix & u_h, const Data_Struct<funcType>& data, const eigenMatrix & meshX, const eigenMatrix & meshY, const eigenMatrix& u_old, const unsigned mpi_rank, const unsigned mpi_size){
    
    const unsigned last = data.n - 1;
    const double h = abs(data.x2 - data.x1) / (data.n - 1);

    if constexpr(execution_mode == ExecutionMode::SEQUENTIAL){

         for (unsigned i = 1; i < last; i++) {
            u_h(i, 0)    = u_old(i, 1)        + h * data.f4(meshX(i, 0), meshY(i, 0));       // Left edge (f4)
            u_h(i, last) = u_old(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last)); // Right edge (f2)
            u_h(0, i)    = u_old(1, i)        + h * data.f1(meshX(0, i), meshY(0, i));       // Bottom edge (f1)
            u_h(last, i) = u_old(last - 1, i) + h * data.f3(meshX(last, i), meshY(last, i)); // Top edge (f3)
        }

        //Corners
        u_h(0, 0)       = 0.5*(u_old(1, 0)        + u_old(0, 1))        + 0.5*h*(data.f1(meshX(0,0),         meshY(0,0))         + data.f4(meshX(0,0),         meshY(0,0)));
        u_h(0, last)    = 0.5*(u_old(1, last)      + u_old(0, last-1))  + 0.5*h*(data.f1(meshX(0,last),      meshY(0,last))      + data.f2(meshX(0,last),      meshY(0,last)));
        u_h(last, 0)    = 0.5*(u_old(last-1, 0)   + u_old(last, 1))    + 0.5*h*(data.f3(meshX(last,0),      meshY(last,0))      + data.f4(meshX(last,0),      meshY(last,0)));
        u_h(last, last) = 0.5*(u_old(last-1, last) + u_old(last,last-1))+ 0.5*h*(data.f3(meshX(last,last),   meshY(last,last))   + data.f2(meshX(last,last),   meshY(last,last)));

    } else if constexpr(execution_mode == ExecutionMode::PARALLEL){

        // As with Dirichlet, each process will update the boundary conditions of its own subdomain -> maybe the work is not well balanced?

        const unsigned local_last_row = u_h.rows() - 1;
        // Left and right edges
        for(unsigned i = 1; i < local_last_row; i++){
            u_h(i, 0)    = u_old(i, 1) + h * data.f4(meshX(i, 0), meshY(i, 0));
            u_h(i, last) = u_old(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last));
        }
        
        // Bottom edge
        if(mpi_rank == 0) {
            for(unsigned i = 1; i < last; i++) {
                u_h(0, i) = u_old(1, i) + h * data.f1(meshX(0, i), meshY(0, i));
            }
            u_h(0, 0) = 0.5*(u_old(1, 0)    + u_old(0, 1))    + 0.5*h*(data.f1(meshX(0,0),    meshY(0,0))    + data.f4(meshX(0,0),    meshY(0,0)));
            u_h(0, last) = 0.5*(u_old(1, last) + u_old(0, last-1))+ 0.5*h*(data.f1(meshX(0,last), meshY(0,last)) + data.f2(meshX(0,last), meshY(0,last)));
        }

        // Top edge
        if(mpi_rank == mpi_size - 1) {
            for(unsigned i = 1; i < last; i++) {
                u_h(local_last_row, i) = u_old(local_last_row - 1, i) + h * data.f3(meshX(local_last_row, i), meshY(local_last_row, i));
            }
            u_h(local_last_row, 0)    = 0.5*(u_old(local_last_row-1, 0)    + u_old(local_last_row, 1))    + 0.5*h*(data.f3(meshX(local_last_row,0),    meshY(local_last_row,0))    + data.f4(meshX(local_last_row,0),    meshY(local_last_row,0)));
            u_h(local_last_row, last) = 0.5*(u_old(local_last_row-1, last) + u_old(local_last_row, last-1))+ 0.5*h*(data.f3(meshX(local_last_row,last), meshY(local_last_row,last)) + data.f2(meshX(local_last_row,last), meshY(local_last_row,last)));
        }
    }
}

/**
     * @brief General function for Robin boundary conditions. This covers both the homogeneous and non-homogeneus cases
     * The template function can select the execution policy (sequential or parallel)
     */

template <ExecutionMode execution_mode, typename funcType>
void apply_robin_condition(eigenMatrix & u_h, const Data_Struct<funcType>& data, const eigenMatrix & meshX, const eigenMatrix& meshY, const eigenMatrix & u_old, const unsigned mpi_rank, const unsigned mpi_size){
    const unsigned last = data.n - 1;
    const double h = abs(data.x2 - data.x1) / (data.n - 1);
    const double den = 1 + h*data.alpha; // Denominator for the Robin condition update formula

if constexpr(execution_mode == ExecutionMode::SEQUENTIAL){
         for (unsigned i = 0; i < data.n; ++i) {
                u_h(i, 0)    = (u_old(i, 1) + h * data.f4(meshX(i, 0), meshY(i, 0))) / den;       // Left edge (f4)
                u_h(i, last) = (u_old(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last))) / den;  // Right edge (f2)
                u_h(0, i)    = (u_old(1, i) + h * data.f1(meshX(0, i), meshY(0, i))) / den;       // Bottom edge (f1)
                u_h(last, i) = (u_old(last - 1, i) + h * data.f3(meshX(last, i), meshY(last, i))) / den; // Top edge (f3)
        }

        // Corners
        u_h(0, 0)       = 0.5 * ((u_old(1, 0) + h * data.f1(meshX(0,0), meshY(0,0))) / den + 
                                (u_old(0, 1) + h * data.f4(meshX(0,0), meshY(0,0))) / den);
                                
        u_h(0, last)    = 0.5 * ((u_old(1, last) + h * data.f1(meshX(0,last), meshY(0,last))) / den + 
                                (u_old(0, last-1) + h * data.f2(meshX(0,last), meshY(0,last))) / den);
                                
        u_h(last, 0)    = 0.5 * ((u_old(last-1, 0) + h * data.f3(meshX(last,0), meshY(last,0))) / den + 
                                (u_old(last, 1) + h * data.f4(meshX(last,0), meshY(last,0))) / den);
                                
        u_h(last, last) = 0.5 * ((u_old(last-1, last) + h * data.f3(meshX(last,last), meshY(last,last))) / den + 
                                (u_old(last, last-1) + h * data.f2(meshX(last,last), meshY(last,last))) / den);
    }
    
    if constexpr(execution_mode == ExecutionMode::PARALLEL){
        // As with Dirichlet, each process will update the boundary conditions of its own subdomain -> maybe the work is not well balanced?

        // Bottom edge
        if(mpi_rank == 0) {
            for(unsigned i = 0; i < data.n; i++) {
                u_h(0, i) = (u_old(1, i) + h * data.f1(meshX(0, i), meshY(0, i))) / den;
            }
        }
        
        if(mpi_rank == mpi_size - 1) {
            const unsigned local_last_row = u_h.rows() - 1;
            for(unsigned i = 0; i < data.n; i++) {
                u_h(local_last_row, i) = (u_old(local_last_row - 1, i) + h * data.f3(meshX(local_last_row, i), meshY(local_last_row, i))) / den;
            }
        }

        // Left and right edges
        for(unsigned i = 0; i < u_h.rows(); i++){
            u_h(i, 0)    = (u_old(i, 1) + h * data.f4(meshX(i, 0), meshY(i, 0))) / den;
            u_h(i, last) = (u_old(i, last - 1) + h * data.f2(meshX(i, last), meshY(i, last))) / den;
        }
    }
}

    /**
     * @brief General function for applying boundary conditions.
     * Available boundary conditions are Dirichlet, Neumann and Robin. They are available for both sequential and parallel execution modes, and the function will dispatch to the correct implementation based on the template parameters. 
     */

template <BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
void apply_boundary_condition(eigenMatrix & u_h, const Data_Struct<funcType>& data, const eigenMatrix & meshX, const eigenMatrix &meshY, const eigenMatrix & u_old = {}, unsigned mpi_rank = -1, unsigned mpi_size = -1){
    if constexpr(boundary_condition == BoundaryCondition::DIRICHLET) apply_dirichlet_condition<execution_mode, funcType>(u_h, data, meshX, meshY, mpi_rank, mpi_size);
    if constexpr(boundary_condition == BoundaryCondition::NEUMANN) apply_neumann_condition<execution_mode, funcType>(u_h, data, meshX, meshY, u_old, mpi_rank, mpi_size);
    if constexpr(boundary_condition == BoundaryCondition::ROBIN) apply_robin_condition<execution_mode, funcType>(u_h, data, meshX, meshY, u_old, mpi_rank, mpi_size);
}

#endif
