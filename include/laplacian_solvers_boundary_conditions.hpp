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
        
        // Bottom side
        if(mpi_rank == 0) {
            for(unsigned i = 0; i < data.n; i++) {
                u_h(0, i) = data.f1(meshX(0, i), meshY(0, i));
            }
        }
        
        // Top side
        if(mpi_rank == mpi_size - 1) {
            const unsigned local_last_row = u_h.rows() - 1; 
            const unsigned mesh_last_row = meshY.rows() - 1; 
            for(unsigned i = 0; i < data.n; i++) {
                u_h(local_last_row, i) = data.f3(meshX(mesh_last_row, i), meshY(mesh_last_row, i));
            }
        }
        
        // Left - right sides
        // Aways exclude first and last rows
        for(unsigned i = 1; i <= local_rows; i++){
            //std::cout << i << " " << u_h.rows() <<std::endl;
            u_h(i, 0) = data.f4(meshX(i - 1, 0), meshY(i - 1, 0));
            u_h(i, last) = data.f2(meshX(i - 1, last), meshY(i - 1, last));
        }
    }
}

/**
     * @brief General function for Neumann boundary conditions. This covers both the homogeneous and non-homogeneus cases
     * The template function can select the execution policy (sequential or parallel). Notice that the compatibility condition is not checked
     */
template <ExecutionMode execution_mode, typename funcType>
void apply_neumann_condition(eigenMatrix& u_h, const Data_Struct<funcType>& data, const eigenMatrix& meshX, const eigenMatrix& meshY, const eigenMatrix& u_old, const unsigned mpi_rank, const unsigned mpi_size){
    const unsigned last = data.n - 1;
    const double h  = std::abs(data.x2 - data.x1) / (data.n - 1);
    const double h2 = h * h;

    if constexpr (execution_mode == ExecutionMode::SEQUENTIAL) {

        for (unsigned i = 1; i < last; i++) {
            // Bottom side
            u_h(0, i) = 0.25*(2*u_old(1,i) + u_old(0,i-1) + u_old(0,i+1) + h2*data.f0(meshX(0,i), meshY(0,i)) + 2*h*data.f1(meshX(0,i), meshY(0,i)));
            // Top side 
            u_h(last, i) = 0.25*(2*u_old(last-1,i) + u_old(last,i-1) + u_old(last,i+1) + h2*data.f0(meshX(last,i), meshY(last,i)) + 2*h*data.f3(meshX(last,i), meshY(last,i)));
            // Left side
            u_h(i, 0) = 0.25*(u_old(i-1,0) + u_old(i+1,0) + 2*u_old(i,1) + h2*data.f0(meshX(i,0), meshY(i,0)) + 2*h*data.f4(meshX(i,0), meshY(i,0)));
            // Right side
            u_h(i, last) = 0.25*(u_old(i-1,last) + u_old(i+1,last) + 2*u_old(i,last-1) + h2*data.f0(meshX(i,last), meshY(i,last)) + 2*h*data.f2(meshX(i,last), meshY(i,last)));
        }

        // Corners
        u_h(0, 0) = 0.25*(2*u_old(1,0) + 2*u_old(0,1) + h2*data.f0(meshX(0,0), meshY(0,0)) + 2*h*(data.f1(meshX(0,0), meshY(0,0)) + data.f4(meshX(0,0), meshY(0,0))));
        u_h(0, last) = 0.25*(2*u_old(1,last) + 2*u_old(0,last-1) + h2*data.f0(meshX(0,last), meshY(0,last)) + 2*h*(data.f1(meshX(0,last), meshY(0,last)) + data.f2(meshX(0,last), meshY(0,last))));
        u_h(last, 0) = 0.25*(2*u_old(last-1,0) + 2*u_old(last,1) + h2*data.f0(meshX(last,0), meshY(last,0)) + 2*h*(data.f3(meshX(last,0), meshY(last,0)) + data.f4(meshX(last,0), meshY(last,0))));
        u_h(last, last) = 0.25*(2*u_old(last-1,last) + 2*u_old(last,last-1) + h2*data.f0(meshX(last,last), meshY(last,last)) + 2*h*(data.f3(meshX(last,last), meshY(last,last)) + data.f2(meshX(last,last), meshY(last,last))));

    } else if constexpr (execution_mode == ExecutionMode::PARALLEL) {

        // Compute indexes
        const unsigned up_row = (mpi_rank == 0)? 0 : 1;
        const unsigned down_row = (mpi_rank == mpi_size - 1) ? 0 : 1;
        const unsigned real_first = up_row;
        const unsigned real_last = u_h.rows() - 1 - down_row;

        // Left - right sides without the corners
        for (unsigned i = real_first; i <= real_last; i++) {
            const bool is_global_bottom = (mpi_rank == 0 && i == real_first);
            const bool is_global_top = (mpi_rank == mpi_size - 1 && i == real_last);
            if (is_global_bottom || is_global_top) continue;

            const unsigned mi = i - up_row;
            u_h(i, 0) = 0.25*(u_old(i-1,0) + u_old(i+1,0) + 2*u_old(i,1) + h2*data.f0(meshX(mi,0), meshY(mi,0)) + 2*h*data.f4(meshX(mi,0), meshY(mi,0)));
            u_h(i, last) = 0.25*(u_old(i-1,last)  + u_old(i+1,last) + 2*u_old(i,last-1) + h2*data.f0(meshX(mi,last), meshY(mi,last)) + 2*h*data.f2(meshX(mi,last), meshY(mi,last)));
        }

        // Bottom side + corner
        if (mpi_rank == 0) {
            for (unsigned j = 1; j < last; j++)
                u_h(0, j) = 0.25*(2*u_old(1,j) + u_old(0,j-1) + u_old(0,j+1) + h2*data.f0(meshX(0,j), meshY(0,j)) + 2*h*data.f1(meshX(0,j), meshY(0,j)));

            u_h(0, 0) = 0.25*(2*u_old(1,0) + 2*u_old(0,1) + h2*data.f0(meshX(0,0), meshY(0,0)) + 2*h*(data.f1(meshX(0,0), meshY(0,0)) + data.f4(meshX(0,0), meshY(0,0))));
            u_h(0, last) = 0.25*(2*u_old(1,last) + 2*u_old(0,last-1) + h2*data.f0(meshX(0,last), meshY(0,last)) + 2*h*(data.f1(meshX(0,last), meshY(0,last)) + data.f2(meshX(0,last), meshY(0,last))));
        }

        // Top side + corner
        if (mpi_rank == mpi_size - 1) {
            const unsigned mt = real_last - up_row;
            for (unsigned j = 1; j < last; j++)
                u_h(real_last, j) = 0.25*(2*u_old(real_last-1,j) + u_old(real_last,j-1) + u_old(real_last,j+1) + h2*data.f0(meshX(mt,j), meshY(mt,j)) + 2*h*data.f3(meshX(mt,j), meshY(mt,j)));

            u_h(real_last, 0) = 0.25*(2*u_old(real_last-1,0) + 2*u_old(real_last,1) + h2*data.f0(meshX(mt,0), meshY(mt,0))+ 2*h*(data.f3(meshX(mt,0), meshY(mt,0)) + data.f4(meshX(mt,0), meshY(mt,0))));
            u_h(real_last, last) = 0.25*(2*u_old(real_last-1,last) + 2*u_old(real_last,last-1) + h2*data.f0(meshX(mt,last), meshY(mt,last)) + 2*h*(data.f3(meshX(mt,last), meshY(mt,last)) + data.f2(meshX(mt,last), meshY(mt,last))));
        }
    }
}

/**
     * @brief General function for Robin boundary conditions. This covers both the homogeneous and non-homogeneus cases
     * The template function can select the execution policy (sequential or parallel)
     */

template <ExecutionMode execution_mode, typename funcType>
void apply_robin_condition(eigenMatrix & u_h, const Data_Struct<funcType>& data, const eigenMatrix & meshX, const eigenMatrix& meshY, const eigenMatrix & u_old, const unsigned mpi_rank, const unsigned mpi_size){
    
    // General variables
    const unsigned last = data.n - 1;
    const double h = std::abs(data.x2 - data.x1) / (data.n - 1);
    const double h2 = h * h;

    // Denominators
    const double den_edge = 4.0 + 2.0 * h * data.alpha;
    const double den_corner = 4.0 + 4.0 * h * data.alpha;

    if constexpr(execution_mode == ExecutionMode::SEQUENTIAL){
        // Left - right sides without the corners
        for (unsigned i = 1; i < last; ++i) {
            u_h(i, 0)    = (u_old(i-1, 0) + u_old(i+1, 0) + 2.0 * u_old(i, 1) + h2 * data.f0(meshX(i, 0), meshY(i, 0)) + 2.0 * h * data.f4(meshX(i, 0), meshY(i, 0))) / den_edge;
            u_h(i, last) = (u_old(i-1, last) + u_old(i+1, last) + 2.0 * u_old(i, last-1) + h2 * data.f0(meshX(i, last), meshY(i, last)) + 2.0 * h * data.f2(meshX(i, last), meshY(i, last))) / den_edge;
        }

        // Bottom - top sides without the corners
        for (unsigned j = 1; j < last; ++j) {
            u_h(0, j)    = (2.0 * u_old(1, j) + u_old(0, j-1) + u_old(0, j+1) + h2 * data.f0(meshX(0, j), meshY(0, j)) + 2.0 * h * data.f1(meshX(0, j), meshY(0, j))) / den_edge;
            u_h(last, j) = (2.0 * u_old(last-1, j) + u_old(last, j-1) + u_old(last, j+1) + h2 * data.f0(meshX(last, j), meshY(last, j)) + 2.0 * h * data.f3(meshX(last, j), meshY(last, j))) / den_edge;
        }

        // Corners
        u_h(0, 0) = (2.0 * u_old(1, 0) + 2.0 * u_old(0, 1) + h2 * data.f0(meshX(0,0), meshY(0,0)) + 2.0 * h * (data.f1(meshX(0,0), meshY(0,0)) + data.f4(meshX(0,0), meshY(0,0)))) / den_corner;
        u_h(0, last) = (2.0 * u_old(1, last) + 2.0 * u_old(0, last-1) + h2 * data.f0(meshX(0,last), meshY(0,last)) + 2.0 * h * (data.f1(meshX(0,last), meshY(0,last)) + data.f2(meshX(0,last), meshY(0,last)))) / den_corner;
        u_h(last, 0) = (2.0 * u_old(last-1, 0) + 2.0 * u_old(last, 1) + h2 * data.f0(meshX(last,0), meshY(last,0)) + 2.0 * h * (data.f3(meshX(last,0), meshY(last,0)) + data.f4(meshX(last,0), meshY(last,0)))) / den_corner;
        u_h(last, last) = (2.0 * u_old(last-1, last) + 2.0 * u_old(last, last-1) + h2 * data.f0(meshX(last,last), meshY(last,last)) + 2.0 * h * (data.f3(meshX(last,last), meshY(last,last)) + data.f2(meshX(last,last), meshY(last,last)))) / den_corner;
    }
    
    if constexpr(execution_mode == ExecutionMode::PARALLEL){

        // Compute indexes
        const unsigned remainder_rows = data.n % mpi_size;
        const unsigned local_rows = data.n / mpi_size + (mpi_rank < remainder_rows ? 1 : 0);

        const unsigned up_row = (mpi_rank == 0 ? 0 : 1);
        const unsigned down_row = (mpi_rank == mpi_size - 1 ? 0 : 1);
        
        const unsigned first_real_row = up_row;
        const unsigned last_real_row = u_h.rows() - 1 - down_row;

        // Left - right sides without corners
        for(unsigned i = first_real_row; i <= last_real_row; i++){
            const bool is_global_bottom = (mpi_rank == 0 && i == first_real_row);
            const bool is_global_top    = (mpi_rank == mpi_size - 1 && i == last_real_row);
            if (is_global_bottom || is_global_top) continue;

            unsigned mesh_i = i - up_row; 
            
            u_h(i, 0)    = (u_old(i-1, 0) + u_old(i+1, 0) + 2.0 * u_old(i, 1) + h2 * data.f0(meshX(mesh_i, 0), meshY(mesh_i, 0)) + 2.0 * h * data.f4(meshX(mesh_i, 0), meshY(mesh_i, 0))) / den_edge;
            u_h(i, last) = (u_old(i-1, last) + u_old(i+1, last) + 2.0 * u_old(i, last-1) + h2 * data.f0(meshX(mesh_i, last), meshY(mesh_i, last)) + 2.0 * h * data.f2(meshX(mesh_i, last), meshY(mesh_i, last))) / den_edge;
        }

        
        if(mpi_rank == 0) {
            // Bottom side
            for(unsigned j = 1; j < last; j++) u_h(0, j) = (2.0 * u_old(1, j) + u_old(0, j-1) + u_old(0, j+1) + h2 * data.f0(meshX(0, j), meshY(0, j)) + 2.0 * h * data.f1(meshX(0, j), meshY(0, j))) / den_edge;
            // Left - bottom corner
            u_h(0, 0) = (2.0 * u_old(1, 0) + 2.0 * u_old(0, 1) + h2 * data.f0(meshX(0,0), meshY(0,0)) + 2.0 * h * (data.f1(meshX(0,0), meshY(0,0)) + data.f4(meshX(0,0), meshY(0,0)))) / den_corner;
            // Right - bottom corner
            u_h(0, last) = (2.0 * u_old(1, last) + 2.0 * u_old(0, last-1) + h2 * data.f0(meshX(0,last), meshY(0,last)) + 2.0 * h * (data.f1(meshX(0,last), meshY(0,last)) + data.f2(meshX(0,last), meshY(0,last)))) / den_corner;
        }
        
        // Top side
        if(mpi_rank == mpi_size - 1) {
            const unsigned local_last_row = u_h.rows() - 1;
            const unsigned mesh_last_row = local_rows - 1;
            
            // Top side
            for(unsigned j = 1; j < last; j++) u_h(local_last_row, j) = (2.0 * u_old(local_last_row - 1, j) + u_old(local_last_row, j-1) + u_old(local_last_row, j+1) + h2 * data.f0(meshX(mesh_last_row, j), meshY(mesh_last_row, j)) + 2.0 * h * data.f3(meshX(mesh_last_row, j), meshY(mesh_last_row, j))) / den_edge;
            // Left - top corner
            u_h(local_last_row, 0) = (2.0 * u_old(local_last_row-1, 0) + 2.0 * u_old(local_last_row, 1) + h2 * data.f0(meshX(mesh_last_row,0), meshY(mesh_last_row,0)) + 2.0 * h * (data.f3(meshX(mesh_last_row,0), meshY(mesh_last_row,0)) + data.f4(meshX(mesh_last_row,0), meshY(mesh_last_row,0)))) / den_corner;
            // Right - top corner
            u_h(local_last_row, last) = (2.0 * u_old(local_last_row-1, last) + 2.0 * u_old(local_last_row, last-1) + h2 * data.f0(meshX(mesh_last_row,last), meshY(mesh_last_row,last)) + 2.0 * h * (data.f3(meshX(mesh_last_row,last), meshY(mesh_last_row,last)) + data.f2(meshX(mesh_last_row,last), meshY(mesh_last_row,last)))) / den_corner;
        }
    }
}

    /**
     * @brief General function for applying boundary conditions.
     * Available boundary conditions are Dirichlet, Neumann and Robin. They are available for both sequential and parallel execution modes, and the function will dispatch to the correct implementation based on the template parameters. 
     */

template <BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
void apply_boundary_condition(eigenMatrix & u_h, const Data_Struct<funcType>& data, const eigenMatrix & meshX, const eigenMatrix &meshY, const eigenMatrix & u_old = {}, unsigned mpi_rank = 0, unsigned mpi_size = 1){
    if constexpr(boundary_condition == BoundaryCondition::DIRICHLET) apply_dirichlet_condition<execution_mode, funcType>(u_h, data, meshX, meshY, mpi_rank, mpi_size);
    if constexpr(boundary_condition == BoundaryCondition::NEUMANN) apply_neumann_condition<execution_mode, funcType>(u_h, data, meshX, meshY, u_old, mpi_rank, mpi_size);
    if constexpr(boundary_condition == BoundaryCondition::ROBIN) apply_robin_condition<execution_mode, funcType>(u_h, data, meshX, meshY, u_old, mpi_rank, mpi_size);
}

#endif
