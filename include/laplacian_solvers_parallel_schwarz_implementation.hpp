#ifndef LAPLACIAN_SOLVERS_PARALLEL_SCHWARZ_IMPLEMENTATION_HPP
#define LAPLACIAN_SOLVERS_PARALLEL_SCHWARZ_IMPLEMENTATION_HPP

#include "laplacian_solvers.hpp"
#include "laplacian_solvers_boundary_conditions.hpp"
#include <iostream>
#include <cmath>
#include <vector>

/**
 * @file laplacian_solvers_parallel_schwarz_implementation.hpp
 * @brief Hybrid MPI+OpenMP parallel implementation of the Schwarz domain decomposition method.
 */

#define MAX_SCHWARZ_ITERATIONS 10

namespace laplacian_solvers{

    /**
     * @brief Solves the Laplacian problem using a parallel alternating Schwarz method.
     * * Divides the domain into subdomains allocated to different MPI ranks. The algorithm 
     * alternates between global macro-steps (inter-process halo updates via MPI) and local 
     * micro-steps (multi-threaded relaxation loops via OpenMP confined to the subdomains).
     * * @tparam solver_type Iterative algorithm selector.
     * @tparam boundary_condition Boundary condition policy.
     * @tparam execution_mode Parallel execution policy backend.
     * @tparam funcType Callable type for boundary and source terms.
     * * @return A @ref Result_Struct containing the assembled solution on Rank 0, or partial fields on other ranks.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::schwarz_parallel(){
        
        // Compute indexes and communication info
        const unsigned remainder_rows = data.n % mpi_size;
        const unsigned local_rows = data.n / mpi_size + (static_cast<unsigned>(mpi_rank) < remainder_rows ? 1 : 0); 

        // Indexes for meshX and meshY
        
        // Find process neighbors
        const int rank_up = (mpi_rank == 0) ? MPI_PROC_NULL : mpi_rank - 1; 
        const int rank_down = (mpi_rank == mpi_size - 1) ? MPI_PROC_NULL : mpi_rank + 1;

        const unsigned up_row = (mpi_rank == 0 ? 0 : 1);
        const unsigned down_row = (mpi_rank == mpi_size - 1 ? 0 : 1);
        const unsigned total_rows = local_rows + up_row + down_row;

        // Build local matrixes
        eigenMatrix u_h_local = eigenMatrix::Zero(total_rows, data.n);

        // Apply boundary conditions to the initial guess
        if constexpr(boundary_condition == BoundaryCondition::DIRICHLET || boundary_condition == BoundaryCondition::ROBIN) 
            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_local, data, meshX, meshY, u_h_local, mpi_rank, mpi_size);

        // Initialize loop variables
        eigenMatrix u_h_new_local(u_h_local);
        eigenMatrix u_h_old_global_step(u_h_local);
        double global_err = data.tolerance + 1.0;
        unsigned global_iter = 0;
        const double h2 = h * h;
            
        // Global Schwarz iteration loop

        #pragma omp parallel
        {
            while(global_err > data.tolerance && global_iter < data.max_iterations){
                #pragma omp single
                {
                    // 1. Handle communication with other processes (Exchange Ghost Cells)
                    MPI_Sendrecv(u_h_local.data() + up_row * data.n, data.n, MPI_DOUBLE, rank_up, 0,
                            u_h_local.data() + (total_rows - 1) * data.n, data.n, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Sendrecv(u_h_local.data() + (total_rows - 1 - down_row) * data.n, data.n, MPI_DOUBLE, rank_down, 1,
                            u_h_local.data(), data.n, MPI_DOUBLE, rank_up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
                    // Save state for global error computation later
                    u_h_old_global_step = u_h_local;        
                }


            // Local Schwarz iterations
                for(unsigned local_iter = 0; local_iter < MAX_SCHWARZ_ITERATIONS; ++local_iter) {
                
                    #pragma omp for collapse(2)
                        for(unsigned i = 1; i < total_rows - 1; i++){
                            for(unsigned j = 1; j < data.n - 1; j++){
                                u_h_new_local(i, j) = 0.25 * (u_h_local(i-1, j) + u_h_local(i+1, j) + u_h_local(i, j-1) + u_h_local(i, j+1) + h2 * data.f0(meshX(i - up_row, j), meshY(i - up_row, j)));         
                             }
                         }
                
                    #pragma omp single
                    {
                        // Preserve ghost information - only rows, as columns undergo BC assignement for every processor
                        u_h_new_local.row(0) = u_h_local.row(0);
                        u_h_new_local.row(total_rows - 1) = u_h_local.row(total_rows - 1);
                        //u_h_new_local.col(0) = u_h_local.col(0);
                        //u_h_new_local.col(data.n - 1) = u_h_local.col(data.n - 1);
                
                        // Apply Neumann boundary conditions locally
                        if constexpr(boundary_condition == BoundaryCondition::NEUMANN || boundary_condition == BoundaryCondition::ROBIN) 
                            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_new_local, data, meshX, meshY, u_h_local, mpi_rank, mpi_size);

                        u_h_local.swap(u_h_new_local);
                    }
                }
                // ====================================================================
                #pragma omp single
                {   
                // 2. Apply zero-average constraint GLOBALLY at the macro-step
                    if constexpr(boundary_condition == BoundaryCondition::NEUMANN){
                        double avg = 0.0;
                        const double local_sum = u_h_local.block(up_row, 0, local_rows, data.n).sum();
                        MPI_Allreduce(&local_sum, &avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                        // Shift the entire local domain to enforce uniqueness
                        u_h_local.array() -= avg / (data.n * data.n);
                    }

            
                    // 3. Global error computation based on real rows ONLY
                    const double macro_local_err = (u_h_local.block(up_row, 0, local_rows, data.n) - 
                                                u_h_old_global_step.block(up_row, 0, local_rows, data.n)).squaredNorm();
            
                    MPI_Allreduce(&macro_local_err, &global_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
                    global_err = std::sqrt(h * global_err); 
            
                    global_iter++;       
            
                }// Single
            }// While
        }// Parallel
        
        // Prepare communication for Gather
        std::vector<int> recv_counts(mpi_size);
        std::vector<int> displs(mpi_size);

        for(int i = 0; i < mpi_size; i++){
            recv_counts[i] = data.n * (data.n / mpi_size + (static_cast<unsigned>(i) < remainder_rows ? 1 : 0));
            displs[i] = i == 0 ? 0 : displs[i-1] + recv_counts[i-1];
        }

        // Assemble global solution and mesh
        eigenMatrix u_h_global(data.n, data.n);
        eigenMatrix meshX_global(data.n, data.n);
        eigenMatrix meshY_global(data.n, data.n);

        MPI_Gatherv(u_h_local.data() + up_row * data.n, local_rows * data.n, MPI_DOUBLE, u_h_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(meshX.data(), local_rows * data.n, MPI_DOUBLE, meshX_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(meshY.data(), local_rows * data.n, MPI_DOUBLE, meshY_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Return results
        result.u_h.swap(u_h_global);
        result.X.swap(meshX_global);
        result.Y.swap(meshY_global);
        result.iterations = global_iter;
        result.iteration_residue = global_err;
        if(mpi_rank == 0) result.valid = true;

        return result;
    }
}// namespace laplacian_solvers

#endif // LAPLACIAN_SOLVERS_PARALLEL_SCHWARZ_IMPLEMENTATION_HPP
