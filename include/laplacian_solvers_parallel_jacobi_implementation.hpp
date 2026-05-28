#ifndef LAPLACIAN_SOLVERS_PARALLEL_JACOBI_IMPLEMENTATION_HPP
#define LAPLACIAN_SOLVERS_PARALLEL_JACOBI_IMPLEMENTATION_HPP

#include "laplacian_solvers.hpp"
#include "laplacian_solvers_boundary_conditions.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

namespace laplacian_solvers{

    /**
     * @brief Solves the Laplacian using the parallel Jacobi method with Dirichlet BCs.
     * Uses MPI for domain decomposition and rows exchange. Each process updates its own 
     * strip of the matrix. MPI_Sendrecv ensures the first and last rows are exchanged between 
     * neighbors before each computation step. OpenMP is used for multi-threading within each node.
     */
    /*
        template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
        Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_parallel_dirichlet(){
            
            // Get mpi info
            int mpi_rank, mpi_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            
            // Compute indexes and communication info
            const unsigned remainder_rows = data.n % mpi_size;
            const unsigned local_rows = data.n / mpi_size + (mpi_rank < remainder_rows ? 1 : 0); 

            // Indexes for meshX and meshY
            const unsigned start_row = mpi_rank * (data.n / mpi_size) + std::min(static_cast<unsigned>(mpi_rank), remainder_rows);
            const unsigned end_row = start_row + local_rows - 1;

            // Find process neighbors
            const int rank_up = (mpi_rank == 0) ? MPI_PROC_NULL : mpi_rank - 1; 
            const int rank_down = (mpi_rank == mpi_size - 1) ? MPI_PROC_NULL : mpi_rank + 1;

            const unsigned up_row = (mpi_rank == 0? 0: 1);
            const unsigned down_row = (mpi_rank == mpi_size -1? 0: 1);
            const unsigned total_rows = local_rows + up_row + down_row;

            // Build local matrixes. It will also host the upper and lower rows for later communication
            eigenMatrix u_h_local = eigenMatrix::Zero(total_rows, data.n);
            eigenMatrix u_h_new_local = eigenMatrix::Zero(total_rows, data.n);

            // Initialize loop
            double err = data.tolerance + 1.0;
            unsigned iter = 0;
            const double h2 = h * h;

            // Apply boundary contitions to the initial guess

            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_local, data, meshX, meshY, u_h_local, mpi_rank, mpi_size);
            
            while(err > data.tolerance && iter < data.max_iterations){
                // First - Handle communication with other processes

                // Send first row and receive last row
                MPI_Sendrecv(u_h_local.data() + up_row * data.n, data.n, MPI_DOUBLE, rank_up, 0,
                            u_h_local.data() + (total_rows - 1) * data.n, data.n, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Send last row and receive first row
                MPI_Sendrecv(u_h_local.data() + (total_rows - 1 - down_row) * data.n, data.n, MPI_DOUBLE, rank_down, 1,
                            u_h_local.data(), data.n, MPI_DOUBLE, rank_up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                
                
                // Second - Compute new values, but not the received ones!
                #pragma omp parallel for
                for(unsigned i = 1; i < total_rows - 1; i++){
                    for(unsigned j = 1; j < data.n - 1; j++){
                        u_h_new_local(i, j) = 0.25 * (u_h_local(i-1, j) + u_h_local(i+1, j) + u_h_local(i, j-1) + u_h_local(i, j+1) + h2 * data.f0(meshX(i - up_row, j), meshY(i - up_row, j)));         
                    }
                }

                // Third - Apply boundary conditions 
                // apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_new_local, data, meshX, meshY, u_h_local, mpi_rank, mpi_size); // This is probably wrong

                // Fourth - zero-average constraint for Neumann BCs (required to ensure uniqueness of the solution)
                /*
                double avg = 0.0;
                const double local_sum = u_h_new_local.block(up_row, 0, local_rows, data.n).sum();
                MPI_Allreduce(&local_sum, &avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                u_h_new_local.array() -= avg / (data.n * data.n); */
    /*
                // Fourth - Compute local error in L2 norm (frobenius norm) with h
                const double local_err = (u_h_new_local - u_h_local).squaredNorm();
                
                // Fifth - Compute global error and prepare next iteration
                MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
                
                err = std::sqrt(h * err); // Maybe it is possible to avoid the sqrt?
                iter++;          
                u_h_local.swap(u_h_new_local);
                

            }
            
            // Prepare communication 
            std::vector<int> recv_counts(mpi_size);
            std::vector<int> displs(mpi_size);

            for(unsigned i = 0; i < mpi_size; i++){
                recv_counts[i] = data.n * (data.n / mpi_size + (i < remainder_rows? 1: 0));
                displs[i] = i == 0? 0: displs[i-1] + recv_counts[i-1];
            }

            // Assemble global solution and mesh
            eigenMatrix u_h_global(data.n, data.n);
            eigenMatrix meshX_global(data.n, data.n);
            eigenMatrix meshY_global(data.n, data.n);


            MPI_Gatherv(u_h_local.data() + up_row * data.n, local_rows * data.n, MPI_DOUBLE, u_h_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(meshX.data(), local_rows * data.n, MPI_DOUBLE, meshX_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(meshY.data(), local_rows * data.n, MPI_DOUBLE, meshY_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Return results - caution: only rank 0 has the correct return structure!
            result.u_h.swap(u_h_global);
            result.X.swap(meshX_global);
            result.Y.swap(meshY_global);
            result.iterations = iter;
            result.iterartion_residue = err;

            return result;
    }*/
    /*
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_parallel_neumann(){
        
        // Get mpi info
        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        // Compute indexes and communication info
        const unsigned remainder_rows = data.n % mpi_size;
        const unsigned local_rows = data.n / mpi_size + (mpi_rank < remainder_rows ? 1 : 0); 

        // Indexes for meshX and meshY
        const unsigned start_row = mpi_rank * (data.n / mpi_size) + std::min(static_cast<unsigned>(mpi_rank), remainder_rows);
        const unsigned end_row = start_row + local_rows - 1;

        // Find process neighbors
        const int rank_up = (mpi_rank == 0) ? MPI_PROC_NULL : mpi_rank - 1; 
        const int rank_down = (mpi_rank == mpi_size - 1) ? MPI_PROC_NULL : mpi_rank + 1;

        const unsigned up_row = (mpi_rank == 0? 0: 1);
        const unsigned down_row = (mpi_rank == mpi_size -1? 0: 1);
        const unsigned total_rows = local_rows + up_row + down_row;

        // Build local matrixes. It will also host the upper and lower rows for later communication
        eigenMatrix u_h_local = eigenMatrix::Zero(total_rows, data.n);
        eigenMatrix u_h_new_local = eigenMatrix::Zero(total_rows, data.n);

        // Initialize loop
        double err = data.tolerance + 1.0;
        unsigned iter = 0;
        const double h2 = h * h;
        
        while(err > data.tolerance && iter < data.max_iterations){
            // First - Handle communication with other processed

            // Send first row and receive last row
            MPI_Sendrecv(u_h_local.data() + up_row * data.n, data.n, MPI_DOUBLE, rank_up, 0,
                         u_h_local.data() + (total_rows - 1) * data.n, data.n, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Send last row and receive first row
            MPI_Sendrecv(u_h_local.data() + (total_rows - 1 - down_row) * data.n, data.n, MPI_DOUBLE, rank_down, 1,
                         u_h_local.data(), data.n, MPI_DOUBLE, rank_up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Second - Compute new values, but not the received ones!
            #pragma omp parallel for
            for(unsigned i = 1; i < total_rows - 1; i++){
                for(unsigned j = 1; j < data.n - 1; j++){
                    u_h_new_local(i, j) = 0.25 * (u_h_local(i-1, j) + u_h_local(i+1, j) + u_h_local(i, j-1) + u_h_local(i, j+1) + h2 * data.f0(meshX(i - up_row, j), meshY(i - up_row, j)));         
                }
            }

            // Third - Apply boundary conditions 
            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_new_local, data, meshX, meshY, u_h_local, mpi_rank, mpi_size); // This is probably wrong

            // Fourth - zero-average constraint for Neumann BCs (required to ensure uniqueness of the solution)
            double avg = 0.0;
            const double local_sum = u_h_new_local.block(up_row, 0, local_rows, data.n).sum();
            MPI_Allreduce(&local_sum, &avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            u_h_new_local.array() -= avg / (data.n * data.n);

            // Fourth - Compute local error in L2 norm (frobenius norm) with h
            const double local_err = (u_h_new_local - u_h_local).squaredNorm();

            // Fifth - Compute global error and prepare next iteration
            MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
            err = std::sqrt(h * err); // Maybe it is possible to avoid the sqrt?
            iter++;          
            u_h_local.swap(u_h_new_local);

        }
        
        // Prepare communication 
        std::vector<int> recv_counts(mpi_size);
        std::vector<int> displs(mpi_size);

        for(unsigned i = 0; i < mpi_size; i++){
            recv_counts[i] = data.n * (data.n / mpi_size + (i < remainder_rows? 1: 0));
            displs[i] = i == 0? 0: displs[i-1] + recv_counts[i-1];
        }

        // Assemble global solution and mesh
        eigenMatrix u_h_global(data.n, data.n); // Maybe initialize result.u_h?
        eigenMatrix meshX_global(data.n, data.n);
        eigenMatrix meshY_global(data.n, data.n);


        MPI_Gatherv(u_h_local.data() + up_row * data.n, local_rows * data.n, MPI_DOUBLE, u_h_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(meshX.data(), local_rows * data.n, MPI_DOUBLE, meshX_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(meshY.data(), local_rows * data.n, MPI_DOUBLE, meshY_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Return results - caution: only rank 0 has the correct return structure!
        result.u_h.swap(u_h_global);
        result.X.swap(meshX_global);
        result.Y.swap(meshY_global);
        result.iterations = iter;
        result.iterartion_residue = err;

        return result;
    }*/
    /*
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_parallel_robin(){

        // Get mpi info
        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        // Compute indexes and communication info
        const unsigned remainder_rows = data.n % mpi_size;
        const unsigned local_rows = data.n / mpi_size + (mpi_rank < remainder_rows ? 1 : 0); 

        // Indexes for meshX and meshY
        const unsigned start_row = mpi_rank * (data.n / mpi_size) + std::min(static_cast<unsigned>(mpi_rank), remainder_rows);
        const unsigned end_row = start_row + local_rows - 1;

        // Find process neighbors
        const int rank_up = (mpi_rank == 0) ? MPI_PROC_NULL : mpi_rank - 1; 
        const int rank_down = (mpi_rank == mpi_size - 1) ? MPI_PROC_NULL : mpi_rank + 1;

        const unsigned up_row = (mpi_rank == 0? 0: 1);
        const unsigned down_row = (mpi_rank == mpi_size -1? 0: 1);
        const unsigned total_rows = local_rows + up_row + down_row;

        // Build local matrixes. It will also host the upper and lower rows for later communication
        eigenMatrix u_h_local = eigenMatrix::Zero(total_rows, data.n);
        eigenMatrix u_h_new_local = eigenMatrix::Zero(total_rows, data.n);

        // Initialize loop
        double err = data.tolerance + 1.0;
        unsigned iter = 0;
        const double h2 = h * h;
        
        // Apply boundary contitions to the initial guess
        apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_local, data, meshX, meshY, u_h_local, mpi_rank, mpi_size);

        while(err > data.tolerance && iter < data.max_iterations){
            // First - Handle communication with other processed

            // Send first row and receive last row
            MPI_Sendrecv(u_h_local.data() + up_row * data.n, data.n, MPI_DOUBLE, rank_up, 0,
                         u_h_local.data() + (total_rows - 1) * data.n, data.n, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Send last row and receive first row
            MPI_Sendrecv(u_h_local.data() + (total_rows - 1 - down_row) * data.n, data.n, MPI_DOUBLE, rank_down, 1,
                         u_h_local.data(), data.n, MPI_DOUBLE, rank_up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Second - Compute new values, but not the received ones!
            #pragma omp parallel for
            for(unsigned i = 1; i < total_rows - 1; i++){
                for(unsigned j = 1; j < data.n - 1; j++){
                    u_h_new_local(i, j) = 0.25 * (u_h_local(i-1, j) + u_h_local(i+1, j) + u_h_local(i, j-1) + u_h_local(i, j+1) + h2 * data.f0(meshX(i - up_row, j), meshY(i - up_row, j)));         
                }
            }

            // Third - Apply boundary conditions 
            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_new_local, data, meshX, meshY, u_h_local, mpi_rank, mpi_size); // This is probably wrong

            // Fourth - zero-average constraint for Neumann BCs (required to ensure uniqueness of the solution)
            /*
            double avg = 0.0;
            const double local_sum = u_h_new_local.block(up_row, 0, local_rows, data.n).sum();
            MPI_Allreduce(&local_sum, &avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            u_h_new_local.array() -= avg / (data.n * data.n); */
    /*
            // Fourth - Compute local error in L2 norm (frobenius norm) with h
            const double local_err = (u_h_new_local - u_h_local).squaredNorm();
    
            // Fifth - Compute global error and prepare next iteration
            MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
            err = std::sqrt(h * err); // Maybe it is possible to avoid the sqrt?
            iter++;          
            u_h_local.swap(u_h_new_local);

        }
        
        // Prepare communication 
        std::vector<int> recv_counts(mpi_size);
        std::vector<int> displs(mpi_size);

        for(unsigned i = 0; i < mpi_size; i++){
            recv_counts[i] = data.n * (data.n / mpi_size + (i < remainder_rows? 1: 0));
            displs[i] = i == 0? 0: displs[i-1] + recv_counts[i-1];
        }

        // Assemble global solution and mesh
        eigenMatrix u_h_global(data.n, data.n); // Maybe initialize result.u_h?
        eigenMatrix meshX_global(data.n, data.n);
        eigenMatrix meshY_global(data.n, data.n);


        MPI_Gatherv(u_h_local.data() + up_row * data.n, local_rows * data.n, MPI_DOUBLE, u_h_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(meshX.data(), local_rows * data.n, MPI_DOUBLE, meshX_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(meshY.data(), local_rows * data.n, MPI_DOUBLE, meshY_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Return results - caution: only rank 0 has the correct return structure!
        result.u_h.swap(u_h_global);
        result.X.swap(meshX_global);
        result.Y.swap(meshY_global);
        result.iterations = iter;
        result.iterartion_residue = err;

        return result;
    }*/

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_parallel(){
        
        // Get mpi info
        /*int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);*/
        
        // Compute indexes and communication info
        const unsigned remainder_rows = data.n % mpi_size;
        const unsigned local_rows = data.n / mpi_size + (mpi_rank < remainder_rows ? 1 : 0); 

        // Indexes for meshX and meshY
        const unsigned start_row = mpi_rank * (data.n / mpi_size) + std::min(static_cast<unsigned>(mpi_rank), remainder_rows);
        const unsigned end_row = start_row + local_rows - 1;

        // Find process neighbors
        const int rank_up = (mpi_rank == 0) ? MPI_PROC_NULL : mpi_rank - 1; 
        const int rank_down = (mpi_rank == mpi_size - 1) ? MPI_PROC_NULL : mpi_rank + 1;

        const unsigned up_row = (mpi_rank == 0? 0: 1);
        const unsigned down_row = (mpi_rank == mpi_size -1? 0: 1);
        const unsigned total_rows = local_rows + up_row + down_row;

        // Build local matrixes. It will also host the upper and lower rows for later communication
        eigenMatrix u_h_local = eigenMatrix::Zero(total_rows, data.n);
        eigenMatrix u_h_new_local = eigenMatrix::Zero(total_rows, data.n);

        // Initialize loop
        double err = data.tolerance + 1.0;
        unsigned iter = 0;
        const double h2 = h * h;
        
        // Apply boundary contitions to the initial guess
        if constexpr(boundary_condition == BoundaryCondition::DIRICHLET || boundary_condition == BoundaryCondition::ROBIN) 
            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_local, data, meshX, meshY, u_h_local, mpi_rank, mpi_size);

        while(err > data.tolerance && iter < data.max_iterations){
            // First - Handle communication with other processed

            // Send first row and receive last row
            MPI_Sendrecv(u_h_local.data() + up_row * data.n, data.n, MPI_DOUBLE, rank_up, 0,
                         u_h_local.data() + (total_rows - 1) * data.n, data.n, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Send last row and receive first row
            MPI_Sendrecv(u_h_local.data() + (total_rows - 1 - down_row) * data.n, data.n, MPI_DOUBLE, rank_down, 1,
                         u_h_local.data(), data.n, MPI_DOUBLE, rank_up, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Second - Compute new values, but not the received ones!
            #pragma omp parallel for
            for(unsigned i = 1; i < total_rows - 1; i++){
                for(unsigned j = 1; j < data.n - 1; j++){
                    u_h_new_local(i, j) = 0.25 * (u_h_local(i-1, j) + u_h_local(i+1, j) + u_h_local(i, j-1) + u_h_local(i, j+1) + h2 * data.f0(meshX(i - up_row, j), meshY(i - up_row, j)));         
                }
            }

            // Third - Apply boundary conditions 

            if constexpr(boundary_condition == BoundaryCondition::NEUMANN || boundary_condition == BoundaryCondition::ROBIN) 
                apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_new_local, data, meshX, meshY, u_h_local, mpi_rank, mpi_size);

            // Fourth - zero-average constraint for Neumann BCs (required to ensure uniqueness of the solution)
            
            if constexpr(boundary_condition == BoundaryCondition::NEUMANN){
                double avg = 0.0;
                const double local_sum = u_h_new_local.block(up_row, 0, local_rows, data.n).sum();
                MPI_Allreduce(&local_sum, &avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                u_h_new_local.array() -= avg / (data.n * data.n); 
            }

            // Fourth - Compute local error in L2 norm (frobenius norm) with h
            const double local_err = (u_h_new_local - u_h_local).squaredNorm();

            // Fifth - Compute global error and prepare next iteration
            MPI_Allreduce(&local_err, &err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  
            err = std::sqrt(h * err); // Maybe it is possible to avoid the sqrt?
            iter++;          
            u_h_local.swap(u_h_new_local);

        }
        
        // Prepare communication 
        std::vector<int> recv_counts(mpi_size);
        std::vector<int> displs(mpi_size);

        for(unsigned i = 0; i < mpi_size; i++){
            recv_counts[i] = data.n * (data.n / mpi_size + (i < remainder_rows? 1: 0));
            displs[i] = i == 0? 0: displs[i-1] + recv_counts[i-1];
        }

        // Assemble global solution and mesh
        eigenMatrix u_h_global(data.n, data.n); // Maybe initialize result.u_h?
        eigenMatrix meshX_global(data.n, data.n);
        eigenMatrix meshY_global(data.n, data.n);


        MPI_Gatherv(u_h_local.data() + up_row * data.n, local_rows * data.n, MPI_DOUBLE, u_h_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(meshX.data(), local_rows * data.n, MPI_DOUBLE, meshX_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(meshY.data(), local_rows * data.n, MPI_DOUBLE, meshY_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Return results - caution: only rank 0 has the correct return structure!
        result.u_h.swap(u_h_global);
        result.X.swap(meshX_global);
        result.Y.swap(meshY_global);
        result.iterations = iter;
        result.iterartion_residue = err;

        return result;
    
    }
}


#endif