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

    /* --- CONSTRUCTOR --- */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename funcType>
    Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::Laplacian_Solver(const Data_Struct<funcType>& d) : data(d) {
        
        build_mesh();
        build_exact_solution();
        u_h = eigenMatrix::Zero(data.n, data.n);

    };


    /* --- SOLVE METHOD --- */

    //sequential or parallel?
    /**
     * @brief Primary dispatcher for the Laplacian solver.
     * Routes the execution flow to either sequential or parallel solvers based on the compile-time 
     * 'execution_mode' parameter using constexpr if-branching.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::solve(){

        if constexpr (execution_mode == ExecutionMode::SEQUENTIAL) {
            return sequential_solve();
        } else {
            return parallel_solve();
        }
        
    }

    // Jacobi or Schwarz?

    /**
     * @brief Dispatcher for sequential solver logic.
     * Routes to the Jacobi sequential implementation. Throws a std::runtime_error if Schwarz 
     * is selected, as it requires distributed-memory parallelism (MPI).
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::sequential_solve(){
        if constexpr (solver_type == SolverType::JACOBI) {
            return jacobi_sequential();
        } else {
            throw std::runtime_error("ERROR:Schwarz is a parallel solver.");
        }
    }

    /**
     * @brief Dispatcher for parallel solver logic.
     * Routes to either Jacobi parallel or Schwarz parallel implementations based on 
     * the compile-time 'solver_type' parameter.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::parallel_solve(){
        if constexpr (solver_type == SolverType::JACOBI) {
            return jacobi_parallel();
        } else {
            return schwarz_parallel();
        }
    }

    // DIRICHLET or NEUMANN or ROBIN?
    
    /**
     * @brief Dispatches the sequential Jacobi solver to the specific boundary condition implementation.
     * Maps the 'boundary_condition' template parameter to Dirichlet, Neumann, or Robin methods.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_sequential(){
        if constexpr (boundary_condition == BoundaryCondition::DIRICHLET) {
            return jacobi_sequential_dirichlet();
        } else if constexpr (boundary_condition == BoundaryCondition::NEUMANN) {
            return jacobi_sequential_neumann();
        } else {
            return jacobi_sequential_robin();
        }
    }

    /**
     * @brief Dispatches the parallel Jacobi solver to the specific boundary condition implementation.
     * Maps the 'boundary_condition' template parameter to Dirichlet, Neumann, or Robin methods.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_parallel(){
        if constexpr (boundary_condition == BoundaryCondition::DIRICHLET) {
            return jacobi_parallel_dirichlet();
        } else if constexpr (boundary_condition == BoundaryCondition::NEUMANN) {
            return jacobi_parallel_neumann();
        } else {
            return jacobi_parallel_robin();
        }
    }

    /**
     * @brief Dispatches the parallel Schwarz solver to the specific boundary condition implementation.
     * Routes the execution flow to the appropriate domain decomposition implementation.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::schwarz_parallel(){
        if constexpr (boundary_condition == BoundaryCondition::DIRICHLET) {
            return schwarz_parallel_dirichlet();
        } else if constexpr (boundary_condition == BoundaryCondition::NEUMANN) {
            return schwarz_parallel_neumann();
        } else {
            return schwarz_parallel_robin();
        }
    }

    /* --- ACTUAL SOLVERS JACOBI --- */
   

    /**
     * @brief Solves the Laplacian using the parallel Jacobi method with Dirichlet BCs.
     * Uses MPI for domain decomposition and rows exchange. Each process updates its own 
     * strip of the matrix. MPI_Sendrecv ensures the first and last rows are exchanged between 
     * neighbors before each computation step. OpenMP is used for multi-threading within each node.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_parallel_dirichlet(){
    
        //MPI initialization
        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        //Local rows
        unsigned local_rows = data.n / mpi_size;
        unsigned r = data.n % mpi_size;
        if (static_cast<unsigned>(mpi_rank) < r) {
            local_rows++;
        }

        //Global indexes for meshX and meshY
        unsigned start_row = static_cast<unsigned>(mpi_rank) * (data.n / mpi_size) + std::min(static_cast<unsigned>(mpi_rank), r);

        //Rank up and rank down for communication
        int rank_up = mpi_rank - 1;
        int rank_down = mpi_rank + 1;
        if (rank_up < 0) rank_up = MPI_PROC_NULL;
        if (rank_down >= mpi_size) rank_down = MPI_PROC_NULL;

        //Local matrices
        eigenMatrix u_h_local = eigenMatrix::Zero(local_rows + 2, data.n);
        eigenMatrix u_h_new_local = eigenMatrix::Zero(local_rows + 2, data.n); 
        eigenMatrix meshX_local = meshX.block(start_row, 0, local_rows, data.n);
        eigenMatrix meshY_local = meshY.block(start_row, 0, local_rows, data.n);

        apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_local, data, meshX, meshY, mpi_rank, mpi_size);

        u_h_local.block(1, 0, local_rows, data.n) = u_h.block(start_row, 0, local_rows, data.n);
        
        if (mpi_rank == 0) {
            u_h_local.row(0) = u_h.row(0);
        }
        if (mpi_rank == mpi_size - 1) {
            u_h_local.row(local_rows + 1) = u_h.row(data.n - 1);
        }
        u_h_new_local = u_h_local;

        // Cycle
        unsigned i_start = (mpi_rank == 0) ? 2 : 1;
        unsigned i_end = (mpi_rank == mpi_size - 1) ? local_rows - 1 : local_rows;
        unsigned iter = 0;
        double err = data.tolerance + 1.0;
        const double h2 = h * h;

        while (err > data.tolerance && iter < data.max_iterations){
            double loc_err = 0.0;
            MPI_Sendrecv(u_h_local.data() + 1 * data.n, data.n, MPI_DOUBLE, rank_up, 0, 
                         u_h_local.data() + 0 * data.n, data.n, MPI_DOUBLE, rank_up, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(u_h_local.data() + local_rows * data.n, data.n, MPI_DOUBLE, rank_down, 1, 
                         u_h_local.data() + (local_rows + 1) * data.n, data.n, MPI_DOUBLE, rank_down, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            #pragma omp parallel for reduction(max:loc_err)
            for (unsigned i = i_start; i <= i_end; ++i){
                for (unsigned j = 1; j < data.n - 1; ++j){
                    u_h_new_local(i, j) = 0.25 * (u_h_local(i-1, j) + u_h_local(i+1, j) + u_h_local(i, j-1) + u_h_local(i, j+1) + h2 * data.f0(meshX_local(i-1, j), meshY_local(i-1, j)));                    
                    double local_err = std::abs(u_h_new_local(i, j) - u_h_local(i, j));
                    if (local_err > loc_err) {
                        loc_err = local_err;
                    }
                }
            }
            //Gather results for next iteration
            MPI_Allreduce(&loc_err, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            u_h_local.swap(u_h_new_local);
            iter++;
        }

        //Gather final results
        std::vector<int> recv_counts(mpi_size);
        std::vector<int> displs(mpi_size);
        int send_elements = local_rows * data.n;
        MPI_Gather(&send_elements, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < mpi_size; ++i) {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
        }
        eigenMatrix u_h_gathered;
        if (mpi_rank == 0) {
            u_h_gathered.resize(data.n, data.n);
            u_h_gathered.row(0) = u_h.row(0);
            u_h_gathered.row(data.n - 1) = u_h.row(data.n - 1);
        }

        // Row-wise and Column-wise gather
        MPI_Gatherv(u_h_local.data() + 1 * data.n, send_elements, MPI_DOUBLE, 
                    u_h_gathered.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            u_h = u_h_gathered; 
        }

        //Return result
        Result_Struct result;
        result.u_h = u_h;
        result.iterations = iter;
        result.X = meshX;
        result.Y = meshY;
        return result;

    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_parallel_neumann(){
        // Implementazione del metodo di Jacobi parallelo con condizioni al contorno di Neumann
        return Result_Struct(); // Placeholder
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_parallel_robin(){
        // Implementazione del metodo di Jacobi parallelo con condizioni al contorno di Robin
        return Result_Struct(); // Placeholder
    }

    /* --- ACTUAL SOLVERS SCHWARZ --- */

    /**
     * @brief Solves the Laplacian using the additive Schwarz parallel method with Dirichlet BCs.
     * This method employs a domain decomposition strategy where each process solves a sub-problem 
     * to a local tolerance (relaxed for efficiency). After reaching local convergence, 
     * boundary values are synchronized. The global convergence is monitored via MPI_Allreduce 
     * to ensure consistency across the entire grid.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::schwarz_parallel_dirichlet(){
        //MPI initialization
        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        //Local rows
        unsigned local_rows = data.n / mpi_size;
        unsigned r = data.n % mpi_size;
        if (static_cast<unsigned>(mpi_rank) < r) {
        local_rows++;
        }

        //Global indexes for meshX and meshY
        unsigned start_row = static_cast<unsigned>(mpi_rank) * (data.n / mpi_size) + std::min(static_cast<unsigned>(mpi_rank), r);

        //Rank up and rank down for communication
        int rank_up = mpi_rank - 1;
        int rank_down = mpi_rank + 1;
        if (rank_up < 0) rank_up = MPI_PROC_NULL;
        if (rank_down >= mpi_size) rank_down = MPI_PROC_NULL;

        //Local matrices
        eigenMatrix u_h_local = eigenMatrix::Zero(local_rows + 2, data.n);
        eigenMatrix u_h_new_local = eigenMatrix::Zero(local_rows + 2, data.n); 
        eigenMatrix meshX_local = meshX.block(start_row, 0, local_rows, data.n);
        eigenMatrix meshY_local = meshY.block(start_row, 0, local_rows, data.n);

        apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_h_local, data, meshX, meshY, mpi_rank, mpi_size);

        u_h_local.block(1, 0, local_rows, data.n) = u_h.block(start_row, 0, local_rows, data.n);
        
        if (mpi_rank == 0) {
            u_h_local.row(0) = u_h.row(0);
        }
        if (mpi_rank == mpi_size - 1) {
            u_h_local.row(local_rows + 1) = u_h.row(data.n - 1);
        }
        u_h_new_local = u_h_local;

        // Cycle
        unsigned i_start = (mpi_rank == 0) ? 2 : 1;
        unsigned i_end = (mpi_rank == mpi_size - 1) ? local_rows - 1 : local_rows;
        unsigned iter = 0;
        double err = data.tolerance + 1.0;
        const double h2 = h * h;

        while(err > data.tolerance && iter < data.max_iterations){
            double loc_err = data.tolerance + 1.0;
            double first_err = 0.0;
            bool first_iter = (iter == 0);
            unsigned loc_iter = 0;
            MPI_Sendrecv(u_h_local.data() + 1 * data.n, data.n, MPI_DOUBLE, rank_up, 0, 
                         u_h_local.data() + 0 * data.n, data.n, MPI_DOUBLE, rank_up, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(u_h_local.data() + local_rows * data.n, data.n, MPI_DOUBLE, rank_down, 1, 
                         u_h_local.data() + (local_rows + 1) * data.n, data.n, MPI_DOUBLE, rank_down, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            double local_tolerance = data.tolerance*100;
            while(loc_err > local_tolerance && loc_iter < data.max_iterations){
                loc_err = 0.0;
                #pragma omp parallel for reduction(max:loc_err)
                for (unsigned i = i_start; i <= i_end; ++i){
                    for (unsigned j = 1; j < data.n - 1; ++j){
                        u_h_new_local(i, j) = 0.25 * (u_h_local(i-1, j) + u_h_local(i+1, j) + u_h_local(i, j-1) + u_h_local(i, j+1) + h2 * data.f0(meshX_local(i-1, j), meshY_local(i-1, j)));                    
                        double local_err = std::abs(u_h_new_local(i, j) - u_h_local(i, j));
                        if (local_err > loc_err) {
                            loc_err = local_err;
                        }
                    }
                }
                if (first_iter) {
                    first_err = loc_err;
                }
                u_h_local.swap(u_h_new_local);
                loc_iter++;
            }
            //Gather results for next iteration
            MPI_Allreduce(&loc_err, &err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            iter++;
        }

        //Gather final results
        std::vector<int> recv_counts(mpi_size);
        std::vector<int> displs(mpi_size);
        int send_elements = local_rows * data.n;
        MPI_Gather(&send_elements, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (mpi_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < mpi_size; ++i) {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
        }
        eigenMatrix u_h_gathered;
        if (mpi_rank == 0) {
            u_h_gathered.resize(data.n, data.n);
            u_h_gathered.row(0) = u_h.row(0);
            u_h_gathered.row(data.n - 1) = u_h.row(data.n - 1);
        }

        // Row-wise and Column-wise gather
        MPI_Gatherv(u_h_local.data() + 1 * data.n, send_elements, MPI_DOUBLE, 
                    u_h_gathered.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (mpi_rank == 0) {
            u_h = u_h_gathered; 
        }

        //Return result
        Result_Struct result;
        result.u_h = u_h;
        result.iterations = iter;
        result.X = meshX;
        result.Y = meshY;
        return result;
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::schwarz_parallel_neumann(){
        // Implementazione del metodo di Schwarz parallelo con condizioni al contorno di Neumann
        return Result_Struct(); // Placeholder
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::schwarz_parallel_robin(){
        // Implementazione del metodo di Schwarz parallelo con condizioni al contorno di Robin
        return Result_Struct(); // Placeholder
    }


    /* --- CONVERGENCE TEST METHOD --- */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Convergence_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::convergence_test(){
        return Convergence_Struct(); // Placeholder
    }
}

#endif