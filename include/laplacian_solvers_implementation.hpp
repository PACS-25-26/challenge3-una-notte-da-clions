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

    /* --- MESH GENERATION AND INITIALIZATION --- */

    /**
     * @brief Builds the mesh according to sequential or parallel execution mode.
     * Since the required operation are trivial, the overhead cost of paralleization might actually be a detriment. 
     */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_mesh(){

        meshX = eigenMatrix::Zero(data.n, data.n);
        meshY = eigenMatrix::Zero(data.n, data.n);
        h = abs(data.x2 - data.x1) / (data.n - 1); // Uniform grid spacing. Works even if x2 < x1. 

        if constexpr (execution_mode == ExecutionMode::SEQUENTIAL) build_mesh_sequential();
        else build_mesh_parallel();
    }

    /**
     * @brief Builds the mesh according to sequential execution mode.
     * Simply loops over the grid points and assigns the corresponding coordinates to meshX and meshY.
     */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_mesh_sequential(){
        
        for(unsigned i = 0; i < data.n; i++) for(unsigned j = 0; j < data.n; j++) {
            meshX(i, j) = data.x1 + j * h;
            meshY(i, j) = data.x2 - i * h;
        }
    }

    /**
     * @brief Builds the mesh according to parallel execution mode.
     * This is done by decompsing the domain according to example shown on the challenge pdf file
     */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_mesh_parallel(){
        
        // Get rank information
        const int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        // Distribute rows among processes. If the number of rows is not divided by the number of threads, the remaining ones will be assigned to the
        // fisrt threads, as shown in Figure 1 of the challenge pdf file.
        const unsigned local_rows = data.n / mpi_size;
        const unsigned remainder_rows = data.n % mpi_size;

        const unsigned start_row = mpi_rank * local_rows + std::min(remainder_rows, static_cast<unsigned>(mpi_rank));
        const unsigned end_row = start_row + local_rows + (mpi_rank < remainder_rows ? 1 : 0);

        auto local_meshX = eigenMatrix::Zero(end_row - start_row, data.n);
        auto local_meshY = eigenMatrix::Zero(end_row - start_row, data.n);


        // Thread safety is guaranteed since each process has different rows.
        for(unsigned i = 0; i < end_row - start_row; i++) for(unsigned j = 0; j < data.n; j++){
            local_meshX(i, j) = data.x1 + j * h;
            local_meshY(i, j) = data.x2 - (i + start_row) * h;
        }

        // Gather the local meshes into the global one. Al processes will have a copy of the complete mesh.
        MPI_Allgather(local_meshX.data(), (end_row - start_row) * data.n, MPI_DOUBLE, meshX.data(), (end_row - start_row) * data.n, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(local_meshY.data(), (end_row - start_row) * data.n, MPI_DOUBLE, meshY.data(), (end_row - start_row) * data.n, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    /**
     * @brief Builds the mesh according to parallel execution mode.
     * This is done by decompsing the domain according to example shown on the challenge pdf file
     */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_exact_solution(){

        u_exact = eigenMatrix::Zero(data.n, data.n);

        if constexpr (execution_mode == ExecutionMode::SEQUENTIAL) build_exact_solution_sequential();
        else build_exact_solution_parallel();
    }

    /**
     * @brief Builds the exact solution according to sequential execution mode.
     * Simply loops over the grid points, evaluates the exact solution and stores the result.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_exact_solution_sequential(){
        
        for(unsigned i = 0; i < data.n; i++) for(unsigned j = 0; j < data.n; j++) 
            u_exact(i, j) = data.u_exact_lambda(meshX(i, j), meshY(i, j));
        
    }

    /**
     * @brief Builds the exact solution according to parallel execution mode.
     * This is done by decompsing the domain according to example shown on the challenge pdf file
     */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_exact_solution_parallel(){
        
        // Get rank information
        const int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        // Distribute rows among processes. If the number of rows is not divided by the number of threads, the remaining ones will be assigned to the
        // fisrt threads, as shown in Figure 1 of the challenge pdf file.
        const unsigned local_rows = data.n / mpi_size;
        const unsigned remainder_rows = data.n % mpi_size;

        const unsigned start_row = mpi_rank * local_rows + std::min(remainder_rows, static_cast<unsigned>(mpi_rank));
        const unsigned end_row = start_row + local_rows + (mpi_rank < remainder_rows ? 1 : 0);

        auto local_exact_solution = eigenMatrix::Zero(end_row - start_row, data.n);

        // Thread safety is guaranteed since each process has different rows.
        for(unsigned i = 0; i < end_row - start_row; i++) for(unsigned j = 0; j < data.n; j++){
            local_exact_solution(i, j) = data.u_exact_lambda(meshX(i + start_row, j), meshY(i + start_row, j));
        }

        // Gather the local exact solutions into the global one. All processes will have a copy of the complete exact solution.
        MPI_Allgather(local_exact_solution.data(), (end_row - start_row) * data.n, MPI_DOUBLE, u_exact.data(), (end_row - start_row) * data.n, MPI_DOUBLE, MPI_COMM_WORLD);
    }

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
     * @brief Solves the Laplacian using the sequential Jacobi iterative method with Dirichlet BCs.
     * The method updates the entire domain grid iteratively by averaging neighbors until 
     * the residual error falls below the specified tolerance or max iterations are reached.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_sequential_dirichlet(){

        eigenMatrix u_new = u_h; // u_h è stato inizializzato?
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

    Result_Struct result;
    result.u_h = u_h;
    result.iterations = iter;
    result.X = meshX;
    result.Y = meshY;
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
        double error = data.tolerance + 1.0;
        const double h2 = h * h;
        const unsigned last = data.n - 1;

        while (error > data.tolerance && iter < data.max_iterations) {
            error = 0.0;

            for (unsigned i = 1; i < last; ++i) {
                for (unsigned j = 1; j < last; ++j) {
                    u_new(i, j) = 0.25 * (u_h(i-1, j) + u_h(i+1, j) + u_h(i, j-1) + u_h(i, j+1) + h2 * data.f0(meshX(i, j), meshY(i, j)));
                    
                    double diff = std::abs(u_new(i, j) - u_h(i, j));
                    if (diff > error) error = diff;
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

            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_new, data, meshX, meshY);

            for (unsigned i = 0; i < data.n; ++i) {
                error = std::max({error, std::abs(u_new(i, 0) - u_h(i, 0)), 
                                        std::abs(u_new(i, last) - u_h(i, last)),
                                        std::abs(u_new(0, i) - u_h(0, i)), 
                                        std::abs(u_new(last, i) - u_h(last, i))});
            }

            u_h = u_new;
            iter++;
        }

        Result_Struct result;
        result.u_h = u_h;
        result.iterations = iter;
        result.X = meshX;
        result.Y = meshY;
        return result;
    }

    /**
     * @brief Solves the Laplacian using the sequential Jacobi iterative method with Robin BCs.
     * Implements mixed boundary conditions. The update rule involves a weighted average 
     * regulated by the 'gamma' parameter to represent the Robin boundary constraint.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_sequential_robin(){
        eigenMatrix u_new = u_h; // Inizializzata a zeri dal costruttore
        unsigned iter = 0;
        double error = data.tolerance + 1.0;
        const double h2 = h * h;
        const unsigned last = data.n - 1;

        const double den = 1.0 + data.gamma * h;

        while (error > data.tolerance && iter < data.max_iterations) {
            error = 0.0;

            for (unsigned i = 1; i < last; ++i) {
                for (unsigned j = 1; j < last; ++j) {
                    u_new(i, j) = 0.25 * (u_h(i-1, j) + u_h(i+1, j) + 
                                        u_h(i, j-1) + u_h(i, j+1) + 
                                        h2 * data.f0(meshX(i, j), meshY(i, j)));
                    
                    double diff = std::abs(u_new(i, j) - u_h(i, j));
                    if (diff > error) error = diff;
                }
            }

            /**
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

            apply_boundary_condition<boundary_condition, execution_mode, funcType>(u_new, data, meshX, meshY);

            for (unsigned i = 0; i < data.n; ++i) {
                error = std::max({error, std::abs(u_new(i, 0) - u_h(i, 0)), 
                                        std::abs(u_new(i, last) - u_h(i, last)),
                                        std::abs(u_new(0, i) - u_h(0, i)), 
                                        std::abs(u_new(last, i) - u_h(last, i))});
            }

            u_h = u_new;
            iter++;
        }

        Result_Struct result;
        result.u_h = u_h;
        result.iterations = iter;
        result.X = meshX;
        result.Y = meshY;
        return result;
    }

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
        ParallelMatrix u_h_local = eigenMatrix::Zero(local_rows + 2, data.n);
        ParallelMatrix u_h_new_local = eigenMatrix::Zero(local_rows + 2, data.n); 
        ParallelMatrix meshX_local = meshX.block(start_row, 0, local_rows, data.n);
        ParallelMatrix meshY_local = meshY.block(start_row, 0, local_rows, data.n);

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
        ParallelMatrix u_h_gathered;
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
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::jacobi_parallel_robin(){
        // Implementazione del metodo di Jacobi parallelo con condizioni al contorno di Robin
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
        ParallelMatrix u_h_local = eigenMatrix::Zero(local_rows + 2, data.n);
        ParallelMatrix u_h_new_local = eigenMatrix::Zero(local_rows + 2, data.n); 
        ParallelMatrix meshX_local = meshX.block(start_row, 0, local_rows, data.n);
        ParallelMatrix meshY_local = meshY.block(start_row, 0, local_rows, data.n);

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
        ParallelMatrix u_h_gathered;
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
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Result_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::schwarz_parallel_robin(){
        // Implementazione del metodo di Schwarz parallelo con condizioni al contorno di Robin
    }


    /* --- CONVERGENCE TEST METHOD --- */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    Convergence_Struct Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::convergence_test(){
        
    }

    /* --- PRINT AND EXPORT METHODS --- */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::print(){
        // STAMPA GRIGLIA, SOLUZIONE DISCRETA, SOLUZIONE ESATTA SULLA GRIGLIA
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::export_to_vtk(const eigenMatrix& meshX, const eigenMatrix& meshY, const eigenMatrix& u_h, const std::string& filename){
        // Esporta i dati in formato VTK per la visualizzazione con Paraview
    }

}

#endif