#ifndef LAPLACIAN_SOLVERS_INITIALIZER_HPP
#define LAPLACIAN_SOLVERS_INITIALIZER_HPP

#include <omp.h>
#include <mpi.h>

#include "laplacian_solvers.hpp"

namespace laplacian_solvers{

    
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
            meshY(i, j) = data.x1 + i * h;
        }
    }

    /**
     * @brief Builds the mesh according to parallel execution mode.
     * This is done by decompsing the domain according to example shown on the challenge pdf file
     */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_mesh_parallel(){
        
        // Get rank information
        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        // Distribute rows among processes. If the number of rows is not divided by the number of threads, the remaining ones will be assigned to the
        // fisrt threads, as shown in Figure 1 of the challenge pdf file.
        const unsigned local_rows = data.n / mpi_size;
        const unsigned remainder_rows = data.n % mpi_size;

        const unsigned start_row = mpi_rank * local_rows + std::min(remainder_rows, static_cast<unsigned>(mpi_rank));
        const unsigned end_row = start_row + local_rows + (mpi_rank < remainder_rows ? 1 : 0);

        /*
        eigenMatrix local_meshX = eigenMatrix::Zero(end_row - start_row, data.n);
        eigenMatrix local_meshY = eigenMatrix::Zero(end_row - start_row, data.n);
        */

        for(unsigned i = 0; i < end_row - start_row; i++) for(unsigned j = 0; j < data.n; j++){
           meshX(i, j) = data.x1 + j * h;
           meshY(i, j) = data.x1 + (i + start_row) * h;
        }
    }

    /**
     * @brief Builds the mesh according to parallel execution mode.
     * This is done by decompsing the domain according to example shown on the challenge pdf file
     */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
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
        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        // Distribute rows among processes. If the number of rows is not divided by the number of threads, the remaining ones will be assigned to the
        // fisrt threads, as shown in Figure 1 of the challenge pdf file.
        const unsigned local_rows = meshX.rows();

        /*
        const unsigned start_row = mpi_rank * local_rows + std::min(remainder_rows, static_cast<unsigned>(mpi_rank));
        const unsigned end_row = start_row + local_rows + (mpi_rank < remainder_rows ? 1 : 0);
        */

        u_exact = eigenMatrix::Zero(local_rows, data.n);

        // Thread safety is guaranteed since each process has different rows.
        for(unsigned i = 0; i < local_rows; i++) for(unsigned j = 0; j < data.n; j++){
            u_exact(i, j) = data.u_exact_lambda(meshX(i, j), meshY(i, j));
        }

        // Gather the local exact solutions into the global one. All processes will have a copy of the complete exact solution.
        //MPI_Allgather(local_exact_solution.data(), (end_row - start_row) * data.n, MPI_DOUBLE, u_exact.data(), (end_row - start_row) * data.n, MPI_DOUBLE, MPI_COMM_WORLD);
    }
}

#endif