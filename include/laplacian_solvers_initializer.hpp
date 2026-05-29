#ifndef LAPLACIAN_SOLVERS_INITIALIZER_HPP
#define LAPLACIAN_SOLVERS_INITIALIZER_HPP

#include <omp.h>
#include <mpi.h>

#include "laplacian_solvers.hpp"

/**
 * @file laplacian_solvers_initializer.hpp
 * @brief Implementation of initialization, mesh generation, and exact solution building for Laplacian_Solver.
 */

namespace laplacian_solvers{

    /* --- CONSTRUCTOR --- */

    /**
     * @brief Constructs a new Laplacian Solver object.
     * * Initializes problem data parameters, checks process topology validity via the filter,
     * and allocates/populates both the computational grid grid and the exact solution matrix.
     * * @tparam solver_type Iterative algorithm selector.
     * @tparam boundary_condition Boundary condition policy.
     * @tparam execution_mode Parallelism execution backend policy.
     * @tparam funcType Callable type for boundary and source terms.
     * * @param d Configuration data structure containing physical and analytical bounds.
     */

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename funcType>
    Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::Laplacian_Solver(const Data_Struct<funcType>& d) : data(d) {
        
        // Controls process availability, builds mesh and exact solution
        process_filter();
        build_mesh();
        build_exact_solution();
    
    };

    /**
     * @brief Filters active MPI processes and checks communicator sizing validity.
     * * In parallel execution mode, ensures that the number of allocated MPI ranks does not
     * exceed the total number of mesh rows (\f$ N \f$), preventing empty or unassigned processes.
     * If the topology is invalid, Rank 0 prints an error and aborts execution.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    inline void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::process_filter(){

        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        // If more processes are available than rows throw error
        
        if constexpr(execution_mode == ExecutionMode::PARALLEL){
            if(mpi_size > static_cast<int>(data.n)) {
                if (mpi_rank == 0) std::cerr << "Error: program launched with " << mpi_size << " processors, but number of rows is " << data.n << ". Aborting..." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);                
            }
        }

    }
    
    /* --- MESH GENERATION AND INITIALIZATION --- */

    /**
     * @brief Computes the uniform grid spacing and routes mesh generation to the active backend.
     * * Calculates the uniform grid step \f$ h = \frac{|x_2 - x_1|}{N - 1} \f$. It guarantees 
     * correct computation even if domain bounds are specified in reversed spatial orientation (\f$ x_2 < x_1 \f$).
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode,
              typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_mesh(){

        h = abs(data.x2 - data.x1) / (data.n - 1); // Uniform grid spacing. Works even if x2 < x1. 

        if constexpr (execution_mode == ExecutionMode::SEQUENTIAL) build_mesh_sequential();
        else build_mesh_parallel();
    }
   
    /**
     * @brief Generates the global coordinates mesh on a single thread.
     * * Allocates and populates an \f$ N \times N \f$ matrix representing the full grid domain layout,
     * assigning spatial continuous coordinate mappings directly to `meshX` and `meshY`.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_mesh_sequential(){

        meshX = eigenMatrix::Zero(data.n, data.n);
        meshY = eigenMatrix::Zero(data.n, data.n);
        
        for(unsigned i = 0; i < data.n; i++) for(unsigned j = 0; j < data.n; j++) {
            meshX(i, j) = data.x1 + j * h;
            meshY(i, j) = data.x1 + i * h;
        }
    }

    /**
     * @brief Generates a local sub-grid slice allocated to the invoking MPI process.
     * * Distributes grid rows dynamically across active MPI processes. If the row count \f$ N \f$ 
     * is not perfectly divisible by the number of processes, the remainder rows are assigned 
     * one-by-one to the first lower-ranked processes to ensure optimal workload distribution.
     * Internal cell row populations are accelerated locally via multi-threaded OpenMP parallel loops.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_mesh_parallel(){
        
        // Get rank information

        // Distribute rows among processes. If the number of rows is not divided by the number of threads, the remaining ones will be assigned to the
        // fisrt threads, as shown in Figure 1 of the challenge pdf file.
        const unsigned local_rows = data.n / mpi_size;
        const unsigned remainder_rows = data.n % mpi_size;

        const unsigned start_row = mpi_rank * local_rows + std::min(remainder_rows, static_cast<unsigned>(mpi_rank));
        const unsigned end_row = start_row + local_rows + (static_cast<unsigned>(mpi_rank) < remainder_rows ? 1 : 0);

        meshX = eigenMatrix::Zero(end_row - start_row, data.n);
        meshY = eigenMatrix::Zero(end_row - start_row, data.n);  
        
        #pragma omp parallel for
        for(unsigned i = 0; i < end_row - start_row; i++) for(unsigned j = 0; j < data.n; j++){
           meshX(i, j) = data.x1 + j * h;
           meshY(i, j) = data.x1 + (i + start_row) * h;
        }
    }

    /**
     * @brief Allocates space for the exact analytical solution matrix and triggers initialization.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_exact_solution(){

        u_exact = eigenMatrix::Zero(data.n, data.n);

        if constexpr (execution_mode == ExecutionMode::SEQUENTIAL) build_exact_solution_sequential();
        else build_exact_solution_parallel();
    }

    /**
     * @brief Evaluates the exact analytical lambda expression sequentially across the full grid.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_exact_solution_sequential(){
        
        for(unsigned i = 0; i < data.n; i++) for(unsigned j = 0; j < data.n; j++) 
            u_exact(i, j) = data.u_exact_lambda(meshX(i, j), meshY(i, j));
        
    }

    /**
     * @brief Evaluates the exact analytical lambda function in parallel over the local grid slice.
     * * Re-allocates the local `u_exact` array to match the chunk sizing generated in @ref build_mesh_parallel.
     * Computations are offloaded to an active OpenMP thread pool operating on local memory rows.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::build_exact_solution_parallel(){
        

        // Distribute rows among processes. If the number of rows is not divided by the number of threads, the remaining ones will be assigned to the
        // fisrt threads, as shown in Figure 1 of the challenge pdf file.
        const unsigned local_rows = meshX.rows();

        u_exact = eigenMatrix::Zero(local_rows, data.n);

        #pragma omp parallel for
        for(unsigned i = 0; i < local_rows; i++) for(unsigned j = 0; j < data.n; j++){
            u_exact(i, j) = data.u_exact_lambda(meshX(i, j), meshY(i, j));
        }

    }
}// namespace laplacian_solvers

#endif // LAPLACIAN_SOLVERS_INITIALIZER_HPP