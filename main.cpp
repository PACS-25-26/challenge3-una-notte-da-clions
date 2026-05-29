#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#ifndef AVAILABLE_THREADS
#define AVAILABLE_THREADS 10
#endif
#endif

#include "laplacian_solvers_test_1.hpp"
#include "laplacian_solvers_test_2.hpp"

int main(int argc, char* argv[]) {
    // 1. Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Set the amout of threads each MPI process is able to create
    #ifdef _OPENMP
        const unsigned n_threads = AVAILABLE_THREADS / mpi_size;
        const unsigned remainder_threads = AVAILABLE_THREADS % mpi_size;
        omp_set_num_threads(n_threads + (mpi_rank < remainder_threads));
    #endif

    #ifndef NDEBUG
    // Print parallelism info
    if(mpi_rank == 0){
        std::cout << "Started MPI session with number of processes: " << mpi_size << std::endl;
        #ifdef _OPENMP
        const unsigned n_threads = AVAILABLE_THREADS / mpi_size;
        const unsigned remainder_threads = AVAILABLE_THREADS % mpi_size;
        std::cout << "Started OpenMP session with total number of threads: " << AVAILABLE_THREADS << std::endl;
        for(int i = 0; i < mpi_size; i++)
            std::cout << "Rank " << i << " is allowed to create " << n_threads + (i < remainder_threads) << " threads" << std::endl;
        
        #endif
    }
    #endif

    // 2. Execute Tests (Uncomment the ones you want to run)
    /*
    // --- Sequential Tests ---
    laplacian_solvers::run_test_1_dirichlet_sequential(mpi_rank);
    laplacian_solvers::run_test_2_neumann_sequential(mpi_rank);
    laplacian_solvers::run_test_3_robin_sequential(mpi_rank);

    // --- Parallel Tests Jacobi ---
    laplacian_solvers::run_test_4_dirichlet_parallel(mpi_rank);
    laplacian_solvers::run_test_5_neumann_parallel(mpi_rank);
    laplacian_solvers::run_test_6_robin_parallel(mpi_rank);
    
    // --- Parallel Tests Schwarz ---
    laplacian_solvers::run_test_7_dirichlet_parallel_schwarz(mpi_rank);
    laplacian_solvers::run_test_8_neumann_parallel_schwarz(mpi_rank);
    laplacian_solvers::run_test_9_robin_parallel_schwarz(mpi_rank);
    */
    // --- Convergence tests Jacobi ---
    laplacian_solvers::run_test_10_dirichlet_sequential_vs_parallel_jacobi(mpi_rank);
    laplacian_solvers::run_test_11_neumann_sequential_vs_parallel_jacobi(mpi_rank);
    laplacian_solvers::run_test_12_robin_sequential_vs_parallel_jacobi(mpi_rank);

    // --- Convergence tests Schwarz ---
    laplacian_solvers::run_test_13_dirichlet_sequential_vs_parallel_schwarz(mpi_rank);
    laplacian_solvers::run_test_14_neumann_sequential_vs_parallel_schwarz(mpi_rank);
    laplacian_solvers::run_test_15_robin_sequential_vs_parallel_schwarz(mpi_rank); 
    /*
    // --- Additional Tests ---
    laplacian_solvers::run_test_16_dirichlet_sequential(mpi_rank);
    laplacian_solvers::run_test_17_neumann_sequential(mpi_rank);
    laplacian_solvers::run_test_18_robin_sequential(mpi_rank);
    laplacian_solvers::run_test_19_dirichlet_parallel(mpi_rank); 
    laplacian_solvers::run_test_20_neumann_parallel(mpi_rank);
    laplacian_solvers::run_test_21_robin_parallel(mpi_rank);
    laplacian_solvers::run_test_22_neumann_sequential_vs_parallel_schwarz(mpi_rank); // Remember tha neumann has a zero avg condition!
    */
    // 3. Clean up MPI environment
    MPI_Finalize();
    return 0;
}