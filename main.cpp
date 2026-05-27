#include <mpi.h>
#include <omp.h>
#include "laplacian_solvers_test.hpp"

int main(int argc, char* argv[]) {
    // 1. Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // 2. Execute Tests (Uncomment the ones you want to run)
    // --- Sequential Tests ---
    //run_test_1_dirichlet_sequential(mpi_rank);
    //run_test_2_neumann_sequential(mpi_rank);
    run_test_3_robin_sequential(mpi_rank);

    // --- Parallel Tests Jacobi ---
    //run_test_4_dirichlet_parallel(mpi_rank);
    //run_test_5_neumann_parallel(mpi_rank);
    run_test_6_robin_parallel(mpi_rank);

    // --- Parallel Tests Schwarz ---
    //run_test_7_dirichlet_schwarz(mpi_rank);
    //run_test_8_neumann_schwarz(mpi_rank);
    //run_test_9_robin_schwarz(mpi_rank);

    // 3. Clean up MPI environment
    MPI_Finalize();
    return 0;
}