#include <mpi.h>
#include <omp.h>
#include "laplacian_solvers_test_1.hpp"
#include "laplacian_solvers_test_2.hpp"

int main(int argc, char* argv[]) {
    // 1. Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // 2. Execute Tests (Uncomment the ones you want to run)
    
    // --- Sequential Tests ---
    //laplacian_solvers::run_test_1_dirichlet_sequential(mpi_rank);
    //laplacian_solvers::run_test_2_neumann_sequential(mpi_rank);
    //laplacian_solvers::run_test_3_robin_sequential(mpi_rank);

    // --- Parallel Tests Jacobi ---
    //laplacian_solvers::run_test_4_dirichlet_parallel(mpi_rank);
    laplacian_solvers::run_test_5_neumann_parallel(mpi_rank);
    //laplacian_solvers::run_test_6_robin_parallel(mpi_rank);

    // --- Parallel Tests Schwarz ---
    //laplacian_solvers::run_test_7_dirichlet_parallel_schwarz(mpi_rank);
    //laplacian_solvers::run_test_8_neumann_parallel_schwarz(mpi_rank);
    //laplacian_solvers::run_test_9_robin_parallel_schwarz(mpi_rank);

    // --- Convergence tests Jacobi ---
    //laplacian_solvers::run_test_10_dirichlet_sequential_vs_parallel_jacobi(mpi_rank);
    //laplacian_solvers::run_test_11_neumann_sequential_vs_parallel_jacobi(mpi_rank);
    //laplacian_solvers::run_test_12_robin_sequential_vs_parallel_jacobi(mpi_rank);

    // --- Convergence tests Schwarz ---
    //laplacian_solvers::run_test_13_dirichlet_sequential_vs_parallel_schwarz(mpi_rank);
    //laplacian_solvers::run_test_14_neumann_sequential_vs_parallel_schwarz(mpi_rank);
    //laplacian_solvers::run_test_15_robin_sequential_vs_parallel_schwarz(mpi_rank); 

    // --- Additional Tests ---
    //laplacian_solvers::run_test_16_dirichlet_sequential(mpi_rank);
    //laplacian_solvers::run_test_17_neumann_sequential(mpi_rank);
    //laplacian_solvers::run_test_18_robin_sequential(mpi_rank);
    //laplacian_solvers::run_test_19_dirichlet_parallel(mpi_rank);
    //laplacian_solvers::run_test_20_neumann_parallel(mpi_rank);
    //laplacian_solvers::run_test_21_robin_parallel(mpi_rank);
    //laplacian_solvers::run_test_22_neumann_sequential_vs_parallel_schwarz(mpi_rank);

    // 3. Clean up MPI environment
    MPI_Finalize();
    return 0;
}