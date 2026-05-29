#ifndef LAPLACIAN_CONVERGENCE_TEST_HPP
#define LAPLACIAN_CONVERGENCE_TEST_HPP

#include "laplacian_solvers.hpp"
#include "laplacian_solvers_implementation.hpp"
#include <cmath>
#include <vector>
#include <chrono>

/**
 * @file laplacian_convergence_test.hpp
 * @brief Convergence test suite for verifying solver accuracy and performance speedup.
 */

namespace laplacian_solvers {

    /**
     * @struct Convergence_Struct
     * @brief Stores collected benchmark metrics across multiple grid refinement levels.
     * * Tracks L2 errors, iteration counts, and computational times for both serial and parallel
     * runs to facilitate convergence order verification and speedup analysis.
     */
    struct Convergence_Struct {
        std::vector<double> errors_serial;       ///< L2 norm error tracking for sequential solver runs.
        std::vector<double> errors_parallel;     ///< L2 norm error tracking for parallel solver runs.
        std::vector<unsigned> n;                 ///< Grid resolution size sequence values.
        std::vector<double> iterations_serial;   ///< Number of inner solver loops completed sequentially.
        std::vector<double> iterations_parallel; ///< Number of inner solver loops completed in parallel.
        std::vector<double> times_serial;        ///< Elapsed execution duration for sequential loops (ms).
        std::vector<double> times_parallel;      ///< Elapsed execution duration for parallel loops (ms).
        std::vector<double> speedups;            ///< Speedup scaling metric ratio (Serial Time / Parallel Time).
    };

    /**
     * @class Convergence_Test
     * @brief Benchmark coordinator running serial and parallel execution comparison groups.
     * * Evaluates numerical execution trends across variable space step metrics to compute 
     * discretization error orders and performance profiling benchmarks.
     * * @tparam solver_type Iterative algorithm selector.
     * @tparam boundary_condition Boundary condition policy.
     * @tparam funcType Callable type for boundary and source terms.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, typename funcType>
    class Convergence_Test {  

        private:
        Data_Struct<funcType> data_original; ///< Base structural configurations copy.
        std::vector<unsigned> refinement_levels = {8, 16, 32, 64, 128}; ///< Active mesh sampling steps.
        
        /**
         * @brief Internal utility evaluating L2 discrete space error bounds.
         */
        double compute_error(const eigenMatrix& u_h, const eigenMatrix& X, const eigenMatrix& Y, funcType u_exact_lambda);
        
        Convergence_Struct convergence_data_struct; ///< Internal evaluation data metrics cache.
        bool is_test_done = false;                  ///< Validation flag verifying benchmark sequence completion.
        int mpi_rank;                               ///< Rank of the active MPI process allocation context.

        public:
        /**
         * @brief Constructs a Convergence_Test runner instance.
         * @param data Fundamental configuration context parameters container reference.
         */
        Convergence_Test(const Data_Struct<funcType>& data) : data_original(data) {
            // Initialize MPI rank for potential use in parallel mesh building or solver execution
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        }

        /**
         * @brief Executes sequential and parallel benchmark solvers sequentially over every refinement level.
         * @return Populated @ref Convergence_Struct containing collected performance profiles.
         */
        Convergence_Struct run_convergence_test();

        /**
         * @brief Formats and prints gathered convergence tracking details to standard output channels.
         * * Restricts visual render triggers to Rank 0 to bypass standard console racing errors.
         */
        void print_convergence_results() const;

    };

    template <SolverType solver_type, BoundaryCondition boundary_condition, typename funcType>
    Convergence_Struct Convergence_Test<solver_type, boundary_condition, funcType>::run_convergence_test() {

        
        for(unsigned n_val : refinement_levels) {
            convergence_data_struct.n.push_back(n_val);
            
            data_original.n = n_val; 

            // Serial Run
            auto start_serial = std::chrono::steady_clock::now();

            Laplacian_Solver<solver_type, boundary_condition, ExecutionMode::SEQUENTIAL, funcType> solver_dirichlet_sequential(data_original);
            solver_dirichlet_sequential.build_mesh();
            auto result_serial = solver_dirichlet_sequential.solve();

            auto end_serial = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::milli> elapsed_serial = end_serial - start_serial;
            convergence_data_struct.times_serial.push_back(elapsed_serial.count());

        
            double error_serial = compute_error(result_serial.u_h, result_serial.X, result_serial.Y, data_original.u_exact_lambda);
            convergence_data_struct.errors_serial.push_back(error_serial);
            convergence_data_struct.iterations_serial.push_back(result_serial.iterations);
        }
        unsigned i = 0;
        for(unsigned n_val : refinement_levels) {
            data_original.n = n_val; 
            
            // Parallel Run
            auto start_parallel = std::chrono::steady_clock::now();

            Laplacian_Solver<solver_type, boundary_condition, ExecutionMode::PARALLEL, funcType> solver_dirichlet_parallel(data_original);
            solver_dirichlet_parallel.build_mesh();
            auto result_parallel = solver_dirichlet_parallel.solve();
            
            auto end_parallel = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::milli> elapsed_parallel = end_parallel - start_parallel;
            convergence_data_struct.times_parallel.push_back(elapsed_parallel.count());

            double error_parallel = compute_error(result_parallel.u_h, result_parallel.X, result_parallel.Y, data_original.u_exact_lambda);
            convergence_data_struct.errors_parallel.push_back(error_parallel);
            convergence_data_struct.iterations_parallel.push_back(result_parallel.iterations);
            convergence_data_struct.speedups.push_back(convergence_data_struct.times_serial[i] / convergence_data_struct.times_parallel[i]);
            i++;
        }

        is_test_done = true;
        return convergence_data_struct;
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, typename funcType>
    double Convergence_Test<solver_type, boundary_condition, funcType>::compute_error(const eigenMatrix& u_h, const eigenMatrix& X, const eigenMatrix& Y, funcType u_exact_lambda) {
        double error_sum = 0.0;
        unsigned n = u_h.rows();
        double h = (data_original.x2 - data_original.x1) / (n - 1);
        for (unsigned i = 0; i < n; ++i) {
            for (unsigned j = 0; j < n; ++j) {
                double u_exact = u_exact_lambda(X(i, j), Y(i, j));
                error_sum += h*std::pow(u_h(i, j) - u_exact, 2);
            }
        }
        return std::sqrt(error_sum);

    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, typename funcType>
    void Convergence_Test<solver_type, boundary_condition, funcType>::print_convergence_results() const {
        if (mpi_rank != 0) {
            // Only the master process should print results to avoid clutter
            return;
        }
        
        if (!is_test_done) {
            std::cerr << "Convergence test has not been run yet. Please run run_convergence_test() before printing results." << std::endl;
            return;
        } 
        std::cout << "Refinement Level (n) | L2 Error (Serial) | L2 Error (Parallel) | Iterations (Serial) | Iterations (Parallel) | Time (ms, Serial) | Time (ms, Parallel) | Speedup" << std::endl;
        for (size_t i = 0; i < convergence_data_struct.n.size(); ++i) {
            std::cout << convergence_data_struct.n[i] << " | "
                      << convergence_data_struct.errors_serial[i] << " | "
                      << convergence_data_struct.errors_parallel[i] << " | "
                      << convergence_data_struct.iterations_serial[i] << " | "
                      << convergence_data_struct.iterations_parallel[i] << " | "
                      << convergence_data_struct.times_serial[i] << " | "
                      << convergence_data_struct.times_parallel[i] << " | "
                      << convergence_data_struct.speedups[i] << std::endl;    
        }
        return;
    }

} // namespace laplacian_solvers
    
#endif // LAPLACIAN_CONVERGENCE_TEST_HPP