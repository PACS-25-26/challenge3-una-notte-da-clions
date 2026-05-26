#ifndef LAPLACIAN_CONVERGENCE_TEST_HPP
#define LAPLACIAN_CONVERGENCE_TEST_HPP

// Test functions for Laplacian solver convergence
#include "laplacian_solvers.hpp"
#include "laplacian_solvers_implementation.hpp"
#include <cmath>
#include <vector>
#include <chrono>

namespace laplacian_solvers {

    /**
     * @struct Convergence_Struct
     * @brief Stores collected benchmark metrics across multiple grid refinement levels.
     * 
     * Tracks L2 errors, iteration counts, and computational times for both serial and parallel
     * runs to facilitate convergence order verification and speedup analysis.
     */
    struct Convergence_Struct {
        std::vector<double> errors_serial;
        std::vector<double> errors_parallel; 
        std::vector<unsigned> n;
        std::vector<double> iterations_serial;
        std::vector<double> iterations_parallel;
        std::vector<double> times_serial;
        std::vector<double> times_parallel;
    };

    template <SolverType solver_type, BoundaryCondition boundary_condition, typename funcType>
    class Convergence_Test {  
        private:
        Data_Struct<funcType> data_original;
        std::vector<unsigned> refinement_levels = {16, 32, 64, 128, 256}; 
        double compute_error(const eigenMatrix& u_h, const eigenMatrix& X, const eigenMatrix& Y, funcType u_exact_lambda);
        public:
        Convergence_Test(const Data_Struct<funcType>& data) : data_original(data) {}
        Convergence_Struct run_convergence_test();
    };

    template <SolverType solver_type, BoundaryCondition boundary_condition, typename funcType>
    Convergence_Struct Convergence_Test<solver_type, boundary_condition, funcType>::run_convergence_test() {
        Convergence_Struct convergence_data;

        for(unsigned n_val : refinement_levels) {
            convergence_data.n.push_back(n_val);
            
            data_original.n = n_val; 

            // Serial Run
            auto start_serial = std::chrono::steady_clock::now();

            Laplacian_Solver<solver_type, boundary_condition, ExecutionMode::SEQUENTIAL, funcType> solver_dirichlet_sequential(data_original);
            solver_dirichlet_sequential.build_mesh();
            auto result_serial = solver_dirichlet_sequential.solve();

            auto end_serial = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::milli> elapsed_serial = end_serial - start_serial;
            convergence_data.times_serial.push_back(elapsed_serial.count());
        
            double error_serial = compute_error(result_serial.u_h, result_serial.X, result_serial.Y, data_original.u_exact_lambda);
            convergence_data.errors_serial.push_back(error_serial);
            convergence_data.iterations_serial.push_back(result_serial.iterations);

            // Parallel Run
            auto start_parallel = std::chrono::steady_clock::now();

            Laplacian_Solver<solver_type, boundary_condition, ExecutionMode::PARALLEL, funcType> solver_dirichlet_parallel(data_original);
            solver_dirichlet_parallel.build_mesh();
            auto result_parallel = solver_dirichlet_parallel.solve();
            
            auto end_parallel = std::chrono::steady_clock::now();

            std::chrono::duration<double, std::milli> elapsed_parallel = end_parallel - start_parallel;
            convergence_data.times_parallel.push_back(elapsed_parallel.count());

            double error_parallel = compute_error(result_parallel.u_h, result_parallel.X, result_parallel.Y, data_original.u_exact_lambda);
            convergence_data.errors_parallel.push_back(error_parallel);
            convergence_data.iterations_parallel.push_back(result_parallel.iterations);
            
        }

        return convergence_data;
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

}
    


#endif // LAPLACIAN_CONVERGENCE_TEST_HPP