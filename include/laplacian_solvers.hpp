#ifndef LAPLACIAN_SOLVERS_HPP
#define LAPLACIAN_SOLVERS_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <algorithm>

/**
 * @file laplacian_solvers.hpp
 * @brief Definition of the Laplacian/Poisson equation solver with multi-backend support (MPI, OpenMP, Sequential).
 */

/// Alias for Eigen matrices optimized in RowMajor to ensure better memory alignment during parallel access.
using eigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

#define MAX_SCHWARZ_ITERATIONS 10

/**
 * @enum SolverType
 * @brief Specifies the iterative numerical algorithm used to solve the linear system.
 */
enum class SolverType {
    JACOBI, ///< Jacobi iterative method
    SCHWARZ ///< Schwarz iterative method
};  

/**
 * @enum BoundaryCondition
 * @brief Specifies the physical boundary condition type applied to the domain edges.
 */
enum class BoundaryCondition {
    DIRICHLET, ///< Prescribed value on the boundary.
    NEUMANN, ///< Prescribed flux (normal derivative) on the boundary.
    ROBIN ///< Mixed condition (linear combination of value and normal derivative).
};

/**
 * @enum ExecutionMode
 * @brief Specifies the execution policy of the solver (sequential or parallel).
 */
enum class ExecutionMode {
    SEQUENTIAL, ///< Sequential execution policy.
    PARALLEL ///< Parallel execution policy using MPI and OpenMP.
};

/**
 * @struct Data_Struct
 * @brief Holds all geometric, physical, and numerical configuration parameters for the problem.
 * @tparam funcType Type of the callables representing source terms and BCs.
 */
template <typename funcType>
struct Data_Struct {
    unsigned n; ///< Number of discretization subintervals per spatial coordinate.
    double x1, x2; 

    funcType f0;  ///< Source term
    funcType f1;  ///< BC bottom edge
    funcType f2;  ///< BC right edge
    funcType f3;  ///< BC top edge
    funcType f4;  ///< BC left edge
    funcType u_exact_lambda; ///< Analytical solution (for error analysis)
    double tolerance; ///< Convergence tolerance for the iterative solver.
    unsigned max_iterations; ///< Maximum number of iterations for the iterative solver.
    double alpha; ///< Parameter for Robin boundary condition: gamma*u + du/dn = g
    unsigned max_schwarz_iterations = 10; ///< Maximum number of Schwarz iterations (if using Schwarz method)
};

/**
 * @struct Result_Struct
 * @brief Data structure to store the final computational results computed by the solver.
 */
struct Result_Struct {
    eigenMatrix X;                 ///< Matrix containing the X coordinates of the structured mesh nodes.
    eigenMatrix Y;                 ///< Matrix containing the Y coordinates of the structured mesh nodes.
    eigenMatrix u_h;               ///< Computed numerical approximate solution.
    unsigned iterations;           ///< Number of iterations actually performed.
    double iteration_residue;     ///< Norm of the final residual at the last iterative step.
    bool valid = false;            ///< Validity flag (set to true only on Rank 0 after gathering data).
};

/**
 * @namespace laplacian_solvers
 * @brief Namespace dedicated to the computation and numerical resolution of the Laplacian operator.
 */
namespace laplacian_solvers {

    /**
     * @class Laplacian_Solver
     * @brief Core class for solving the Laplace/Poisson equation on structured grids.
     * * Leverages compile-time metaprogramming (via template parameters) to statically configure
     * the algorithm, boundary conditions, and execution backend, maximizing runtime efficiency.
     * * @tparam solver_type The iterative algorithm to be used (@ref SolverType).
     * @tparam boundary_condition The type of BC applied to the domain (@ref BoundaryCondition).
     * @tparam execution_mode The computational paradigm backend (@ref ExecutionMode).
     * @tparam funcType The type of functional descriptors for sources and boundaries.
     */
    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    class Laplacian_Solver {
        private:
            Data_Struct<funcType> data; ///< Configuration of the problem and parameters.
            eigenMatrix meshX;    ///< Matrix containing the X coordinates of the structured grid.
            eigenMatrix meshY;    ///< Matrix containing the Y coordinates of the structured grid.
            double h;             ///< Uniform grid spacing parameter h.
            eigenMatrix u_exact;  ///< Analytical solution over the mesh points matrix.
            Result_Struct result; ///< Struct to hold the final results of the solver.

            int mpi_rank; ///< Rank of the MPI process (used in parallel execution).
            int mpi_size; ///< Number of participating MPI processes (used in parallel execution).
            int mpi_used_size; ///< Actual MPI_COMM_WORLD size (for reference).
            bool participating = true; ///< True if this rank participates in computation (rank < mpi_size)

            
        public:
        
            
            Laplacian_Solver(const Data_Struct<funcType>& d); //OK

            
            Result_Struct solve(); //OK

           
            void print() const;

            
            void export_to_vtk(const std::string& filename) const;

            void build_mesh(); //OK

            private:
            // PRIVATE HELPER METHODS

            // INITIALIZERS
            
            void build_mesh_sequential(); //OK
            void build_mesh_parallel(); //OK
            void process_filter();

            void build_exact_solution(); //OK
            void build_exact_solution_sequential(); //OK
            void build_exact_solution_parallel(); //OK

            // PRINT AND POSTPROCESSING
            void print_mesh() const; //OK
            void print_solution() const; //OK
            void print_exact_solution() const; //OK

            //SOLVER STRUCTURE: sequential / parallel -> jacobi / schwarz -> dirichlet / neumann / robin
            Result_Struct parallel_solve(); //OK

            Result_Struct jacobi_sequential(); //OK
            Result_Struct jacobi_parallel(); //OK
            Result_Struct schwarz_parallel(); //OK

            
    };

} // namespace laplacian_solvers

#include "laplacian_solvers_implementation.hpp"
#include "laplacian_solvers_postprocessing.hpp"
#include "laplacian_solvers_initializer.hpp"

#include "laplacian_solvers_sequential_implementations.hpp"
#include "laplacian_solvers_parallel_jacobi_implementation.hpp"
#include "laplacian_solvers_parallel_schwarz_implementation.hpp"


#endif 