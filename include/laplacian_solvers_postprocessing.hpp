#ifndef LAPLACIAN_SOLVERS_POSTPROCESSING_HPP
#define LAPLACIAN_SOLVERS_POSTPROCESSING_HPP

#include "laplacian_solvers.hpp"

namespace laplacian_solvers{

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::print() const{
        std::cout << "Mesh points (x, y):" << std::endl;
        print_mesh();
        std::cout << "Discrete solution u_h:" << std::endl;
        print_solution();
        std::cout << "Exact solution u_exact:" << std::endl;
        print_exact_solution();
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::print_mesh() const{

        if constexpr (execution_mode == ExecutionMode::PARALLEL) {
            int mpi_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            if(mpi_rank != 0) return; // Only the root process
        }

        for(unsigned i = 0; i < data.n; i++){
            for(unsigned j = 0; j < data.n; j++) std::cout << "(" << meshX(i, j) << ", " << meshY(i, j) << ") ";
             std::cout << std::endl;
        }
       
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::print_solution() const{

        if constexpr (execution_mode == ExecutionMode::PARALLEL) {
            int mpi_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            if(mpi_rank != 0) return; // Only the root process prints
        }


        for(unsigned i = 0; i < data.n; i++){
            for(unsigned j = 0; j < data.n; j++) std::cout << u_h(i, j) << " ";
            std::cout << std::endl;
        }   
    }
    

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::print_exact_solution() const{

        if constexpr (execution_mode == ExecutionMode::PARALLEL) {
            int mpi_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            if(mpi_rank != 0) return; // Only the root process prints
        }

        if(result.iterartion_residue == -1.){
          std::cout << "Solver was not called" << std::endl; 
          return;
        }

        for(unsigned i = 0; i < data.n; i++){
            for(unsigned j = 0; j < data.n; j++) std::cout << u_exact(i, j) << " ";
            std::cout << std::endl;
        }   

        std::cout << "Residue error: " << result.iterartion_residue << std::endl;
        std::cout << "Total iterations: " << result.iterations << std::endl;
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::export_to_vtk(const eigenMatrix& meshX, const eigenMatrix& meshY, const eigenMatrix& u_h, const std::string& filename) const{
        std::ofstream vtk_file(filename);
        
        if(!vtk_file.is_open()) throw std::runtime_error("Could not open / create file for writing: " + filename);

        vtk_file << "# vtk DataFile Version 3.0\n";
        vtk_file << "Laplacian Solver Output\n";
        vtk_file << "ASCII\n";
        vtk_file << "DATASET STRUCTURED_POINTS\n";

        vtk_file << "DIMENSIONS " << data.n << " " << data.n << " 1\n";

        // E poi si va avanti una volta che la mesh è fatta correttamente
        
    }


}
#endif
