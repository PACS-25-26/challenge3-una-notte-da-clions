#ifndef LAPLACIAN_SOLVERS_POSTPROCESSING_HPP
#define LAPLACIAN_SOLVERS_POSTPROCESSING_HPP

#include "laplacian_solvers.hpp"

namespace laplacian_solvers{

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::print() const{
        print_mesh();
        print_solution();
        print_exact_solution();
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::print_mesh() const{

        eigenMatrix meshX_global(data.n, data.n), meshY_global(data.n, data.n);
        
        if constexpr (execution_mode == ExecutionMode::PARALLEL) { // Gather data from other processes

            int mpi_rank, mpi_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

            const unsigned remainder_rows = data.n % mpi_size;
            const unsigned local_rows = data.n / mpi_size + (mpi_rank < remainder_rows ? 1 : 0); 

            std::vector<int> recv_counts(mpi_size);
            std::vector<int> displs(mpi_size);

            for(unsigned i = 0; i < mpi_size; i++){
                recv_counts[i] = data.n * (data.n / mpi_size + (i < remainder_rows? 1: 0));
                displs[i] = i == 0? 0: displs[i-1] + recv_counts[i-1];
            }

            MPI_Gatherv(meshX.data(), local_rows * data.n, MPI_DOUBLE, meshX_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(meshY.data(), local_rows * data.n, MPI_DOUBLE, meshY_global.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
         

        } else{
            meshX_global = meshX;
            meshY_global = meshY;
        }

        if(mpi_rank != 0) return; // Only the root process
        std::cout << "Mesh points (x, y):" << std::endl;

        for(unsigned i = 0; i < data.n; i++){
            for(unsigned j = 0; j < data.n; j++) std::cout << "(" << meshX_global(i, j) << ", " << meshY_global(i, j) << ") ";
             std::cout << std::endl;
        }
       
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::print_solution() const{
        
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

        if(mpi_rank != 0) return;

        std::cout << "Discrete solution u_h:" << std::endl;

        if(result.iterartion_residue == -1.) {
          std::cout << "Solver was not called" << std::endl; 
          return;
        }
    
        for(unsigned i = 0; i < data.n; i++){
            for(unsigned j = 0; j < data.n; j++) std::cout << result.u_h(i, j) << " ";
            std::cout << std::endl;
        }   

        std::cout << "Residue error: " << result.iterartion_residue << std::endl;
        std::cout << "Total iterations: " << result.iterations << std::endl;
    }
    

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::print_exact_solution() const{

        eigenMatrix exact_solution(data.n, data.n);

        if constexpr (execution_mode == ExecutionMode::PARALLEL) { // Gather data from other processes

            int mpi_rank, mpi_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

            const unsigned remainder_rows = data.n % mpi_size;
            const unsigned local_rows = data.n / mpi_size + (mpi_rank < remainder_rows ? 1 : 0); 

            std::vector<int> recv_counts(mpi_size);
            std::vector<int> displs(mpi_size);

            for(unsigned i = 0; i < mpi_size; i++){
                recv_counts[i] = data.n * (data.n / mpi_size + (i < remainder_rows? 1: 0));
                displs[i] = i == 0? 0: displs[i-1] + recv_counts[i-1];
            }

            MPI_Gatherv(u_exact.data(), local_rows * data.n, MPI_DOUBLE, exact_solution.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        } else{
            exact_solution = u_exact;
        }

        if(mpi_rank != 0) return; // Only the root process

        std::cout << "Exact solution u_exact:" << std::endl;
        
        for(unsigned i = 0; i < data.n; i++){
            for(unsigned j = 0; j < data.n; j++) std::cout << exact_solution(i, j) << " ";
            std::cout << std::endl;
        }   
    }

    template <SolverType solver_type, BoundaryCondition boundary_condition, ExecutionMode execution_mode, typename funcType>
    void Laplacian_Solver<solver_type, boundary_condition, execution_mode, funcType>::export_to_vtk(const std::string& filename) const {
        std::ofstream vtk_file(filename);
        
        if(!vtk_file.is_open()) {
            throw std::runtime_error("Could not open / create file for writing: " + filename);
        }

        // 1. VTK Standard Header
        vtk_file << "# vtk DataFile Version 3.0\n";
        vtk_file << "Laplacian Solver Output\n";
        vtk_file << "ASCII\n";
        
        // Using STRUCTURED_GRID because coordinates are explicitly stored in meshX and meshY
        vtk_file << "DATASET STRUCTURED_GRID\n";

        // 2. Grid Topology Definition
        vtk_file << "DIMENSIONS " << data.n << " " << data.n << " 1\n";
        
        size_t total_points = data.n * data.n;
        vtk_file << "POINTS " << total_points << " double\n";

        // Write grid coordinates.
        // Iterating row by row (j) and then column by column (i).
        // Since matrices are Row-Major, this layout ensures contiguous memory access 
        // and matches the VTK format requirements (X coordinate varies fastest).
        for (unsigned j = 0; j < data.n; ++j) {       // Row index (Y-direction)
            for (unsigned i = 0; i < data.n; ++i) {   // Column index (X-direction)
                vtk_file << meshX(j, i) << " " << meshY(j, i) << " 0.0\n";
            }
        }

        // 3. Point-Data Section (Datasets associated with nodes)
        vtk_file << "POINT_DATA " << total_points << "\n";
        
        // Field 1: Computed Numerical Solution (u_h)
        vtk_file << "SCALARS Numerical_Solution float 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for (unsigned j = 0; j < data.n; ++j) {
            for (unsigned i = 0; i < data.n; ++i) {
                vtk_file << u_h(j, i) << "\n";
            }
        }

        // Field 2: Analytical / Exact Solution (u_exact)
        vtk_file << "SCALARS Exact_Solution float 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for (unsigned j = 0; j < data.n; ++j) {
            for (unsigned i = 0; i < data.n; ++i) {
                vtk_file << u_exact(j, i) << "\n";
            }
        }

        vtk_file.close();
    }


}
#endif
