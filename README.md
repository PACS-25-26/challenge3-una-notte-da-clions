# Challenge 3: 2D Laplacian Solver Suite

C++ project for the numerical solution of the 2D Poisson/Laplacian equation on a structured grid.

## Overview

This project implements a solver suite for elliptic boundary value problems on a square domain, with support for:

- **Dirichlet**, **Neumann**, and **Robin** boundary conditions
- iterative solvers **Jacobi** and **Schwarz**
- **sequential** and **parallel** execution using **MPI** + **OpenMP**
- verification tests with analytical solutions and convergence monitoring

## Main structure

- `main.cpp` — initializes MPI/OpenMP and runs the test cases
- `include/laplacian_solvers.hpp` — solver classes, data structures, and algorithm definitions
- `include/laplacian_solvers_test_1.hpp` — main verification tests and baseline cases
- `include/laplacian_solvers_test_2.hpp` — advanced tests and non-homogeneous boundary cases
- `Makefile` — build rules using `mpic++`
- `executables/` — target directory for the compiled executable

## Requirements

- C++ compiler with **C++23** support
- **MPI** runtime installed
- **OpenMP** support
- `mpic++` available in the PATH

## Build

From the project root:

```bash
make clean && make
```

This command builds the executable at `executables/exe`.

### Build modes

- `make` — standard build
- `make debug` — build with `-Wall -g -O0`
- `make release` — build with `-O3 -DNDEBUG`

## Run

Run the executable with MPI:

```bash
mpirun -np 4 ./executables/exe
```

The program executes a predefined sequence of test cases covering boundary conditions and iterative methods.

## OpenMP settings

In the `Makefile`, you can configure the total number of OpenMP threads:

- `USE_OPENMP=true`
- `AVAILABLE_THREADS=16`

The number of threads per MPI rank is computed automatically based on the number of launched processes.

## Output

- convergence information, residuals, and iteration counts printed to the console
- VTK output files generated for visualization (e.g. `test_1_dirichlet_sequential.vtk`)

## Test suites

### `include/laplacian_solvers_test_1.hpp`

This file implements the core validation cases for homogeneous boundary conditions and algorithmic comparisons. It includes:

- `run_test_1_dirichlet_sequential` — sequential Jacobi with exact solution `u(x,y) = \sin(\pi x) \sin(\pi y)`
- `run_test_2_neumann_sequential` — sequential Jacobi with exact solution `u(x,y) = \cos(\pi x) \cos(\pi y)`
- `run_test_3_robin_sequential` — sequential Jacobi with exact solution `u(x,y) = e^{x+y}`
- `run_test_4_dirichlet_parallel`, `run_test_5_neumann_parallel`, `run_test_6_robin_parallel` — MPI+OpenMP parallel Jacobi tests
- `run_test_7_dirichlet_parallel_schwarz`, `run_test_8_neumann_parallel_schwarz`, `run_test_9_robin_parallel_schwarz` — parallel Schwarz domain decomposition tests
- `run_test_10_dirichlet_sequential_vs_parallel_jacobi`, `run_test_11_neumann_sequential_vs_parallel_jacobi`, `run_test_12_robin_sequential_vs_parallel_jacobi` — sequential vs parallel convergence comparisons for Jacobi
- `run_test_13_dirichlet_sequential_vs_parallel_schwarz`, `run_test_14_neumann_sequential_vs_parallel_schwarz`, `run_test_15_robin_sequential_vs_parallel_schwarz` — sequential vs parallel convergence comparisons for Schwarz

The mathematical model solved in these tests is:

```tex
-\Delta u(x,y) = f(x,y), \qquad (x,y) \in (0,1)^2
```

with boundary conditions of the form:

```tex
u|_{\partial\Omega} = g_D \quad \text{(Dirichlet)}
```

```tex
\frac{\partial u}{\partial n}\Big|_{\partial\Omega} = g_N \quad \text{(Neumann)}
```

```tex
\alpha\,u + \beta\,\frac{\partial u}{\partial n}\Big|_{\partial\Omega} = g_R \quad \text{(Robin)}
```

The tests use manufactured solutions to derive the source term `f(x,y)` and the boundary values, enabling direct comparison between the numerical and analytical solutions.

### `include/laplacian_solvers_test_2.hpp`

This file adds advanced non-homogeneous benchmark cases and a convergence study framework. It includes:

- `run_test_16_dirichlet_sequential` — Dirichlet test with exact solution `u(x,y) = x^2 y + x y^2 + x + y + 1`
- `run_test_17_neumann_sequential` — Neumann test for the same polynomial exact solution
- `run_test_18_robin_sequential` — Robin test with exact solution `u(x,y) = \sin(x+y) + 2x + 3y`
- `run_test_19_dirichlet_parallel`, `run_test_20_neumann_parallel`, `run_test_21_robin_parallel` — MPI+OpenMP parallel Jacobi versions of the non-homogeneous tests
- `run_test_22_neumann_sequential_vs_parallel_schwarz` — convergence and performance benchmark for Schwarz under Neumann conditions

The convergence benchmark evaluates serial and parallel runs across refinement levels:

```tex
n = 8,\ 16,\ 32,\ 64,\ 128
```

and records:

- L2 error norms of the numerical solution
- iteration counts
- execution time in milliseconds
- parallel speedup ratios

For these cases, the forcing term `f(x,y)` is computed from the exact solution's analytic Laplacian, and boundary conditions are set accordingly.

## Doxygen and LaTeX documentation

This repository already includes generated documentation in both HTML and LaTeX form.

### Open the Doxygen site

Open one of the generated HTML files in your browser:

- `xdg-open html/index.html`

### Open the LaTeX reference manual

Use a PDF viewer to open the generated LaTeX reference manual:

```bash
xdg-open latex/refman.pdf
```

If `latex/refman.pdf` is not built yet, you can generate it from the LaTeX source inside `latex/` using the provided `Makefile`.

## Notes

- `main.cpp` runs a fixed sequence of tests for Jacobi and Schwarz solvers
- the implementation supports both sequential and parallel execution
- boundary conditions and forcing terms are defined analytically within the test cases
