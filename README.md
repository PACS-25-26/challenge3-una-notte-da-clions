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

### VTK visualization

The solver writes solution output in VTK format, which can be opened with ParaView for interactive visualization.

## Test Suites

The integrated test suite (`laplacian_solvers_test_1.hpp` and `laplacian_solvers_test_2.hpp`) verifies correctness, measures parallel efficiency, and analyses convergence of the implemented solvers. It covers three boundary condition types – Dirichlet, Neumann, and Robin – for both the **Jacobi** and **Schwarz** (domain decomposition) solvers, in **sequential** and **MPI‑parallel** execution modes.

Each test constructs a known analytical solution \( u_{\text{exact}}(x,y) \), derives the corresponding source term \( f_0 = -\Delta u_{\text{exact}} \) and boundary data, then compares the numerical solution against the exact one. Convergence tests sweep over increasing mesh resolutions to compute experimental orders of convergence (EOC) and speedup profiles.

### Exact Solutions & Laplacians

| Test group | Exact solution \( u(x,y) \) | Source term \( f_0(x,y) = -\Delta u \) |
| 1,4,7,10,13 | \( \sin(\pi x)\sin(\pi y) \) | \( 2\pi^2 \sin(\pi x)\sin(\pi y) \) |
| 2,5,8,11,14 | \( \cos(\pi x)\cos(\pi y) \) | \( 2\pi^2 \cos(\pi x)\cos(\pi y) \) |
| 3,6,9,12,15 | \( e^{x+y} \) | \( -2e^{x+y} \) |
| 16,17,19,20,22 | \( x^2y + xy^2 + x + y + 1 \) | \( -2(x+y) \) |
| 18,21 | \( \sin(x+y) + 2x + 3y \) | \( 2\sin(x+y) \) |

### Boundary Conditions

- **Dirichlet** – Prescribed value \( u = g \) on \(\partial\Omega\).
- **Neumann** – Prescribed normal derivative \( \frac{\partial u}{\partial n} = g \) on \(\partial\Omega\).
- **Robin** – Linear combination \( u + \frac{\partial u}{\partial n} = g \) on \(\partial\Omega\).

### List of Tests

| Test | Solver | Mode | BC | Exact solution | Purpose |
| 1 | Jacobi | Sequential | Dirichlet | \( \sin(\pi x)\sin(\pi y) \) | Baseline validation |
| 2 | Jacobi | Sequential | Neumann | \( \cos(\pi x)\cos(\pi y) \) | Neumann consistency |
| 3 | Jacobi | Sequential | Robin | \( e^{x+y} \) | Robin verification |
| 4 | Jacobi | Parallel (MPI) | Dirichlet | \( \sin(\pi x)\sin(\pi y) \) | Parallel correctness |
| 5 | Jacobi | Parallel | Neumann | \( \cos(\pi x)\cos(\pi y) \) | Parallel Neumann |
| 6 | Jacobi | Parallel | Robin | \( e^{x+y} \) | Parallel Robin |
| 7 | Schwarz | Parallel | Dirichlet | \( \sin(\pi x)\sin(\pi y) \) | Domain decomposition |
| 8 | Schwarz | Parallel | Neumann | \( \cos(\pi x)\cos(\pi y) \) | DD with Neumann |
| 9 | Schwarz | Parallel | Robin | \( e^{x+y} \) | DD with Robin |
| 10 | Jacobi | Sequential *vs* Parallel | Dirichlet | \( \sin(\pi x)\sin(\pi y) \) | Convergence & speedup |
| 11 | Jacobi | Sequential *vs* Parallel | Neumann | \( \cos(\pi x)\cos(\pi y) \) | Convergence & speedup |
| 12 | Jacobi | Sequential *vs* Parallel | Robin | \( e^{x+y} \) | Convergence & speedup |
| 13 | Schwarz | Sequential *vs* Parallel | Dirichlet | \( \sin(\pi x)\sin(\pi y) \) | DD convergence |
| 14 | Schwarz | Sequential *vs* Parallel | Neumann | \( \cos(\pi x)\cos(\pi y) \) | DD Neumann convergence |
| 15 | Schwarz | Sequential *vs* Parallel | Robin | \( e^{x+y} \) | DD Robin convergence |
| 16 | Jacobi | Sequential | Dirichlet | \( x^2y + xy^2 + x + y + 1 \) | Non‑homogeneous polynomial |
| 17 | Jacobi | Sequential | Neumann | \( x^2y + xy^2 + x + y + 1 \) | Neumann polynomial |
| 18 | Jacobi | Sequential | Robin | \( \sin(x+y) + 2x + 3y \) | Robin trigonometric |
| 19 | Jacobi | Parallel | Dirichlet | polynomial | Parallel polynomial |
| 20 | Jacobi | Parallel | Neumann | polynomial | Parallel Neumann polynomial |
| 21 | Jacobi | Parallel | Robin | trigonometric | Parallel Robin trigonometric |
| 22 | Schwarz | Sequential *vs* Parallel | Neumann | polynomial | DD convergence (polynomial) |

All tests export the computed solution to **VTK** files for visual inspection. Convergence tests report \( L_2 \) errors, experimental order of convergence, and parallel speedup relative to the sequential baseline.

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
