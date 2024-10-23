cmake_minimum_required(VERSION 3.20)
project(petscasm
	LANGUAGES C
	VERSION 0.0.1)

set(CMAKE_COLOR_DIAGNOSTICS ON)

find_package(MPI REQUIRED COMPONENTS C)

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC PETSc>=3.17 IMPORTED_TARGET REQUIRED)

add_library(tlasm tlasm.c)
target_compile_options(tlasm PRIVATE -Wall -Wextra -Wpedantic -pedantic)
target_link_libraries(tlasm PRIVATE PkgConfig::PETSC MPI::MPI_C)

add_executable(main main.c)
target_compile_options(main PRIVATE -Wall -Wextra -Wpedantic -pedantic)
target_link_libraries(main PRIVATE PkgConfig::PETSC MPI::MPI_C tlasm)
