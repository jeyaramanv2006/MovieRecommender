cmake_minimum_required(VERSION 3.15)
project(jakube_bindings LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python for nanobind (use the modern "Python" package with Development.Module)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Add nanobind from ../nanobind directory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../nanobind ${CMAKE_CURRENT_BINARY_DIR}/nanobind)

# Build the Python module (link your .cpp/.h)
nanobind_add_module(
    jakube_ext
    bindings.cpp
)

target_include_directories(jakube_ext PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Link against Python targets; this also propagates include dirs
target_link_libraries(jakube_ext PRIVATE Python::Module)

set_target_properties(jakube_ext PROPERTIES
    OUTPUT_NAME "jakube"
    PREFIX ""
)
