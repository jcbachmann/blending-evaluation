cmake_minimum_required(VERSION 3.15)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FetchContent_Declare(
    blending-simulation
    GIT_REPOSITORY https://github.com/jcbachmann/blending-simulation
    GIT_TAG v2026.1
)
FetchContent_MakeAvailable(blending-simulation)

pybind11_add_module(
    _blending_simulator_lib
    src/blending_simulator_lib/_bindings.cpp
)

set_target_properties(
    _blending_simulator_lib PROPERTIES
    CXX_STANDARD_REQUIRED 17
)

target_link_libraries(
    _blending_simulator_lib PRIVATE
    BlendingSimulator::Lib
    BlendingSimulator::FastLib
    pybind11::pybind11
)

if(BUILD_DETAILED_SIMULATOR)
    # Disabled for Apple
    target_link_libraries(_blending_simulator_lib PRIVATE BlendingSimulator::DetailedLib)
    target_compile_definitions(_blending_simulator_lib PRIVATE BUILD_DETAILED_SIMULATOR)
endif()

install(TARGETS _blending_simulator_lib DESTINATION ${SKBUILD_PROJECT_NAME})
