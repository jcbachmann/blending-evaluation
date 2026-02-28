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
    GIT_TAG master
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
    BlendingSimulator::DetailedLib
    pybind11::pybind11
)

install(TARGETS _blending_simulator_lib DESTINATION ${SKBUILD_PROJECT_NAME})
