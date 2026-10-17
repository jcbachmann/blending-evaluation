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

# The simulator release the bindings are built from, available as blending_simulator_lib.simulator_version: bmh_ml records it with every
# simulation result, so that results of different simulator versions are never mixed
set(BLENDING_SIMULATOR_TAG v2026.2)

FetchContent_Declare(
    blending-simulation
    GIT_REPOSITORY https://github.com/jcbachmann/blending-simulation
    GIT_TAG ${BLENDING_SIMULATOR_TAG}
)
FetchContent_MakeAvailable(blending-simulation)

string(REGEX REPLACE "^v" "" BLENDING_SIMULATOR_VERSION ${BLENDING_SIMULATOR_TAG})
if(FETCHCONTENT_SOURCE_DIR_BLENDING-SIMULATION)
    # Built from a local checkout, which may hold any state of the simulator: never the release itself, but its commit if it is known
    # (with "dirty" for uncommitted changes), so that results of two local states are not taken for the same simulator
    execute_process(
        COMMAND git describe --always --dirty=.dirty --abbrev=7 --exclude=*
        WORKING_DIRECTORY ${FETCHCONTENT_SOURCE_DIR_BLENDING-SIMULATION}
        OUTPUT_VARIABLE BLENDING_SIMULATOR_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    string(APPEND BLENDING_SIMULATOR_VERSION "+local")
    if(BLENDING_SIMULATOR_COMMIT)
        string(APPEND BLENDING_SIMULATOR_VERSION ".${BLENDING_SIMULATOR_COMMIT}")
    endif()
endif()

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
    BlendingSimulator::LatticeLib
    BlendingSimulator::ConeLib
    pybind11::pybind11
)

target_compile_definitions(_blending_simulator_lib PRIVATE BLENDING_SIMULATOR_VERSION="${BLENDING_SIMULATOR_VERSION}")

if(BUILD_DETAILED_SIMULATOR)
    target_link_libraries(_blending_simulator_lib PRIVATE BlendingSimulator::DetailedLib)
    target_compile_definitions(_blending_simulator_lib PRIVATE BUILD_DETAILED_SIMULATOR)
endif()

# Simulators from 2026.3 on simulate in detail with the discrete element method and take its parameters (detailed_parameters)
if(EXISTS "${blending-simulation_SOURCE_DIR}/BlendingSimulatorDetailedLib/include/BlendingSimulator/dem/World.h")
    target_compile_definitions(_blending_simulator_lib PRIVATE BLENDING_SIMULATOR_DEM)
endif()

install(TARGETS _blending_simulator_lib DESTINATION ${SKBUILD_PROJECT_NAME})
