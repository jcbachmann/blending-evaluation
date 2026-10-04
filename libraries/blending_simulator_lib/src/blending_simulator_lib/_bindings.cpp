#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include "BlendingSimulator/BlendingSimulatorFast.h"
#include "BlendingSimulator/BlendingSimulatorLattice.h"
#ifdef BUILD_DETAILED_SIMULATOR
#include "BlendingSimulator/BlendingSimulatorDetailed.h"
#endif
#include "BlendingSimulator/ParticleParameters.h"

namespace bs = blendingsimulator;

class BlendingSimulatorLibPython
{
	public:
		BlendingSimulatorLibPython(float heapWorldSizeX, float heapWorldSizeZ, float reclaimAngle, float particlesPerCubicMeter, bool circular,
			float eightLikelihood, float bulkDensityFactor, float dropHeight, bool detailed, float reclaimIncrement, std::optional<std::uint32_t> seed,
			bool lattice, float latticeAngleOfRepose, bool recordParticles)
			: reclaimIncrement(reclaimIncrement)
			, verbose(false)
		{
			bs::SimulationParameters simulationParameters;
			simulationParameters.heapWorldSizeX = heapWorldSizeX;
			simulationParameters.heapWorldSizeZ = heapWorldSizeZ;
			simulationParameters.reclaimAngle = reclaimAngle;
			simulationParameters.particlesPerCubicMeter = particlesPerCubicMeter;
			simulationParameters.circular = circular;
			simulationParameters.eightLikelihood = eightLikelihood;
			// Without a visualizer, visualize only records the particles for get_particles
			simulationParameters.visualize = recordParticles;
			simulationParameters.bulkDensityFactor = bulkDensityFactor;
			simulationParameters.dropHeight = dropHeight;
			simulationParameters.seed = seed;
			simulationParameters.latticeAngleOfRepose = latticeAngleOfRepose;

			if (detailed && lattice) {
				throw std::invalid_argument("choose either the detailed or the lattice simulator");
			}
			if (lattice) {
				simulator = new bs::BlendingSimulatorLattice<bs::AveragedParameters>(simulationParameters);
			} else if (detailed) {
#ifdef BUILD_DETAILED_SIMULATOR
				simulator = new bs::BlendingSimulatorDetailed<bs::AveragedParameters>(simulationParameters);
#else
				throw std::runtime_error("Detailed simulator not available on this platform.");
#endif
			} else {
				simulator = new bs::BlendingSimulatorFast<bs::AveragedParameters>(simulationParameters);
			}
		}

		~BlendingSimulatorLibPython()
		{
			delete simulator;
		}

		void stack(double timestamp, float x, float z, double volume, const std::vector<double>& parameter)
		{
			useParameterCount(parameter.size());
			simulator->stack(x, z, bs::AveragedParameters(volume, parameter));
		}

		void stackList(const py::array_t<double>& data, const std::vector<std::string>& columns)
		{
//			int timestampCol = -1;
			int xCol = -1;
			int zCol = -1;
			int volumeCol = -1;
			std::vector<int> parameterColumnIndices;
			std::vector<std::string> names;

			for (int i = 0; i < columns.size(); i++) {
				if (columns[i] == "timestamp") {
//					timestampCol = i;
				} else if (columns[i] == "x") {
					xCol = i;
				} else if (columns[i] == "z") {
					zCol = i;
				} else if (columns[i] == "volume") {
					volumeCol = i;
				} else {
					parameterColumnIndices.push_back(i);
					names.push_back(columns[i]);
				}
			}

			if (xCol < 0 || zCol < 0 || volumeCol < 0) {
				throw std::invalid_argument("columns x, z and volume are required");
			}

			useParameterColumns(names);

			auto dataRef = data.unchecked<2>();

			for (py::ssize_t i = 0; i < dataRef.shape(0); i++) {
				std::vector<double> values(parameterColumnIndices.size());
				for (int j = 0; j < parameterColumnIndices.size(); j++) {
					values[j] = dataRef(i, parameterColumnIndices[j]);
				}
				simulator->stack(
					(float)dataRef(i, xCol),
					(float)dataRef(i, zCol),
					bs::AveragedParameters(dataRef(i, volumeCol), values)
				);
			}
		}

		py::dict reclaim()
		{
			finishStacking();

			if (verbose) {
				std::cerr << "Reclaiming" << std::endl;
			}

			py::list x;
			py::list volume;
			std::vector<py::list> parameter(parameterColumns.size());

			float position = 0.0f;
			while (!simulator->reclaimingFinished()) {
				bs::AveragedParameters p = simulator->reclaim(position);
				x.append(position);
				volume.append(p.getVolume());
				const auto& values = p.getValues();
				for (int i = 0; i < parameterColumns.size(); i++) {
					parameter[i].append(i < values.size() ? values[i] : 0);
				}
				position += reclaimIncrement;
			}

			py::dict ret("x"_a = x, "volume"_a = volume);
			for (int i = 0; i < parameterColumns.size(); i++) {
				ret[parameterColumns[i].c_str()] = parameter[i];
			}
			return ret;
		}

		std::vector<std::vector<float>> getHeights()
		{
			finishStacking();

			if (verbose) {
				std::cerr << "Acquiring heights" << std::endl;
			}

			auto heapMapSize = simulator->getHeapMapSize();
			std::vector<std::vector<float>> heights;
			heights.reserve(heapMapSize.second);
			const float* heapMap = simulator->getHeapMap();
			for (unsigned int z = 0; z < heapMapSize.second; z++) {
				heights.emplace_back(heapMap + z * heapMapSize.first, heapMap + (z + 1) * heapMapSize.first);
			}

			return heights;
		}

		// Particles recorded with record_particles: centers, sizes and parameters, as numpy arrays in stacking order
		py::dict getParticles()
		{
			finishStacking();

			std::lock_guard<std::mutex> lock(simulator->outputParticlesMutex);
			const auto& particles = simulator->inactiveOutputParticles;
			const py::ssize_t n = static_cast<py::ssize_t>(particles.size());
			const py::ssize_t m = static_cast<py::ssize_t>(parameterColumns.size());
			py::array_t<double> position({n, py::ssize_t(3)});
			py::array_t<double> size({n, py::ssize_t(3)});
			py::array_t<double> parameters({n, m});
			auto p = position.mutable_unchecked<2>();
			auto s = size.mutable_unchecked<2>();
			auto v = parameters.mutable_unchecked<2>();
			py::ssize_t i = 0;
			for (const auto* particle : particles) {
				p(i, 0) = particle->position.x;
				p(i, 1) = particle->position.y;
				p(i, 2) = particle->position.z;
				s(i, 0) = particle->size.x;
				s(i, 1) = particle->size.y;
				s(i, 2) = particle->size.z;
				for (py::ssize_t j = 0; j < m; j++) {
					v(i, j) = particle->parameters.getValue(static_cast<unsigned int>(j));
				}
				i++;
			}

			py::list columns;
			for (const auto& c : parameterColumns) {
				columns.append(c);
			}
			return py::dict("position"_a = position, "size"_a = size, "parameters"_a = parameters, "columns"_a = columns);
		}

	private:
		bs::BlendingSimulator<bs::AveragedParameters>* simulator;
		float reclaimIncrement;
		std::vector<std::string> parameterColumns;
		bool parameterColumnsKnown = false;
		bool verbose;

		// Parameter names from stack_list, all stacked material has to provide the same parameters
		void useParameterColumns(const std::vector<std::string>& columns)
		{
			if (!parameterColumnsKnown) {
				parameterColumns = columns;
				parameterColumnsKnown = true;
			} else if (columns != parameterColumns) {
				throw std::invalid_argument("parameter columns differ from the material stacked before");
			}
		}

		// Parameters from stack without names are named p_1, p_2, ... like in the CLI reclaim output
		void useParameterCount(std::size_t count)
		{
			if (!parameterColumnsKnown) {
				for (std::size_t i = 0; i < count; i++) {
					parameterColumns.push_back("p_" + std::to_string(i + 1));
				}
				parameterColumnsKnown = true;
			} else if (count != parameterColumns.size()) {
				throw std::invalid_argument(
					"expected " + std::to_string(parameterColumns.size()) + " parameters like the material stacked before, got " + std::to_string(count)
				);
			}
		}

		void finishStacking()
		{
			simulator->finishStacking();

			if (verbose) {
				std::cerr << "Stacking finished" << std::endl;
			}
		}
};

PYBIND11_MODULE(_blending_simulator_lib, m)
{
	m.doc() = "Blending Simulator Lib for Python";

	py::class_<BlendingSimulatorLibPython>(m, "BlendingSimulatorLib")
		.def(
			py::init<float, float, float, float, bool, float, float, float, bool, float, std::optional<std::uint32_t>, bool, float, bool>(),
			"heap_world_size_x"_a,
			"heap_world_size_z"_a,
			"reclaim_angle"_a,
			"particles_per_cubic_meter"_a,
			"circular"_a,
			"eight_likelihood"_a,
			"bulk_density_factor"_a,
			"drop_height"_a,
			"detailed"_a,
			"reclaim_increment"_a,
			"seed"_a = py::none(),
			"lattice"_a = false,
			"lattice_angle_of_repose"_a = 45.0f,
			"record_particles"_a = false
		)
		.def("stack", &BlendingSimulatorLibPython::stack)
		.def("stack_list", &BlendingSimulatorLibPython::stackList)
		.def("reclaim", &BlendingSimulatorLibPython::reclaim)
		.def("get_heights", &BlendingSimulatorLibPython::getHeights)
		.def("get_particles", &BlendingSimulatorLibPython::getParticles);
}
