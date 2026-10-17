#include <atomic>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include "BlendingSimulator/BlendingSimulatorFast.h"
#include "BlendingSimulator/BlendingSimulatorCone.h"
#include "BlendingSimulator/ContinuousDrops.h"
#include "BlendingSimulator/BlendingSimulatorLattice.h"
#ifdef BUILD_DETAILED_SIMULATOR
#include "BlendingSimulator/BlendingSimulatorDetailed.h"
#endif
#include "BlendingSimulator/ParticleParameters.h"

namespace
{
namespace bs_ = blendingsimulator;

// Sets the fields of the detailed simulation's parameters from a dict of snake_case names; unknown names are an error
void applyDetailedParameters(bs_::SimulationParameters& simulationParameters, const std::optional<py::dict>& values)
{
	if (!values) {
		return;
	}
#ifdef BLENDING_SIMULATOR_DEM
	bs_::DetailedParameters& d = simulationParameters.detailed;
	for (const auto& [key, value] : *values) {
		const std::string name = py::str(key);
		if (name == "threads") {
			d.threads = value.cast<int>();
			continue;
		}
		const double v = value.cast<double>();
		if (name == "flow_rate") d.flowRate = v;
		else if (name == "discharge_height") d.dischargeHeight = v;
		else if (name == "belt_speed") d.beltSpeed = v;
		else if (name == "belt_width") d.beltWidth = v;
		else if (name == "packing_fraction") d.packingFraction = v;
		else if (name == "size_spread") d.sizeSpread = v;
		else if (name == "density") d.density = v;
		else if (name == "youngs_modulus") d.youngsModulus = v;
		else if (name == "poisson_ratio") d.poissonRatio = v;
		else if (name == "restitution") d.restitution = v;
		else if (name == "friction") d.friction = v;
		else if (name == "rolling_friction") d.rollingFriction = v;
		else if (name == "rolling_damping") d.rollingDamping = v;
		else if (name == "time_step_factor") d.timeStepFactor = v;
		else if (name == "rest_velocity") d.restVelocity = v;
		else if (name == "rest_time") d.restTime = v;
		else if (name == "skin_factor") d.skinFactor = v;
		else throw std::invalid_argument("unknown detailed parameter " + name);
	}
#else
	(void)simulationParameters;
	if (!values->empty()) {
		throw std::runtime_error("detailed_parameters need a simulator with the discrete element method (2026.3 or later)");
	}
#endif
}
}

namespace bs = blendingsimulator;

class BlendingSimulatorLibPython
{
	public:
		BlendingSimulatorLibPython(float heapWorldSizeX, float heapWorldSizeZ, float reclaimAngle, float particlesPerCubicMeter, bool circular,
			float eightLikelihood, float bulkDensityFactor, float dropHeight, bool detailed, float reclaimIncrement, std::optional<std::uint32_t> seed,
			bool lattice, bool recordParticles, bool cone, std::optional<py::dict> detailedParameters)
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
			applyDetailedParameters(simulationParameters, detailedParameters);

			if (int(detailed) + int(lattice) + int(cone) > 1) {
				throw std::invalid_argument("choose one of the detailed, the lattice and the cone simulator");
			}
			if (cone) {
				simulator = new bs::BlendingSimulatorCone<bs::AveragedParameters>(simulationParameters);
			} else if (lattice) {
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

		// Heights of the stockpile; with finish_stacking (the default) after the material in motion has come to rest. Without it the
		// current state, so that looking at the stockpile while stacking does not pause the stacker of the detailed simulation
		std::vector<std::vector<float>> getHeights(bool finish)
		{
			if (finish) {
				finishStacking();
			}

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

// Stacks and reclaims many inputs in one call: no Python between the simulations, and with several threads the GIL is released and every
// thread simulates whole inputs. `data` has the shape (inputs, rows, columns) with the columns x, z and volume, a parameter column for
// every further name and optionally a timestamp column, which is ignored. Input i is seeded with seed + i if a seed is given.
py::dict simulateBatch(const py::array_t<double, py::array::c_style | py::array::forcecast>& data, const std::vector<std::string>& columns,
	float heapWorldSizeX, float heapWorldSizeZ, float reclaimAngle, float particlesPerCubicMeter, bool circular, float eightLikelihood,
	float reclaimIncrement, std::optional<std::uint32_t> seed, bool lattice, int threads, bool cone, bool detailed,
	std::optional<py::dict> detailedParameters)
{
	if (int(detailed) + int(lattice) + int(cone) > 1) {
		throw std::invalid_argument("choose one of the detailed, the lattice and the cone simulator");
	}
#ifndef BUILD_DETAILED_SIMULATOR
	if (detailed) {
		throw std::runtime_error("Detailed simulator not available on this platform.");
	}
#endif
	if (data.ndim() != 3 || data.shape(2) != py::ssize_t(columns.size())) {
		throw std::invalid_argument("data must have the shape (inputs, rows, columns) with one name per column");
	}
	if (reclaimIncrement <= 0.0f) {
		throw std::invalid_argument("reclaim_increment must be positive");
	}

	int xCol = -1;
	int zCol = -1;
	int volumeCol = -1;
	std::vector<int> parameterCols;
	std::vector<std::string> names;
	for (int i = 0; i < int(columns.size()); i++) {
		if (columns[i] == "x") {
			xCol = i;
		} else if (columns[i] == "z") {
			zCol = i;
		} else if (columns[i] == "volume") {
			volumeCol = i;
		} else if (columns[i] != "timestamp") {
			parameterCols.push_back(i);
			names.push_back(columns[i]);
		}
	}
	if (xCol < 0 || zCol < 0 || volumeCol < 0) {
		throw std::invalid_argument("columns x, z and volume are required");
	}

	const py::ssize_t inputs = data.shape(0);
	const py::ssize_t rows = data.shape(1);
	const py::ssize_t cols = data.shape(2);
	const std::size_t parameterCount = parameterCols.size();
	const double* values = data.data();

	bs::SimulationParameters simulationParameters;
	simulationParameters.heapWorldSizeX = heapWorldSizeX;
	simulationParameters.heapWorldSizeZ = heapWorldSizeZ;
	simulationParameters.reclaimAngle = reclaimAngle;
	simulationParameters.particlesPerCubicMeter = particlesPerCubicMeter;
	simulationParameters.circular = circular;
	simulationParameters.eightLikelihood = eightLikelihood;
	applyDetailedParameters(simulationParameters, detailedParameters);

	// Per input: reclaimed volume per slice, then the parameters per slice (slice-major)
	std::vector<std::vector<double>> volumes(inputs);
	std::vector<std::vector<double>> parameters(inputs);
	std::vector<float> positions;

	auto simulate = [&](py::ssize_t i) {
		bs::SimulationParameters p = simulationParameters;
		if (seed) {
			p.seed = std::uint32_t(*seed + std::uint64_t(i));
		}
		std::unique_ptr<bs::BlendingSimulator<bs::AveragedParameters>> simulator;
		if (cone) {
			simulator = std::make_unique<bs::BlendingSimulatorCone<bs::AveragedParameters>>(p);
		} else if (lattice) {
			simulator = std::make_unique<bs::BlendingSimulatorLattice<bs::AveragedParameters>>(p);
		} else if (detailed) {
#ifdef BUILD_DETAILED_SIMULATOR
			simulator = std::make_unique<bs::BlendingSimulatorDetailed<bs::AveragedParameters>>(p);
#endif
		} else {
			simulator = std::make_unique<bs::BlendingSimulatorFast<bs::AveragedParameters>>(p);
		}

		std::vector<double> rowParameters(parameterCount);
		for (py::ssize_t r = 0; r < rows; r++) {
			const double* row = values + (i * rows + r) * cols;
			for (std::size_t j = 0; j < parameterCount; j++) {
				rowParameters[j] = row[parameterCols[j]];
			}
			simulator->stack(float(row[xCol]), float(row[zCol]), bs::AveragedParameters(row[volumeCol], rowParameters));
		}
		simulator->finishStacking();

		std::vector<float> reclaimPositions;
		float position = 0.0f;
		while (!simulator->reclaimingFinished()) {
			const bs::AveragedParameters slice = simulator->reclaim(position);
			volumes[i].push_back(slice.getVolume());
			const std::vector<double> sliceValues = slice.getValues();
			for (std::size_t j = 0; j < parameterCount; j++) {
				parameters[i].push_back(j < sliceValues.size() ? sliceValues[j] : 0.0);
			}
			reclaimPositions.push_back(position);
			position += reclaimIncrement;
		}
		if (i == 0) {
			positions = std::move(reclaimPositions);
		}
	};

	{
		py::gil_scoped_release release;
		const int workers = int(std::min<py::ssize_t>(std::max(1, threads > 0 ? threads : int(std::thread::hardware_concurrency())), inputs));
		std::atomic<py::ssize_t> next{0};
		std::exception_ptr error;
		std::mutex errorMutex;
		auto work = [&]() {
			try {
				for (py::ssize_t i = next++; i < inputs; i = next++) {
					simulate(i);
				}
			} catch (...) {
				std::lock_guard<std::mutex> lock(errorMutex);
				error = std::current_exception();
				next = inputs;
			}
		};
		std::vector<std::thread> pool;
		for (int t = 1; t < workers; t++) {
			pool.emplace_back(work);
		}
		work();
		for (std::thread& thread : pool) {
			thread.join();
		}
		if (error) {
			std::rethrow_exception(error);
		}
	}

	const py::ssize_t slices = py::ssize_t(positions.size());
	py::array_t<double> x(slices);
	py::array_t<double> volume({inputs, slices});
	std::vector<py::array_t<double>> parameterArrays;
	for (std::size_t j = 0; j < parameterCount; j++) {
		parameterArrays.emplace_back(std::vector<py::ssize_t>{inputs, slices});
	}
	auto xOut = x.mutable_unchecked<1>();
	auto volumeOut = volume.mutable_unchecked<2>();
	for (py::ssize_t s = 0; s < slices; s++) {
		xOut(s) = positions[s];
	}
	for (py::ssize_t i = 0; i < inputs; i++) {
		if (py::ssize_t(volumes[i].size()) != slices) {
			throw std::runtime_error("inputs reclaimed into different numbers of slices");
		}
		for (py::ssize_t s = 0; s < slices; s++) {
			volumeOut(i, s) = volumes[i][s];
		}
		for (std::size_t j = 0; j < parameterCount; j++) {
			auto out = parameterArrays[j].mutable_unchecked<2>();
			for (py::ssize_t s = 0; s < slices; s++) {
				out(i, s) = parameters[i][s * parameterCount + j];
			}
		}
	}

	py::dict result("x"_a = x, "volume"_a = volume);
	for (std::size_t j = 0; j < parameterCount; j++) {
		result[names[j].c_str()] = parameterArrays[j];
	}
	return result;
}

// Drops of a material flowing while the stacker moves (BlendingSimulator/ContinuousDrops.h): rows x, z, volume and the parameters of each
// drop's portion, shape (drops, 3 + parameters)
py::array_t<double> continuousDropsArray(const std::vector<double>& volumes,
	const py::array_t<double, py::array::c_style | py::array::forcecast>& parameters, const std::vector<double>& portionTimes,
	const std::vector<double>& pathTimes, const std::vector<double>& pathX, const std::vector<double>& pathZ, double cellX, double cellZ)
{
	if (parameters.ndim() != 2 || parameters.shape(0) != py::ssize_t(volumes.size())) {
		throw std::invalid_argument("parameters must have the shape (portions, parameters)");
	}
	const std::vector<bs::ContinuousDrop> drops = bs::continuousDrops(volumes, portionTimes, pathTimes, pathX, pathZ, cellX, cellZ);
	const py::ssize_t k = parameters.shape(1);
	py::array_t<double> result({py::ssize_t(drops.size()), 3 + k});
	auto out = result.mutable_unchecked<2>();
	auto in = parameters.unchecked<2>();
	for (py::ssize_t r = 0; r < py::ssize_t(drops.size()); r++) {
		out(r, 0) = drops[r].x;
		out(r, 1) = drops[r].z;
		out(r, 2) = drops[r].volume;
		for (py::ssize_t j = 0; j < k; j++) {
			out(r, 3 + j) = in(drops[r].portion, j);
		}
	}
	return result;
}

PYBIND11_MODULE(_blending_simulator_lib, m)
{
	m.doc() = "Blending Simulator Lib for Python";
	m.attr("simulator_version") = BLENDING_SIMULATOR_VERSION;

	py::class_<BlendingSimulatorLibPython>(m, "BlendingSimulatorLib")
		.def(
			py::init<float, float, float, float, bool, float, float, float, bool, float, std::optional<std::uint32_t>, bool, bool, bool,
				std::optional<py::dict>>(),
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
			"record_particles"_a = false,
			"cone"_a = false,
			"detailed_parameters"_a = py::none()
		)
		.def("stack", &BlendingSimulatorLibPython::stack)
		.def("stack_list", &BlendingSimulatorLibPython::stackList)
		.def("reclaim", &BlendingSimulatorLibPython::reclaim)
		.def("get_heights", &BlendingSimulatorLibPython::getHeights, "finish_stacking"_a = true)
		.def("get_particles", &BlendingSimulatorLibPython::getParticles);

	m.def(
		"simulate_batch",
		&simulateBatch,
		"data"_a,
		"columns"_a,
		"heap_world_size_x"_a,
		"heap_world_size_z"_a,
		"reclaim_angle"_a,
		"particles_per_cubic_meter"_a,
		"circular"_a,
		"eight_likelihood"_a,
		"reclaim_increment"_a,
		"seed"_a = py::none(),
		"lattice"_a = false,
		"threads"_a = 1,
		"cone"_a = false,
		"detailed"_a = false,
		"detailed_parameters"_a = py::none()
	);

	m.def(
		"continuous_drops",
		&continuousDropsArray,
		"volumes"_a,
		"parameters"_a,
		"portion_times"_a,
		"path_times"_a,
		"path_x"_a,
		"path_z"_a,
		"cell_x"_a,
		"cell_z"_a
	);
}
