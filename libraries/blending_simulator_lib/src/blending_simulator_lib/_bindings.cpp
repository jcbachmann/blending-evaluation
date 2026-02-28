#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include "BlendingSimulator/BlendingSimulatorFast.h"
#include "BlendingSimulator/BlendingSimulatorDetailed.h"
#include "BlendingSimulator/ParticleParameters.h"

namespace bs = blendingsimulator;

class BlendingSimulatorLibPython
{
	public:
		BlendingSimulatorLibPython(float heapWorldSizeX, float heapWorldSizeZ, float reclaimAngle, float particlesPerCubicMeter, bool circular,
			float eightLikelihood, float bulkDensityFactor, float dropHeight, bool detailed, float reclaimIncrement)
			: reclaimIncrement(reclaimIncrement)
			, verbose(false)
		{
			bs::SimulationParameters simulationParameters{
				heapWorldSizeX,
				heapWorldSizeZ,
				reclaimAngle,
				particlesPerCubicMeter,
				circular,
				eightLikelihood,
				false, // visualize - not available in python library
				bulkDensityFactor,
				dropHeight
			};

			if (detailed) {
				simulator = new bs::BlendingSimulatorDetailed<bs::AveragedParameters>(simulationParameters);
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
			simulator->stack(x, z, bs::AveragedParameters(volume, parameter));
		}

		void stackList(const py::array_t<double>& data, const std::vector<std::string>& columns)
		{
//			int timestampCol = -1;
			int xCol = -1;
			int zCol = -1;
			int volumeCol = -1;
			std::vector<int> parameterColumnIndices;

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
					parameterColumns.push_back(columns[i]);
				}
			}

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
			const float* heapMap = simulator->getHeapMap(); // +1 for Y coordinate
			for (int z = 0; z < heapMapSize.second; z++) {
				heights.emplace_back(heapMap + z * heapMapSize.first + 1, heapMap + z * heapMapSize.first + 1 + heapMapSize.first);
			}

			return heights;
		}

	private:
		bs::BlendingSimulator<bs::AveragedParameters>* simulator;
		float reclaimIncrement;
		std::vector<std::string> parameterColumns;
		bool verbose;

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
		.def(py::init<float, float, float, float, bool, float, float, float, bool, float>())
		.def("stack", &BlendingSimulatorLibPython::stack)
		.def("stack_list", &BlendingSimulatorLibPython::stackList)
		.def("reclaim", &BlendingSimulatorLibPython::reclaim)
		.def("get_heights", &BlendingSimulatorLibPython::getHeights);
}
