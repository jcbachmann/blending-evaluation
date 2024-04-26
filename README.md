# Blending Evaluation

A collection of Python libraries and apps for simulation, optimization and evaluation of bulk material homogenization.

## How to set up development

Create a virtual environment and install the packages in editable mode by using the requirements-dev.txt

```bash
python -m venv .venv
source .venv/bin/activate.fish
pip install --upgrade pip setuptools wheel
pip install -r requirements-dev.txt
```

## Example script for executing the blending simulator

```python
import pandas as pd

from bmh.benchmark.material_deposition import MaterialDeposition, Material, Deposition
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator

BED_SIZE_X = 300
BED_SIZE_Z = 50

material_deposition = MaterialDeposition(
    material=Material.from_data(pd.DataFrame({
        'timestamp': [0, 1, 2],
        'volume': [1, 1, 1],
        'quality': [1, 2, 3]
    })),
    deposition=Deposition.from_data(
        data=pd.DataFrame({
            'timestamp': [0, 1, 2],
            'x': [50, 150, 250],
            'z': [0.5 * BED_SIZE_Z, 0.5 * BED_SIZE_Z, 0.5 * BED_SIZE_Z]
        }),
        bed_size_x=BED_SIZE_X,
        bed_size_z=BED_SIZE_Z,
        reclaim_x_per_s=1
    )
)

sim = BslBlendingSimulator(
    bed_size_x=BED_SIZE_X,
    bed_size_z=BED_SIZE_Z
)
reclaimed_Material = sim.stack_reclaim(material_deposition)

print(reclaimed_Material.data)
```
