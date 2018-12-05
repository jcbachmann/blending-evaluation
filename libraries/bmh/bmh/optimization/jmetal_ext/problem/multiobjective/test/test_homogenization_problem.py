#!/usr/bin/env python
import unittest

from bmh.benchmark.material_deposition import Deposition, DepositionMeta
from pandas import DataFrame
from pandas.util.testing import assert_frame_equal

from ..homogenization_problem import variables_to_deposition_generic


class TestVariablesToDepositionGeneric(unittest.TestCase):
    def test_simple(self):
        variables = [0.0, 1.0]
        x_min = 25.0
        x_max = 275.0
        bed_size_x = 300.0
        bed_size_z = 50.0
        z = 0.5 * bed_size_z
        max_timestamp = 10.0
        deposition_meta = DepositionMeta.create_empty(
            bed_size_x=bed_size_x,
            bed_size_z=bed_size_z,
            reclaim_x_per_s=1.0
        )
        deposition_prefix = None

        deposition = variables_to_deposition_generic(
            variables=variables, x_min=x_min, x_max=x_max, max_timestamp=max_timestamp,
            deposition_meta=deposition_meta, deposition_prefix=deposition_prefix
        )

        self.assertEqual(deposition.meta.bed_size_x, bed_size_x)
        self.assertEqual(deposition.meta.bed_size_z, bed_size_z)
        self.assertEqual(deposition.meta.time, max_timestamp)

        reference_data = DataFrame({
            'timestamp': [0, max_timestamp],
            'x': [x_min, x_max],
            'z': [z, z],
        })

        assert_frame_equal(reference_data, deposition.data)

    def test_prefix_empty(self):
        variables = [0.0, 1.0]
        x_min = 25.0
        x_max = 275.0
        bed_size_x = 300.0
        bed_size_z = 50.0
        z = 0.5 * bed_size_z
        max_timestamp = 10.0
        deposition_meta = DepositionMeta.create_empty(
            bed_size_x=bed_size_x,
            bed_size_z=bed_size_z,
            reclaim_x_per_s=1.0
        )
        deposition_prefix = Deposition.create_empty(bed_size_x=bed_size_x, bed_size_z=bed_size_z, reclaim_x_per_s=1.0)

        deposition = variables_to_deposition_generic(
            variables=variables, x_min=x_min, x_max=x_max, max_timestamp=max_timestamp,
            deposition_meta=deposition_meta, deposition_prefix=deposition_prefix
        )

        self.assertEqual(deposition.meta.bed_size_x, bed_size_x)
        self.assertEqual(deposition.meta.bed_size_z, bed_size_z)
        self.assertEqual(deposition.meta.time, max_timestamp)

        reference_data = DataFrame({
            'timestamp': [0, max_timestamp],
            'x': [x_min, x_max],
            'z': [z, z],
        })

        assert_frame_equal(reference_data, deposition.data)

    def test_prefix_simple(self):
        variables = [0.0, 1.0]
        x_min = 25.0
        x_max = 275.0
        bed_size_x = 300.0
        bed_size_z = 50.0
        z = 0.5 * bed_size_z
        max_timestamp = 10.0
        deposition_meta = DepositionMeta.create_empty(
            bed_size_x=bed_size_x,
            bed_size_z=bed_size_z,
            reclaim_x_per_s=1.0
        )
        deposition_prefix = Deposition.create_empty(bed_size_x=bed_size_x, bed_size_z=bed_size_z, reclaim_x_per_s=1.0)
        deposition_prefix.data = DataFrame({
            'timestamp': [0, 1.0 / 3.0 * max_timestamp],
            'x': [x_min, x_max],
            'z': [z, z],
        })

        deposition = variables_to_deposition_generic(
            variables=variables, x_min=x_min, x_max=x_max, max_timestamp=max_timestamp,
            deposition_meta=deposition_meta, deposition_prefix=deposition_prefix
        )

        self.assertEqual(deposition.meta.bed_size_x, bed_size_x)
        self.assertEqual(deposition.meta.bed_size_z, bed_size_z)
        self.assertEqual(deposition.meta.time, max_timestamp)

        reference_data = DataFrame({
            'timestamp': [0, 1.0 / 3.0 * max_timestamp, 2.0 / 3.0 * max_timestamp, max_timestamp],
            'x': [x_min, x_max, x_min, x_max],
            'z': [z, z, z, z],
        })

        assert_frame_equal(reference_data, deposition.data)
