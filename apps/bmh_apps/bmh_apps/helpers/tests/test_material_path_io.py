#!/usr/bin/env python
import unittest

import pandas as pd
from pandas import DataFrame

from bmh_apps.helpers.material_path_io import merge_material_path


class TestMergeMaterialPath(unittest.TestCase):
    def test_one(self):
        material = DataFrame(data=[[0.0, 1.0, 0.25]], columns=['timestamp', 'volume', 'p_1'])
        path = DataFrame(data=[0.0, 1.0], columns=['path'])

        material_path = merge_material_path(
            depth=0,
            length=2,
            material=material,
            path=path,
        )

        pd.testing.assert_frame_equal(
            material_path,
            DataFrame(
                [[0.0, 2.0, 0.0, 1.0, 0.25]],
                columns=['timestamp', 'x', 'z', 'volume', 'p_1']
            ),
            check_like=True
        )

    def test_two(self):
        material = DataFrame(data=[[0.0, 1.0, 0.25], [1.0, 1.0, 0.75]], columns=['timestamp', 'volume', 'p_1'])
        path = DataFrame(data=[0.0, 1.0], columns=['path'])

        material_path = merge_material_path(
            depth=0,
            length=2,
            material=material,
            path=path,
        )

        pd.testing.assert_frame_equal(
            material_path,
            DataFrame(
                [[0.0, 0.0, 0.0, 1.0, 0.25], [1.0, 2.0, 0.0, 1.0, 0.75]],
                columns=['timestamp', 'x', 'z', 'volume', 'p_1']
            ),
            check_like=True
        )

    def test_interpolate(self):
        material = DataFrame(data=[[0.0, 1.0, 0.25], [1.0, 1.0, 0.1], [2.0, 1.0, 0.75]],
                             columns=['timestamp', 'volume', 'p_1'])
        path = DataFrame(data=[0.0, 1.0], columns=['path'])

        material_path = merge_material_path(
            depth=0,
            length=2,
            material=material,
            path=path,
        )

        pd.testing.assert_frame_equal(
            material_path,
            DataFrame(
                [[0.0, 0.0, 0.0, 1.0, 0.25], [1.0, 1.0, 0.0, 1.0, 0.1], [2.0, 2.0, 0.0, 1.0, 0.75]],
                columns=['timestamp', 'x', 'z', 'volume', 'p_1']
            ),
            check_like=True
        )