#!/usr/bin/env python
import unittest

from pandas import DataFrame

from bmh_apps.helpers.stacker import Stacker


class TestStacker(unittest.TestCase):
    def test_one(self):
        res = []

        def aggregate(t, x, z, v, ps):
            res.append([t, x, z, v] + list(ps.values))

        stacker = Stacker(depth=0, length=2)
        material = DataFrame(data=[[0, 1, 0.25]], columns=['timestamp', 'volume', 'p_1'])
        path = DataFrame(data=[0, 1], columns=['path'])
        stacker.run(material=material, path=path, callback=aggregate)

        self.assertEqual(len(res), 1)
        self.assertListEqual(res, [[0, 2, 0, 1, 0.25]])

    def test_two(self):
        res = []

        def aggregate(t, x, z, v, ps):
            res.append([t, x, z, v] + list(ps.values))

        stacker = Stacker(depth=0, length=2)
        material = DataFrame(data=[[0, 1, 0.25], [1, 1, 0.75]], columns=['timestamp', 'volume', 'p_1'])
        path = DataFrame(data=[0, 1], columns=['path'])
        stacker.run(material=material, path=path, callback=aggregate)

        self.assertEqual(len(res), 2)
        self.assertListEqual(res, [[0, 0, 0, 1, 0.25], [1, 2, 0, 1, 0.75]])

    def test_interpolate(self):
        res = []

        def aggregate(t, x, z, v, ps):
            res.append([t, x, z, v] + list(ps.values))

        stacker = Stacker(depth=0, length=2)
        material = DataFrame(data=[[0, 1, 0.25], [1, 1, 0.1], [2, 1, 0.75]], columns=['timestamp', 'volume', 'p_1'])
        path = DataFrame(data=[0, 1], columns=['path'])
        stacker.run(material=material, path=path, callback=aggregate)

        self.assertEqual(len(res), 3)
        self.assertListEqual(res, [[0, 0, 0, 1, 0.25], [1, 1, 0, 1, 0.1], [2, 2, 0, 1, 0.75]])
