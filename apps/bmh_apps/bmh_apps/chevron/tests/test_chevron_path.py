#!/usr/bin/env python
import unittest

from pandas import DataFrame
from pandas.testing import assert_frame_equal

from bmh_apps.chevron.chevron_path import chevron_path


class TestChevronPath(unittest.TestCase):
    def test_simple(self):
        assert_frame_equal(chevron_path(1), DataFrame(data=[[0.0, 0.0], [1.0, 1.0]], columns=["part", "path"]))
        assert_frame_equal(chevron_path(2), DataFrame(data=[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], columns=["part", "path"]))
        assert_frame_equal(chevron_path(3), DataFrame(data=[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]], columns=["part", "path"]))

    def test_partial(self):
        assert_frame_equal(chevron_path(0.5), DataFrame(data=[[0.0, 0.0], [0.5, 0.5]], columns=["part", "path"]))
        assert_frame_equal(chevron_path(1.25), DataFrame(data=[[0.0, 0.0], [1.0, 1.0], [1.25, 0.75]], columns=["part", "path"]))
        assert_frame_equal(chevron_path(1.5), DataFrame(data=[[0.0, 0.0], [1.0, 1.0], [1.5, 0.5]], columns=["part", "path"]))
        assert_frame_equal(chevron_path(2.25), DataFrame(data=[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.25, 0.25]], columns=["part", "path"]))
        assert_frame_equal(chevron_path(2.75), DataFrame(data=[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.75, 0.75]], columns=["part", "path"]))
        assert_frame_equal(chevron_path(3.25), DataFrame(data=[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [3.25, 0.75]], columns=["part", "path"]))
        assert_frame_equal(chevron_path(3.75), DataFrame(data=[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [3.75, 0.25]], columns=["part", "path"]))
