#!/usr/bin/env python
import unittest

from bmh_apps.helpers.chevron_path import chevron_path


class TestChevronPath(unittest.TestCase):
    def test_simple(self):
        self.assertListEqual(list(chevron_path(1)['path'].values), [0, 1])
        self.assertListEqual(list(chevron_path(2)['path'].values), [0, 1, 0])
        self.assertListEqual(list(chevron_path(3)['path'].values), [0, 1, 0, 1])

    def test_partial(self):
        self.assertListEqual(list(chevron_path(0.5)['path'].values), [0, 0.5])
        self.assertListEqual(list(chevron_path(1.25)['path'].values), [0, 1, 0.75])
        self.assertListEqual(list(chevron_path(1.5)['path'].values), [0, 1, 0.5])
        self.assertListEqual(list(chevron_path(2.25)['path'].values), [0, 1, 0, 0.25])
        self.assertListEqual(list(chevron_path(2.75)['path'].values), [0, 1, 0, 0.75])
        self.assertListEqual(list(chevron_path(3.25)['path'].values), [0, 1, 0, 1, 0.75])
        self.assertListEqual(list(chevron_path(3.75)['path'].values), [0, 1, 0, 1, 0.25])
