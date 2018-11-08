#!/usr/bin/env python
import unittest

from ..identifiers import get_identifier, get_identifiers


class TestHelpers(unittest.TestCase):
    def test_get_identifier(self):
        self.assertEqual('foobar', get_identifier('foobar'))

    def test_last_empty(self):
        self.assertEqual('foo', get_identifier('./foo/'))

    def test_path(self):
        self.assertEqual('bar', get_identifier('foo/bar'))

    def test_empty(self):
        with self.assertRaises(IndexError):
            get_identifier('')

    def test_get_identifiers(self):
        self.assertEqual(['foobar', 'foo', 'bar'], get_identifiers(['foobar', './foo/', 'foo/bar']))
