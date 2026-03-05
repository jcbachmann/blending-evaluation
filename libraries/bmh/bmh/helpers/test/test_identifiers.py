#!/usr/bin/env python
import unittest

import pytest

from ..identifiers import get_identifier, get_identifiers


class TestHelpers(unittest.TestCase):
    def test_get_identifier(self):
        assert get_identifier("foobar") == "foobar"

    def test_last_empty(self):
        assert get_identifier("./foo/") == "foo"

    def test_path(self):
        assert get_identifier("foo/bar") == "bar"

    def test_empty(self):
        with pytest.raises(IndexError):
            get_identifier("")

    def test_get_identifiers(self):
        assert get_identifiers(["foobar", "./foo/", "foo/bar"]) == ["foobar", "foo", "bar"]
