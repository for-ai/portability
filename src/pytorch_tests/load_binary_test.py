# -*- coding: utf-8 -*-
# Owner(s): ["oncall: package/deploy"]

from io import BytesIO
from sys import version_info
from textwrap import dedent
from unittest import skipIf

from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


@skipIf(version_info < (3, 7), "ResourceReader API introduced in Python 3.7")
class TestResources(PackageTestCase):
    def test_importer_access(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.save_text("main", "main", "my string")
            he.save_binary("main", "main_binary", "my string".encode("utf-8"))
            src = dedent(
                """\
                import importlib
                import torch_package_importer as resources

                t = resources.load_text('main', 'main')
                b = resources.load_binary('main', 'main_binary')
                """
            )
            he.save_source_string("main", src, is_package=True)
        buffer.seek(0)
        hi = PackageImporter(buffer)
        m = hi.import_module("main")
        self.assertEqual(m.t, "my string")
        self.assertEqual(m.b, "my string".encode("utf-8"))

if __name__ == "__main__":
    run_tests()
