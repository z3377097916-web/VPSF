import json
import tempfile
import unittest
from pathlib import Path
from scripts.verify_package import inventory,verify


class PackageTests(unittest.TestCase):
    def test_manifest_excludes_itself_and_checks_unicode_files(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'说明.md').write_text('test')
            (p/'MANIFEST_SHA256.json').write_text(json.dumps(inventory(p)))
            self.assertEqual(verify(p),dict(checked_files=1,missing=[],unlisted=[],mismatched=[]))

    def test_changed_missing_and_unlisted_files_detected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a').write_text('a');(p/'b').write_text('b')
            (p/'MANIFEST_SHA256.json').write_text(json.dumps(inventory(p)))
            (p/'a').write_text('changed');(p/'b').unlink();(p/'c').write_text('new')
            result=verify(p)
            self.assertEqual(result['mismatched'],['a']);self.assertEqual(result['missing'],['b'])
            self.assertEqual(result['unlisted'],['c'])

    def test_python_caches_do_not_break_delivery_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a.py').write_text('x=1')
            (p/'MANIFEST_SHA256.json').write_text(json.dumps(inventory(p)))
            (p/'__pycache__').mkdir();(p/'__pycache__/a.pyc').write_bytes(b'cache')
            self.assertEqual(verify(p)['unlisted'],[])
