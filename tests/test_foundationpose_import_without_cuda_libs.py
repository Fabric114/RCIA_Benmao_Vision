import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImportWithoutCudaLibsTest(unittest.TestCase):
    def test_utils_imports_without_pytorch3d_and_nvdiffrast(self):
        code = ("import sys; sys.modules['pytorch3d']=None; sys.modules['nvdiffrast']=None; "
                "sys.path.insert(0,'foundation_pose'); import Utils; assert Utils.dr is None; print('ok')")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ok", r.stdout)


if __name__ == "__main__":
    unittest.main()
