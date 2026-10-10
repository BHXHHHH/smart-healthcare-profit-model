"""Small numerical invariants; the CLI performs full-result regression checks."""

import unittest
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import reproduce_single as model


class NumericalTests(unittest.TestCase):
    def test_reference_manifest(self):
        count = 0
        for line in (
            (model.EXPECTED / "CHECKSUMS.txt").read_text(encoding="utf-8").splitlines()
        ):
            if ":" not in line:
                continue
            name, digest = (part.strip() for part in line.split(":", 1))
            if not name.endswith((".csv", ".json")):
                continue
            count += 1
            actual = hashlib.md5((model.EXPECTED / name).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, name)
        self.assertEqual(count, 14)

    def test_verification_rejects_missing_and_changed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            output = root / "output"
            reference.mkdir()
            output.mkdir()
            original = b"value\r\n1\r\n"
            digest = hashlib.md5(original).hexdigest()
            (reference / "CHECKSUMS.txt").write_text(
                f"result.csv: {digest}\n", encoding="utf-8"
            )
            with patch.object(model, "EXPECTED", reference), patch.object(
                model, "OUT", output
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(model.verify())
                (output / "result.csv").write_bytes(b"changed")
                self.assertFalse(model.verify())
                (output / "result.csv").write_bytes(original)
                self.assertTrue(model.verify())

    def test_grid_size_and_order(self):
        grid = model.grid_params(
            [0.2, 0.4, 0.6, 0.8],
            [0.02, 0.05, 0.08, 0.12],
            [0, 0.05, 0.10],
            np.linspace(0.1, 2, 8),
            np.linspace(0.3, 0.8, 6),
            np.linspace(0.7, 1, 5),
        )
        self.assertEqual(len(grid[0]), 10368)
        self.assertTrue(np.all(grid[5] > grid[4]))

    def test_reference_profit(self):
        baseline, dual, delta, valid = model.run_combos([dict(model.REF)])
        self.assertTrue(valid[0])
        self.assertAlmostEqual(baseline["Pi"][0], 0.02900119, places=7)
        self.assertAlmostEqual(dual["Pi"][0], 0.01862698, places=7)
        self.assertAlmostEqual(delta[0], -0.01037421, places=7)

    def test_supply_cap(self):
        rho, supply = model.doctor_equilibrium(np.array([0.0, 2.0]), 0.5, 0.05, 1.0)
        self.assertTrue(np.all((supply >= 0) & (supply <= 1)))
        self.assertAlmostEqual(supply[1], 1.0)
        self.assertAlmostEqual(rho[1], 2.0)


if __name__ == "__main__":
    unittest.main()
