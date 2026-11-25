#!/usr/bin/env python3
"""
Checks for QuTip/benchmark_qtrl.py (repo GRAPE vs QuTiP's qutip-qtrl GRAPE).

On a tiny system (n_t 3, n_c 4, N 20) the two codes must give the same
fidelity and the same gradient for the same pulse, in the repo's F = f^2
convention; otherwise the benchmark compares different problems. Also a
smoke test of both optimisation runners. Skipped when qutip-qtrl is missing
(pip install qutip-qtrl). ReportFromCsv needs no qutip: it rebuilds the README
and both figures from the committed CSVs. Run from the repo root:

    python3 validation/test_qutip_benchmark.py -v      (~8 s)
"""
import os
import re
import shutil
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "QuTip"))

try:
    import qutip_qtrl  # noqa: F401
    HAVE_QTRL = True
except ImportError:
    HAVE_QTRL = False

if HAVE_QTRL:
    import benchmark_qtrl as bq

N_C, N = 4, 20


@unittest.skipUnless(HAVE_QTRL, "qutip-qtrl not installed")
class SameProblem(unittest.TestCase):
    def _setup(self, problem):
        H0, Hc = bq.make_hamiltonian(bq.N_T, N_C)
        psi_i, psi_f = bq.targets(problem, N_C)
        u = bq.start_pulse(N, 3) * 5.0
        return H0, Hc, psi_i, psi_f, u

    def test_fidelity_and_gradient_agree(self):
        for problem in ("fock1", "X"):
            with self.subTest(problem=problem):
                H0, Hc, psi_i, psi_f, u = self._setup(problem)
                F_r, g_r = bq.repo_fid(u, H0, Hc, psi_i, psi_f)
                for pt in ("DIAG", "FRECHET"):
                    optim = bq.qtrl_optimizer(H0, Hc, psi_i, psi_f, N_C, N, prop_type=pt)
                    bq.qtrl_set(optim, u)
                    F_q, g_q = bq.qtrl_fid(optim, u)
                    self.assertLess(abs(F_r - F_q), 1e-12)
                    self.assertLess(np.max(np.abs(g_r - g_q)) / np.max(np.abs(g_r)), 1e-10)

    def test_float64_tau(self):
        H0, Hc, psi_i, psi_f, u = self._setup("fock1")
        optim = bq.qtrl_optimizer(H0, Hc, psi_i, psi_f, N_C, N)
        tau = np.asarray(optim.dynamics.tau)
        self.assertEqual(tau.dtype, np.float64)
        self.assertTrue(np.all(tau == bq.DT))

    def test_optimisers_run(self):
        H0, Hc, psi_i, psi_f, _ = self._setup("fock1")
        u0 = bq.start_pulse(N, 42)
        for runner in ("repo", "qutip"):
            with self.subTest(runner=runner):
                if runner == "repo":
                    r = bq.run_repo_opt(H0, Hc, psi_i, psi_f, N, u0, 5, 60)
                else:
                    r = bq.run_qtrl_opt(H0, Hc, psi_i, psi_f, N_C, N, u0, 5, 60)
                self.assertGreater(len(r["trace"]), 0)
                self.assertLessEqual(r["iterations"], 5)
                F_trace = max(F for _, F in r["trace"])
                self.assertAlmostEqual(r["F_final"], F_trace, delta=1e-9 + 0.05)
                self.assertGreaterEqual(r["F_final"], r["trace"][0][1] - 1e-12)


class ReportFromCsv(unittest.TestCase):
    """write_report rebuilds README.md + figures from the CSVs alone (no raw JSON, no qutip)."""

    def test_report_from_committed_csvs(self):
        import benchmark_qtrl as bqr
        src = os.path.join(REPO_ROOT, "QuTip", "benchmark_results")
        with tempfile.TemporaryDirectory() as tmp:
            for name in os.listdir(src):
                if name.endswith((".csv", ".json")):
                    shutil.copy(os.path.join(src, name), tmp)
            bqr.write_report(tmp)
            for name in ("README.md", "b3_timing.png", "b4_convergence.png"):
                self.assertTrue(os.path.exists(os.path.join(tmp, name)), name)
            md = open(os.path.join(tmp, "README.md")).read()
        for n in range(1, 10):
            self.assertIn(f"Table {n}.", md)
        for n in (1, 2):
            self.assertIn(f"**Figure {n}.**", md)
        # display maths: every $$ on its own line, in pairs; inline $ balanced per line
        self.assertEqual(md.count("$$") % 2, 0)
        for line in md.replace("$$", "").splitlines():
            self.assertEqual(line.count("$") % 2, 0, line)
        # GitHub applies markdown escapes before rendering maths, so a backslash followed by
        # ASCII punctuation (\{ \, \; \!) loses its backslash: use \lbrace, \thinspace, ...
        bad = re.findall(r"\\[!-/:-@\[-`{-~]", md)
        self.assertEqual(bad, [], "backslash + punctuation in README maths")
        # no literal | inside maths in a table row (it would split the cell)
        for line in md.splitlines():
            if line.startswith("|"):
                for m in re.findall(r"\$([^$]*)\$", line):
                    self.assertNotIn("|", m, line)


if __name__ == "__main__":
    unittest.main()
