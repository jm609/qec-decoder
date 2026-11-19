import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from qec_detector_rtl.scripts.prepare_server_bundle import prepare_bundle


ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "artifacts/qec_detector_rtl/d3_r3_z_v1_20261118_001/vectors"
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")


def shell_path(path):
    value = Path(path).resolve().as_posix()
    return "/" + value[0].lower() + value[2:] if os.name == "nt" else value


@unittest.skipUnless(BASH and Path(BASH).is_file() and VECTORS.is_dir(), "Existing shell or frozen vectors unavailable")
class ServerShellTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        prepare_bundle(VECTORS, self.bundle)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.runtime = self.root / "runtime.sh"
        self.runtime.write_text("""#!/bin/sh
expected= actual= shots= wave= force=0
for arg in "$@"; do
    case "$arg" in
        +expected=*) expected=${arg#*=} ;;
        +actual=*) actual=${arg#*=} ;;
        +shots=*) shots=${arg#*=} ;;
        +vcd=*|+fsdb=*) wave=${arg#*=} ;;
        +force_failure) force=1 ;;
    esac
done
if [ "$force" -eq 1 ] && [ "${MISS_NEGATIVE:-0}" -eq 0 ]; then echo MISMATCH; exit 1; fi
cp "$expected" "$actual"
if [ "${CORRUPT_OUTPUT:-0}" -eq 1 ]; then printf '0\\n' > "$actual"; fi
if [ -n "$wave" ]; then printf 'stub waveform\\n' > "$wave"; fi
printf 'PASS shots=%s inputs=%s detectors=%s\\n' "$shots" "$((shots*33))" "$((shots*24))"
""", encoding="ascii", newline="\n")
        compiler = self.tools / "vcs"
        compiler.write_text("""#!/bin/sh
[ "${FAIL_COMPILE:-0}" -eq 0 ] || exit 7
cp "$FAKE_RUNTIME" simv
chmod +x simv
""", encoding="ascii", newline="\n")
        compiler.chmod(0o755)
        self.environment = dict(os.environ, FAKE_RUNTIME=shell_path(self.runtime))

    def run_shell(self, run_id, *options, **environment):
        return subprocess.run([
            BASH, "-c", 'PATH="$1:$PATH"; export PATH; shift; exec sh "$@"', "shell-test",
            shell_path(self.tools), shell_path(self.bundle / "scripts/run_vcs.sh"), run_id, *options,
        ], env={**self.environment, **environment}, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30)

    def test_complete_shell_flow_and_existing_run_protection(self):
        for name, options in [("basic", ()), ("fsdb", ("--fsdb",))]:
            result = self.run_shell(name, *options)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((self.bundle / "runs" / name / "PASS.txt").is_file())
        self.assertNotEqual(self.run_shell("basic").returncode, 0)

    def test_compile_failure_cannot_pass(self):
        result = self.run_shell("compile_fail", FAIL_COMPILE="1")
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertTrue((self.bundle / "runs/compile_fail/compile.log").is_file())
        self.assertFalse((self.bundle / "runs/compile_fail/PASS.txt").exists())

    def test_success_message_does_not_hide_wrong_output(self):
        result = self.run_shell("wrong_output", CORRUPT_OUTPUT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.bundle / "runs/wrong_output/noiseless.actual.01").read_text(), "0\n")
        self.assertFalse((self.bundle / "runs/wrong_output/PASS.txt").exists())

    def test_negative_check_must_reject_wrong_expected_bit(self):
        result = self.run_shell("miss_negative", MISS_NEGATIVE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("The bench accepted", result.stderr)
        self.assertFalse((self.bundle / "runs/miss_negative/PASS.txt").exists())


if __name__ == "__main__":
    unittest.main()
