#!/bin/sh
# No Python or package installation; use the existing course VCS environment.
set -eu

run_id=${1:-vcs_shell_001}
case "$run_id" in
    ''|*[!A-Za-z0-9_-]*) echo "Use a run ID containing only letters, digits, _ or -" >&2; exit 1 ;;
esac
fsdb=0
if [ "$#" -gt 1 ]; then
    [ "$2" = "--fsdb" ] || { echo "Second argument must be --fsdb" >&2; exit 1; }
    fsdb=1
fi
[ "$#" -le 2 ] || { echo "Too many arguments" >&2; exit 1; }

bundle_root=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
cd "$bundle_root"
for tool in vcs sha256sum cmp grep; do
    command -v "$tool" >/dev/null 2>&1 || { echo "Existing tool not found: $tool; do not install automatically" >&2; exit 1; }
done
mkdir -p runs
mkdir "runs/$run_id"
sha256sum -c CHECKSUMS.sha256 > "runs/$run_id/input_checksums.log" 2>&1
cd "runs/$run_id"
cp ../../rtl/measurement_to_detection.v .
cp ../../tb/tb_measurement_to_detection.sv .
set -- -full64 -sverilog -top tb_measurement_to_detection -o simv
if [ "$fsdb" -eq 1 ]; then
    set -- "$@" -debug_access+all +define+DUMP_FSDB
fi
vcs "$@" measurement_to_detection.v tb_measurement_to_detection.sv > compile.log 2>&1

for spec in noiseless:256 single_flip:34 random_bits:1024 si1000:4096; do
    name=${spec%:*}
    shots=${spec#*:}
    set -- "+measurements=../../vectors/$name.measurements.01" \
        "+expected=../../vectors/$name.expected_detectors.01" \
        "+actual=$name.actual.01" "+shots=$shots"
    if [ "$name" = "single_flip" ]; then
        if [ "$fsdb" -eq 1 ]; then
            wave=single_flip.fsdb
            set -- "$@" "+fsdb=$wave"
        else
            wave=single_flip.vcd
            set -- "$@" "+vcd=$wave"
        fi
    fi
    ./simv "$@" > "$name.log" 2>&1
    grep -q "PASS shots=$shots inputs=" "$name.log"
    cmp "$name.actual.01" "../../vectors/$name.expected_detectors.01"
done
[ -s "$wave" ] || { echo "Requested waveform missing or empty" >&2; exit 1; }

if ./simv +measurements=../../vectors/noiseless.measurements.01 \
    +expected=../../vectors/noiseless.expected_detectors.01 \
    +actual=expected_failure.actual.01 +shots=256 +force_failure > expected_failure.log 2>&1; then
    echo "The bench accepted an intentionally incorrect expected bit" >&2
    exit 1
fi
grep -q MISMATCH expected_failure.log
sha256sum ./*.actual.01 > actual_checksums.sha256
printf 'PASS: 5410 shots, 129840 detector bits, four exact file comparisons, negative check detected\n' > PASS.txt
printf 'PASS: results are in runs/%s\n' "$run_id"
