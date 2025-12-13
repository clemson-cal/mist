#!/bin/bash

# Scaling test for advect1d

cd "$(dirname "$0")"

OUTFILE="scaling_results.csv"

echo "num_threads,num_partitions,num_zones,use_flux_buffer,Mzps" > "$OUTFILE"

run_test() {
    local nt=$1
    local np=$2
    local nz=$3
    local ufb=$4  # use_flux_buffer: 0 or 1

    # Run simulation and extract Mzps from the last progress message
    local mzps=$(./advect1d 2>&1 << EOF | grep -E '^\[' | tail -1 | sed -E 's/.*Mzps=([0-9.eE+-]+).*/\1/'
reset
set exec num_threads=$nt
set initial num_partitions=$np
set initial num_zones=$nz
set physics use_flux_buffer=$ufb
init
n += 100
quit
EOF
)
    echo "$nt,$np,$nz,$ufb,$mzps"
    echo "$nt,$np,$nz,$ufb,$mzps" >> "$OUTFILE"
}

echo "=== Scaling Test ==="
echo ""

# Sweep 1: Thread scaling at nz=1e8, both modes
echo "--- Thread scaling (nz=1e8), both modes ---"
for n in 1 2 4 8 16; do
    run_test $n $n 100000000 0  # fused
    run_test $n $n 100000000 1  # unfused
done

echo ""

# Sweep 2: Problem size scaling (sequential), both modes
echo "--- Problem size scaling (sequential), both modes ---"
for nz in 1000 10000 100000 1000000 10000000 100000000; do
    run_test 0 1 $nz 0  # fused
    run_test 0 1 $nz 1  # unfused
done

echo ""
echo "=== Results saved to $OUTFILE ==="
cat "$OUTFILE"

echo ""
echo "=== Generating plots ==="
source ../../.venv/bin/activate
python3 "$(dirname "$0")/plot_scaling.py" "$OUTFILE"

echo ""
echo "=== Generated files ==="
ls -la scaling_*.csv scaling_*.pdf 2>/dev/null
