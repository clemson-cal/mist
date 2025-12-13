#!/bin/bash
#
# Test srhd1d physics at different resolutions

set -e
cd "$(dirname "$0")"

# Activate virtual environment if present
if [ -f ../../.venv/bin/activate ]; then
    source ../../.venv/bin/activate
fi

# High resolution run
echo "Running Sod problem at 8192 zones..."
./srhd1d <<'EOF'
set initial num_zones=8192
set physics ic=sod rk_order=3
init
repeat 0.01 t show message
t += 0.4
select products
write products prods_hi.dat
stop
EOF

# Low resolution run
echo "Running Sod problem at 256 zones..."
./srhd1d <<'EOF'
set initial num_zones=256
set physics ic=sod rk_order=3
init
repeat 0.01 t show message
t += 0.4
select products
write products prods_lo.dat
stop
EOF

# Generate comparison plot
echo "Generating plot..."
python -m mist.plot_products \
    "hi (N=8192):prods_hi.dat" \
    "lo (N=256):prods_lo.dat" \
    -f density velocity pressure \
    -o sod.pdf \
    -t "Sod Problem (t=0.2)"

echo "Done: sod.pdf"
