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
echo "Running Sod problem at 100000 zones..."
./srhd1d <<'EOF'
set initial num_zones=100000
set physics ic=sod
init
t += 0.2
select products
write products prods_100000.dat
stop
EOF

# Low resolution run
echo "Running Sod problem at 256 zones..."
./srhd1d <<'EOF'
set initial num_zones=256
set physics ic=sod
init
t += 0.2
select products
write products prods_256.dat
stop
EOF

# Generate plots
echo "Generating plots..."
python -m mist.plot_products prods_100000.dat -o sod_100000.pdf -t "Sod Problem (100000 zones, t=0.2)"
python -m mist.plot_products prods_256.dat -o sod_256.pdf -t "Sod Problem (256 zones, t=0.2)"

echo "Done. Output files:"
ls -la sod_*.pdf
