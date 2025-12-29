#!/bin/bash

# Test runs for advect1d
# Configurations: wavespeed = +1/-1, num_patches = 1/8

cd "$(dirname "$0")"

# Run 1: a=+1, np=1
./advect1d << 'EOF'
reset
set physics wavespeed=1.0
set initial num_zones=1000
set initial num_patches=1
init
t -> 0.5
select products concentration cell_x
write products prods.a+1.np1.bin
quit
EOF

# Run 2: a=+1, np=8
./advect1d << 'EOF'
reset
set physics wavespeed=1.0
set initial num_zones=1000
set initial num_patches=8
init
t -> 0.5
select products concentration cell_x
write products prods.a+1.np8.bin
quit
EOF

# Run 3: a=-1, np=1
./advect1d << 'EOF'
reset
set physics wavespeed=-1.0
set initial num_zones=1000
set initial num_patches=1
init
t -> 0.5
select products concentration cell_x
write products prods.a-1.np1.bin
quit
EOF

# Run 4: a=-1, np=8
./advect1d << 'EOF'
reset
set physics wavespeed=-1.0
set initial num_zones=1000
set initial num_patches=8
init
t -> 0.5
select products concentration cell_x
write products prods.a-1.np8.bin
quit
EOF

echo ""
echo "=== Creating plots ==="

source ../../.venv/bin/activate

python3 -m mist.plot_products prods.a+1.np1.bin -o plot.a+1.np1.pdf -t "a=+1, np=1"
python3 -m mist.plot_products prods.a+1.np8.bin -o plot.a+1.np8.pdf -t "a=+1, np=8"
python3 -m mist.plot_products prods.a-1.np1.bin -o plot.a-1.np1.pdf -t "a=-1, np=1"
python3 -m mist.plot_products prods.a-1.np8.bin -o plot.a-1.np8.pdf -t "a=-1, np=8"

echo ""
echo "=== Generated files ==="
ls -la prods.*.bin plot.*.pdf
