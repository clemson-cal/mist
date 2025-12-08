#!/bin/bash
# Test interactive mode by sending commands via expect or using echo with stop
echo -e "n++\nstatus\nstop" | ./advection-1d
