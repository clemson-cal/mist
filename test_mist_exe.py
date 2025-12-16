#!/usr/bin/env python3
"""Test the mist_exe module."""

import mist

print("=== Testing Mist class ===")
with mist.mist_exe.Mist("./examples/advect1d/advect1d") as sim:
    # Initialize
    sim.init()
    print(f"Initialized: time = {sim.time}")

    # Configure products
    sim.select_products(["concentration"])

    # Get initial products
    products = sim.products
    print(f"Initial products:")
    for key, value in products.items():
        print(f"  {key}: {value}")

    # Advance the simulation
    sim.run(dt=0.1)
    print(f"After dt=0.1: time = {sim.time}")

    # Run to a specific time
    sim.run(t=0.5)
    print(f"After t=0.5: time = {sim.time}")

    # Get products again
    products = sim.products
    print(f"Final products:")
    for key, value in products.items():
        print(f"  {key}: {value}")

    # Show full state info
    print(f"State: {sim.state}")

print("\nTest completed successfully!")
