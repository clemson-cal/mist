"""Example usage of the mist Python interface."""

from mist import Mist

# Basic usage with auto-init
with Mist("./advect1d") as sim:
    sim.select_products(["concentration"])

    # Advance by time increment
    sim.advance_by(0.1)
    print(f"time = {sim.time}, iteration = {sim.iteration}")

    # Advance to specific time
    sim.advance_to(0.5)
    print(f"time = {sim.time}, iteration = {sim.iteration}")

    # Direct array access (auto-concatenates patches)
    u = sim.products["concentration"]
    print(f"u.shape = {u.shape}, u.min() = {u.min():.4f}, u.max() = {u.max():.4f}")

    # Read configs
    print(f"physics: {sim.physics.to_dict()}")
    print(f"initial: {sim.initial.to_dict()}")

print()

# Configure before init
with Mist("./advect1d", physics={"cfl": 0.8}, initial={"num_zones": 100}) as sim:
    print(f"Configured: cfl={sim.physics['cfl']}, zones={sim.initial['num_zones']}")
    sim.advance_to(1.0)
    print(f"time = {sim.time}, iteration = {sim.iteration}, dt = {sim.dt:.6f}")

print()

# Manual init with config modification
with Mist("./advect1d", init=False) as sim:
    # Modify physics (allowed anytime)
    sim.physics["cfl"] = 0.5

    # Modify initial (only before init)
    sim.initial["num_zones"] = 50

    sim.init()
    print(f"After manual init: zones={sim.initial['num_zones']}, cfl={sim.physics['cfl']}")

    # Physics can still be modified after init
    sim.physics["cfl"] = 0.3
    print(f"Changed cfl to {sim.physics['cfl']}")

    # Initial cannot be modified after init
    try:
        sim.initial["num_zones"] = 100
    except RuntimeError as e:
        print(f"Expected error: {e}")
