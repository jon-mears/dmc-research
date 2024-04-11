import numpy as np
import pytest
from DMC_am_031324 import Walkers, dmc, potentialEnergyFunction, diffusionFunction, posFunction

def test_walkers_initialization():
    particles2mass = {'c': 12.0000, 'o': 15.995}
    walkers = Walkers(particles2mass)
    assert len(walkers.particles) == 2, "Walkers should have 2 particle types registered"
    assert walkers.cap == 10000, "Initial capacity should be set correctly"

def test_walkers_realloc():
    walkers = Walkers({'c': 12.0000}, init_cap=5)
    walkers.realloc(10)
    assert walkers.cap == 10, "Capacity should be updated after realloc"

def test_walkers_management():
    walkers = Walkers({'c': 12.0000, 'o': 15.995}, init_cap=100)
    assert walkers.nWalkers == 0, "Initial number of walkers should be zero."

    walkers.make(10, posFunction)
    assert walkers.nWalkers == 10, "Walkers should be added correctly."

    walkers.realloc(20)
    assert walkers.cap >= 20, "Walkers array should be reallocated correctly."

    walkers.delete([0, 1])
    assert walkers.nWalkers == 8, "Walkers should be deleted correctly."

def test_non_empty_potential_energy_array():
    walkers = Walkers({'c': 12.0000, 'o': 15.995})
    walkers.make(5, posFunction)
    energies = potentialEnergyFunction(walkers)
    assert energies.size > 0, "Potential energy array should not be empty."

def test_dmc_setParams():
    sim = dmc()
    sim.setParams(0.1, 100, 200, 1000, potentialEnergyFunction, diffusionFunction, posFunction, Walkers({'c': 12.0000}))
    assert sim.time_step == 0.1, "Time step should be set correctly"
    assert sim.nWalkers == 1000, "Number of walkers should be set correctly"

def test_dmc_equilibration():
    sim = dmc()
    sim.setParams(0.1, 100, 200, 1000, potentialEnergyFunction, diffusionFunction, posFunction, Walkers({'c': 12.0000}))
    sim.equilibration()
    assert sim.walkers.nWalkers <= 1000, "Walker count should be adjusted after equilibration"

def test_compute_ref_val_initialization():
    walkers = Walkers({'c': 12.0000, 'o': 15.995})
    walkers.make(10, posFunction)

    sim = dmc()
    sim.setParams(0.1, 100, 200, 10, potentialEnergyFunction, diffusionFunction, posFunction, walkers)
    sim.cur_walker_count = 10
    sim.nWalkers = 10

    ref_val = sim.compute_ref_val()
    assert ref_val is not None, "Reference value should be properly computed."
    assert isinstance(ref_val, float), "Reference value should be a float."

def test_full_simulation_flow():
    walkers = Walkers({'c': 12.0000, 'o': 15.995})
    sim = dmc()
    sim.setParams(0.1, 100, 200, 1000, potentialEnergyFunction, diffusionFunction, posFunction, walkers)
    sim.run()
    assert sim.walkers.nWalkers > 0, "There should be walkers remaining after the simulation"
    assert len(sim.walker_counts) == 201, "Walker counts should be tracked for each production step"

def test_deleting_excessive_walkers():
    walkers = Walkers({'c': 12.0000}, init_cap=10)
    walkers.make(5, posFunction)
    try:
        walkers.delete(range(10))
        assert False, "Deleting more walkers than exist should raise an error"
    except ValueError:
        assert True

def test_equilibration_with_proper_setup():
    walkers = Walkers({'c': 12.0000, 'o': 15.995})
    walkers.make(100, posFunction)

    sim = dmc()
    sim.setParams(0.1, 100, 200, 100, potentialEnergyFunction, diffusionFunction, posFunction, walkers)
    sim.equilibration()

    assert sim.walkers.nWalkers <= 100 and sim.walkers.nWalkers > 0, \
        "Walkers should be adjusted but not all deleted during equilibration"

