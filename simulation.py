"""simulation.py: Easily modify simulation/molecular parameters and run DMC."""

import numpy as np

import walkers as wlkrs
from dmc import dmc

###  Simulation Parameters:  ###
timeStep = 10.0 # atu
nTimeStepsEq = 10000 # equilibration steps. number of steps over which to "relax" the system.
nTimeStepsProd = 100000 # production steps. number of steps over which to compute an average reference value after equilibration.
nWalkers = 1000

###  Specification of Potential Energy Function:  ###
    
###  Potential Energy Constants:  ###
equilibriumPosition = 0.59707
k = 1.2216

###  Particle Indices:            ###
c = 0
o = 1

'''
walker1
          particles
        p * * * ...
        o * * * ...
        s * * * ...

walker2
          * * * ...
          * * * ...
          * * * ...

...

(nwalkers, 3, nparticles) ndarray
'''
def potentialEnergyFunction(walkers):
    arr = walkers.to_arr()
    dist = np.sqrt(np.sum(np.power(arr[:, :, c] - arr[:, :, o], 2), axis=1))
    return (1/2)*k*(dist - equilibriumPosition)**2

def diffuseFunction(walkers):
    new_arr = walkers.to_arr()

    new_arr[:, :, c] += np.random.normal(0.0, np.sqrt(1/walkers.particles2atomic['c']), new_arr[:, :, c].shape)
    new_arr[:, :, o] += np.random.normal(0.0, np.sqrt(1/walkers.particles2atomic['o']), new_arr[:, :, o].shape)

    walkers.set_arr(new_arr)

def genFunction(count):
    pos = 5 * np.random.rand(count, 3, 2)
    return pos

###  Specification of Walkers:  ###  
particles2mass = {'c': 12.0000, 'o': 15.995} # g/mole

walkers = wlkrs.Walkers()

for ptcl in particles2mass:
    walkers.register_particle(ptcl, particles2mass[ptcl])

sim = dmc()
sim.setParams(timeStep, nTimeStepsEq, nTimeStepsProd, nWalkers, potentialEnergyFunction, diffuseFunction, genFunction, walkers)
sim.run()
sim.visualize_ref_vals()
sim.visualize_walker_population()

print(f'simulation should converge to {(1/2)*1*np.sqrt((k)/walkers.reduced_mass())} and converges approx. to {sim.mean_ref_val()}')