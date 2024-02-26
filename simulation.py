"""simulation.py: Easily modify simulation/molecular parameters and run DMC."""

import numpy as np
import walker
from dmc import dmc

###  Simulation Parameters:  ###
timeStep = 10.0
nTimeSteps = 100
nWalkers = 100

###  Specification of Potential Energy Function:  ###
    
###  Potential Energy Constants:  ###
equilibriumPosition = 1
k = 1      

def potentialEnergyFunction(w):
    x = np.sqrt(np.sum((w.particles['c'].pos - w.particles['o'].pos)**2))
    return (1/2)*k*(x - equilibriumPosition)**2

def diffuseFunction(w):
    for particle in w.particles.values():
        particle.pos += np.random.normal(0.0, np.sqrt(timeStep/particle.mass), (1, 3))

###  Specification of Walkers:  ###  
particles = {'c': 1, 'o': 1}
walkerInitFunc = walker.walker(particles)

d = dmc()

d.setParams(timeStep, nTimeSteps, nWalkers, walkerInitFunc, potentialEnergyFunction, diffuseFunction)
d.run()
d.visualizeRefVals()