"""dmc.py: Encapsulates a DMC simulation in a class."""

import numpy as np
import matplotlib.pyplot as plt

class dmc:
    """DMC Simulation."""
    def __init__(self):
        # simulation parameters
        self.timeStep = None
        self.nTimeSteps = None
        self.nWalkers = None

        self.walkerInitFunc = None
        self.valFunc = None
        self.diffuseFunc = None

        self.walkers = None
        self.walkerPop = None

        self.refVal = None
        self.refVals = None

    def setParams(self, timeStep, nTimeSteps, nWalkers, walkerInitFunc, valFunc, diffuseFunc):
        self.timeStep = timeStep
        self.nTimeSteps = nTimeSteps
        self.nWalkers = nWalkers

        self.walkerInitFunc = walkerInitFunc
        self.valFunc = valFunc
        self.diffuseFunc = diffuseFunc

    def init(self):
        self.walkers = []
        for i in range(self.nWalkers):
            self.walkers.append(self.walkerInitFunc())

        self.walkerPop = np.zeros(self.nTimeSteps+1)
        self.walkerPop[0] = self.nWalkers

        self.refVals = np.zeros(self.nTimeSteps)

    def penalty(self):
        return (1.0-(len(self.walkers)/self.nWalkers))/(2.0*self.timeStep)

    def getRefVal(self):
        return np.mean([self.valFunc(walker) for walker in self.walkers]) + self.penalty()
    
    def diffuse(self):
        for walker in self.walkers:
            self.diffuseFunc(walker) 

    def adjustWalkerPopulation(self):
        survivingWalkers = []
        
        for walker in self.walkers:
            val = self.valFunc(walker)

            print(val)
            print(self.refVal)

            if val > self.refVal:
                probabilityDelete = np.exp(-(val-self.refVal)*self.timeStep)
                if probabilityDelete >= np.random.uniform():
                    survivingWalkers.append(walker)
            elif val < self.refVal:
                survivingWalkers.append(walker)
                probabilityReplicate = np.exp(-(val-self.refVal)*self.timeStep) - 1.0
                if probabilityReplicate > np.random.uniform():
                    newWalker = self.walkerInitFunc()
                    newWalker.particles = dict(walker.particles)
                    survivingWalkers.append(newWalker)
            else:
                survivingWalkers.append(walker)

        self.walkers = survivingWalkers
                
    def run(self):
        self.init()

        for i in range(self.nTimeSteps):
            self.refVal = self.getRefVal()
            self.diffuse()
            self.adjustWalkerPopulation()

            self.walkerPop[i+1] = len(self.walkers)
            self.refVals[i] = self.refVal

    def visualizeRefVals(self):
        fig, axes = plt.subplots(1, 1)
        axes.plot(np.arange(0, len(self.refVals)), self.refVals)
        fig.show()