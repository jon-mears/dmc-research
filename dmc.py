"""dmc.py: Encapsulates a DMC simulation in a class."""

import numpy as np
import matplotlib.pyplot as plt

import walkers as wlkrs

class dmc:
    """DMC Simulation."""
    def __init__(self):
        # simulation parameters
        self.time_step = None
        self.nStepsEq = None
        self.nStepsProd = None
        self.nWalkers = None

        self.val_func = None
        self.diffuse_func = None
        self.gen_func = None

        self.walkers = None
        self.walker_counts = None
        self.cur_walker_count = None

        self.ref_val = None
        self.ref_vals = None

    def setParams(self, time_step, nStepsEq, nStepsProd, nWalkers, val_func, diffuse_func, gen_func, walkers):
        self.time_step = time_step
        self.nStepsEq = nStepsEq
        self.nStepsProd = nStepsProd
        self.nWalkers = nWalkers

        self.val_func = val_func
        self.diffuse_func = diffuse_func
        self.gen_func = gen_func

        self.walkers = walkers

    def init(self):
        self.walkers.make(self.nWalkers, self.gen_func)

        self.walker_counts = np.zeros(self.nStepsProd+1)
        self.ref_vals = np.zeros(self.nStepsProd)

        self.cur_walker_count = self.walkers.nWalkers

    def penalty(self):
        return (1.0-(self.cur_walker_count/self.nWalkers))/(2.0*self.time_step)

    def compute_ref_val(self):
        r = np.mean(self.val_func(self.walkers)) + self.penalty()
        # print(f'refVal is {r}\n')
        return r
    
    def diffuse(self):
        self.diffuse_func(self.walkers)

        # print(f'after diffusion:\n{self.walkers.to_arr()}\n')

    def adjust_walker_population(self):        
        idx = np.array(self.walkers.walkers_idx)

        vals = self.val_func(self.walkers)

        # print(f'potential energies are:\n{vals}\n')

        greater_vals = vals[vals > self.ref_val]
        greater_idx = idx[vals > self.ref_val]  

        prob_delete = np.exp(-(greater_vals-self.ref_val)*self.time_step)

        delete_rand = np.random.uniform(0.0, 1.0, (greater_vals.size))
        delete_idx = greater_idx[prob_delete < delete_rand]

        lesser_vals = vals[vals < self.ref_val]
        lesser_idx = idx[vals < self.ref_val]

        prob_replicate = np.exp(-(lesser_vals-self.ref_val)*self.time_step) - 1.0

        replicate_rand = np.random.uniform(0.0, 1.0, (lesser_vals.size))
        replicate_idx = lesser_idx[prob_replicate > replicate_rand]

        self.walkers.delete(delete_idx.tolist())
        self.walkers.replicate(replicate_idx.tolist())

        # print(f'after population adjustment:\n{self.walkers.to_arr()}\n')

    def equilibration(self):
        for i in range(self.nStepsEq):
            self.ref_val = self.compute_ref_val()
            self.diffuse()
            self.adjust_walker_population()

            self.cur_walker_count = self.walkers.nWalkers

    def production(self):
        self.walker_counts[0] = self.walkers.nWalkers

        for i in range(self.nStepsProd):
            self.ref_val = self.compute_ref_val()
            self.diffuse()
            self.adjust_walker_population()

            self.walker_counts[i+1] = self.walkers.nWalkers
            self.ref_vals[i] = self.ref_val

            self.cur_walker_count = self.walkers.nWalkers

            # print(f'walker count is {self.cur_walker_count}\n')
                
    def run(self):
        self.init()
        self.equilibration()
        self.production()

    def visualize_ref_vals(self):
        plt.plot(np.arange(self.ref_vals.size), self.ref_vals)
        plt.plot(np.arange(self.ref_vals.size), self.mean_ref_val()*np.ones((self.ref_vals.size)))
        plt.title('Reference Value vs. Time Step')
        plt.xlabel('Time Step')
        plt.ylabel('Reference Value')
        plt.show()

    def visualize_walker_population(self):
        plt.plot(np.arange(self.walker_counts.size), self.walker_counts)
        plt.plot(np.arange(self.walker_counts.size), self.mean_walker_population()*np.ones((self.walker_counts.size)))
        plt.title('Walker Count vs. Time Step')
        plt.xlabel('Time Step')
        plt.ylabel('Walker Count')
        plt.show()

    def mean_ref_val(self):
        return np.mean(self.ref_vals)
    
    def mean_walker_population(self):
        return np.mean(self.walker_counts)