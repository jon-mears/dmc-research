'''
Godfred Awuku and Jonathan Mears

DMC_am_041424.py
CS446 -- DMC Research
Spring 2024
----------------------------------------------------------------------------------------------
Edit simulation parameters, potentialEnergyFunction, diffusionFunction, and posFunction below
and enter `python3 DMC_am_031324.py` into a terminal to run a DMC simulation with the specified 
parameters on the system described by the behavior of these three functions. (This has been 
preset to model the vibrating carbon monoxide bond (3) given in the project document, with the
recommended system parameters.
'''

import numpy as np
import matplotlib.pyplot as plt

class Walkers:
    def __init__(self, particles2mass, init_cap=10000):
        '''
        
        '''
        
        self.particles2mass = dict()
        self.particles = list()
        self.particles2atomic = dict() # amu

        for ptcl in particles2mass:
            self.register_particle(ptcl, particles2mass[ptcl])

        prod = 1
        sum = 0
        
        for ptcl in self.particles:
            prod *= self.particles2atomic[ptcl]
            sum += self.particles2atomic[ptcl]

        self.rmass = prod/sum
        
        self.walkers_idx = []
        self.nWalkers = 0
        self.cap = init_cap

        self.next_idx = 0

        self.arr = np.zeros((self.cap, 3, len(self.particles)))

    def register_particle(self, name, mass):
        self.particles.append(name)
        self.particles2mass[name] = mass

        self.particles2atomic[name] = mass/(6.02213670000e23*9.10938970000e-28)

    def to_arr(self):
        return self.arr[self.walkers_idx, :, :]
    
    def set_arr(self, new_arr, isUpdate=True):

        if isUpdate:
            self.arr[self.walkers_idx, :, :] = new_arr
        else:
            self.nWalkers = new_arr.shape[0]
            self.walkers_idx = range(self.nWalkers)
            self.next_idx = self.nWalkers

            if self.nWalkers > self.cap:
                self.realloc(self.nWalkers*2)

            self.arr[self.walkers_idx, :, :] = new_arr
    
    def realloc(self, new_cap=None):
        if new_cap is None:
            new_cap = self.nWalkers*2
        
        self.cap = new_cap
        
        new_arr = np.zeros((self.cap, 3, len(self.particles)))

        new_arr[:self.nWalkers, :, :] = self.arr[self.walkers_idx, :, :]
        self.arr = new_arr

        self.walkers_idx = list(range(self.nWalkers))
        self.next_idx = self.nWalkers

    def make(self, count, gen):
        '''
        count: int. number of walkers to create and add.
        gen: func(count) -> (count) ndarray. func that generates an initialized ndarray of a given shape.
        '''
        if self.next_idx + count > self.cap:
            self.realloc((self.nWalkers+count)*2)

        new_walkers = gen(count)

        self.arr[self.next_idx:self.next_idx+count] = new_walkers
        self.walkers_idx.extend(range(self.next_idx, self.next_idx+count))

        self.nWalkers += count
        self.next_idx += count

    def replicate(self, idx):
        to_copy = self.arr[idx, :, :]

        if self.next_idx + len(idx) > self.cap:
            self.realloc()

        self.arr[self.next_idx:self.next_idx + len(idx), :, :] = to_copy
        self.walkers_idx.extend(range(self.next_idx, self.next_idx + len(idx)))

        self.nWalkers += len(idx)
        self.next_idx += len(idx)
    
    def delete(self, idx):
        self.walkers_idx = list(filter(lambda i: i not in idx, self.walkers_idx))
        self.nWalkers -= len(idx)

    def mass(self, ptcl):
        return self.particles2mass[ptcl]
    
    def atomic_mass(self, ptcl):
        return self.particles2atomic[ptcl]

    def reduced_mass(self):
        return self.rmass
    
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

        self.wavefunc = None

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
        return r

    def diffuse(self):
        self.diffuse_func(self.walkers)

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
    
###  UTILITY FUNCTIONS  ###

def bond_vec(walkers, ptcl1, ptcl2):
    '''return the bond between `ptcl1` and `ptcl2` (global variables that were assigned indices into the third axis
    of the `walkers` ndarray) as a vector in R^3, origin at `ptcl1`.
    walkers: Walkers object. see the Walkers interface below.
    ptcl1: int. index into the third axis of the `arr` ndarray of `walkers`; that is, an index of a particle within
    the array. the tail of the vector will be placed at this particle's position.
    ptcl2: int. index into the third axis of the `arr` ndarray of `walkers`; that is, an index of a particle within 
    the array. the tip of the vector will be placed at this particle's position.
    
    returns: ndarray. shape=(cur_num_walkers, 3). an R^3 vector from `ptcl1` to `ptcl2` for each walker.
    '''
    arr = walkers.to_arr()

    return arr[:, :, ptcl2] - arr[:, :, ptcl1]

def bond_angle(bond1, bond2, bond1_len=None, bond2_len=None):
    if bond1_len is None:
        bond1_len = np.linalg.norm(bond1, axis=1)
    if bond2_len is None:
        bond2_len = np.linalg.norm(bond2, axis=1)

    dprods = np.sum(bond1*bond2, axis=1)
    bond_ang = np.arccos(dprods/(bond1_len*bond2_len))

    return bond_ang

###  BEGIN SIMULATOR EDIT ZONE  ###

###  Simulation Parameters:  ###

# atu
timeStep = 10.0

# equilibration steps. number of steps over which to "relax" the system.
nTimeStepsEq = 1000   

# production steps. number of steps over which to compute an average reference value 
# and walker population after equilibration.
nTimeStepsProd = 1000 

# initial population of walkers in the system (i.e., # of walkers BEFORE equilibration)
nWalkers = 10000

###  Particles:  ###

# particle name -> mass g/mole
particles2mass = {'h1': 1.00784, 'h2': 1.00784, 'o': 15.995}

nParticles = len(particles2mass)

###  Particle Indices:  ###
h1 = 0
h2 = 1
o = 2

###  Potential Energy Function:  ###
    
###  Constants:  ###

# alu
equilibriumPosition = 1.8100173464

# radians
equilibriumAngle = 1.82352000248

# aeu
k = 1

'''
--- walkers interface ---
walkers.to_arr():

walker1
          particles
        p * * * ...
        o * * * ...
        s * * * ...

walker2
          * * * ...
          * * * ...
          * * * ...

etc.

walkers.to_arr(): ndarray. shape=(cur_num_walkers, 3, nParticles)
    gives the ndarray representation of the current walker population. functions should operate upon this ndarray in 
    a vectorized way.

walkers.set_arr(new_arr): void. 
    new_arr: ndarray. shape=(cur_num_walkers, 3, nParticles)
    sets the current walker population (without deletion/replication) through an ndarray structured precisely analogously to 
    walkers.to_arr().

walkers.mass(ptcl): float.
    ptcl: str. name of particle (as specified in particles2mass above).
    retrieves the g/mole mass of the specified particle.

walkers.atomic_mass(ptcl): float.
    ptcl: str. name of particle (as specified in particles2mass above).
    retrieves the amu mass of the specified particle.

walkers.reduced_mass(): float.
    retrieves the amu reduced mass of the system.
'''
def potentialEnergyFunction(walkers):
    '''compute the potential energy of all the walkers (ideally in a vectorized way).
    walkers: Walkers object. see the Walkers interface above.
    
    returns: ndarray. shape=(cur_num_walkers,). a potential energy for each of the walkers currently in the system.
    '''

    bond1 = bond_vec(walkers, o, h1)
    bond2 = bond_vec(walkers, o, h2)

    bond1_len = np.linalg.norm(bond1, axis=1)
    bond2_len = np.linalg.norm(bond2, axis=1)

    bond_ang = bond_angle(bond1, bond2, bond1_len, bond2_len)

    bond_len_energy1 = (1/2)*k*(bond1_len - equilibriumPosition)**2
    bond_len_energy2 = (1/2)*k*(bond2_len - equilibriumPosition)**2
    bond_ang_energy = (1/2)*k*(bond_ang - equilibriumAngle)**2

    return bond_len_energy1 + bond_len_energy2 + bond_ang_energy

def diffusionFunction(walkers):
    '''diffuse (i.e., modify the position) of all the walkers (ideally in a vectorized way).
    walkers: Walkers object. see the walkers interface above.
    '''
    arr = walkers.to_arr()

    arr[:, :, h1] += np.random.normal(0.0, np.sqrt(timeStep/walkers.atomic_mass('h1')), arr[:, :, h1].shape)
    arr[:, :, h2] += np.random.normal(0.0, np.sqrt(timeStep/walkers.atomic_mass('h2')), arr[:, :, h2].shape)
    arr[:, :, o] += np.random.normal(0.0, np.sqrt(timeStep/walkers.atomic_mass('o')), arr[:, :, o].shape)

    walkers.set_arr(arr)

def posFunction(count):
    '''generate an ndarray representation of `count` walkers, with positions (i.e., the ndarray's data) initialized in some way.
    count: int. the number of walkers for which to generate positions.
    
    returns: ndarray. shape=(count, 3, nParticles). an ndarray representation of `count` walkers structured as described in 
    the walkers interface blurb above.
    '''
    pos = np.random.rand(count, 3, nParticles)
    return pos

###  END SIMULATOR EDIT ZONE  ###

if __name__ == '__main__':
    walkers = Walkers(particles2mass)

    sim = dmc()
    sim.setParams(timeStep, nTimeStepsEq, nTimeStepsProd, nWalkers, potentialEnergyFunction, diffusionFunction, posFunction, walkers)
    sim.run()
    sim.visualize_ref_vals()
    sim.visualize_walker_population()

    print(f'simulation should converge to {(1/2)*1*np.sqrt((k)/walkers.reduced_mass())} and converges approx. to {sim.mean_ref_val()}')