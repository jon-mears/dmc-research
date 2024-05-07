'''
Godfred Awuku and Jonathan Mears, with code adapted from Lindsey Madison

DMC_am_050424.py
CS446 -- DMC Research
Spring 2024
----------------------------------------------------------------------------------------------
Edit simulation parameters, potentialEnergyFunction, diffusionFunction, and posFunction below
and enter `python3 DMC_am_050424.py` into a terminal to run a DMC simulation with the specified 
parameters on the system described by the behavior of these three functions. (This has been 
preset to model the vibrating carbon monoxide bond (3) given in the project document, with the
recommended system parameters. 
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

class Walkers:
    def __init__(self, particles2mass, molecules2counts, rigid_or_flexible, init_cap=10000):
        '''
        Walkers constructor.

        particles2mass: dictionary: string -> float. a dictionary that maps particle names to their mass in g/mole.
        molecules2counts: dictionary: string -> int. a dictionary that maps molecule names to their number of occurrences in the target system.
        init_cap: int. the initial capacity of the internal `arr` ndarray in walker count.
        '''

        self.rigid_or_flexible = rigid_or_flexible

        # dictionary. string -> float. maps particle names to mass in g/mole.
        self.particles2mass = particles2mass

        # list of string. all names of particles in the system.
        self.particles = list(particles2mass.keys())

        # dictionary. string -> int. maps molecule names to the number of molecules in the system.
        self.molecules2counts = molecules2counts

        # list of string. all names of molecules in the system.
        self.molecules = list(molecules2counts.keys())

        # int. largest number of particles contained within a single molecule of the target system.
        self.nParticles = None

        # int. total number of molecules in the target system.
        self.nMolecules = None

        # dictionary. string -> int. maps '<molecule-name> <particle-name>' to an index within ndarray `arr`.
        self.ptclidx = dict()

        # dictionary. string -> int. maps '<molecule-name>' to an index within ndarray `arr`.
        self.mlclidx = dict()

        # dictionary. string -> float. maps particle names to mass in amu.
        self.particles2atomic = dict()

        # setup the system's particles
        self.setup_particles()

        # setup the system's molecules
        self.setup_molecules()

        # float. the reduced mass of the system in amu.
        self.rmass = self.compute_reduced_mass()

        # list of ndarray. stores the displacement vector between every pair of particles in the system.
        self.vecs = None

        # list of ndarray. stores the distance between every pair of particles in the system.
        self.dists = None
        
        # list of int. indices of current active walkers within the internal ndarray `arr`.
        self.walkers_idx = []

        # int. number of current active walkers.
        self.nWalkers = 0

        # int. number of walkers that can currently be fit in the internal ndarray `arr`.
        self.cap = init_cap

        # int. the next index at which to place a new walker.
        self.next_idx = 0

        if self.rigid_or_flexible == 'flexible':
            # ndarray. shape=(self.cap, self.nMolecules, self.nParticles, 3). the internal ndarray that contains a description of the walkers, i.e., 
            # several arrangements of Cartesian positions of the particles in the system.
            self.arr = np.zeros((self.cap, self.nMolecules, self.nParticles, 3))
        elif self.rigid_or_flexible == 'rigid':
            self.arr = np.zeros((self.cap, self.nMolecules, 6))

    def setup_particles(self):
        '''
        setup the particles of the system by computing the atomic mass (amu) of each.
        '''

        # compute the atomic mass of each particle, and create an entry for the particle in the amu dictionary 
        for ptcl in self.particles:
            self.particles2atomic[ptcl] = self.particles2mass[ptcl]/(6.02213670000e23*9.10938970000e-28)

    def setup_molecules(self):
        '''setup several aspects of the molecules of the system, including the '''

        self.nMolecules = 0
        # the first index that should be associated to the current molecule
        m_idx = 0

        # the largest number of particles encountered within a single molecule
        max_ptcls = 0

        # for each molecule...
        for mlcl in self.molecules:
            # the total number of particles within the current molecule
            total_ptcls = 0

            # the first index associated to the current particle within the current molecule
            p_idx = 0

            # index within the molecule name; points to the start of the current particle substring
            chi = 0

            # iterate through the particle substrings of the current molecule
            while chi != len(mlcl):

                # index within the molecule name; points one past the end of the current particle substring
                chj = chi + 1

                # index within the molecule name; points to the start of numeric data within the current particle substring, or
                # None if the current particle substring does not contain numeric data
                chk = None 

                # iterate to one character past the end of the current particle substring
                while chj != len(mlcl) and (mlcl[chj].isnumeric() or mlcl[chj].islower()):

                    # record the start of numeric data within the current particle substring
                    if mlcl[chj].isnumeric() and chk is None:
                        chk = chj
                    chj += 1

                # if the current particle substring contains numeric data...
                if chk is not None:

                    # record the name of the particle
                    ptcl_name = mlcl[chi:chk]

                    # record its number of occurrences within the molecule
                    ptcl_count = int(mlcl[chk:chj])

                # if the current particle substring does not contain numeric data...
                else:

                    # record the name of the particle
                    ptcl_name = mlcl[chi:chj]

                    # the particle only occurs once within the molecule
                    ptcl_count = 1

                # record the indices of the current particle within the current molecule
                self.ptclidx[mlcl + ' ' + ptcl_name] = list(range(p_idx, p_idx+ptcl_count))

                # setup the start index for the next particle
                p_idx += ptcl_count

                # increment the total number of particles within this molecule
                total_ptcls += ptcl_count

                # start index of the next particle substring is one-past-the-end of the current particle substring
                chi = chj

            # sum up the total number of molecules in the target system
            self.nMolecules += self.molecules2counts[mlcl]

            # record the indices of the current molecule
            self.mlclidx[mlcl] = list(range(m_idx, m_idx + self.molecules2counts[mlcl]))

            # setup the start index for the next molecule
            m_idx += self.molecules2counts[mlcl]

            # take the max of the number of particles within the current molecule and the largest number of particles seen within a 
            # single molecule thus far
            if total_ptcls > max_ptcls:
                max_ptcls = total_ptcls

        # record the largest number of particles within a single molecule
        self.nParticles = max_ptcls

    def get_vector(self, sourcemlcl, destmlcl, sourceptcl, destptcl):
        return self.vecs[self.molecule(sourcemlcl)][:, self.molecule(destmlcl) - self.molecule(sourcemlcl), self.particle(sourceptcl), self.particle(destptcl), :]

    def molecule(self, key):
        '''given a key of the form '<molecule-name> <1-indexed idx>', returns the index of that molecule in `arr`.
        
        key: string. key of the form '<molecule-name> <1-indexed idx>'. e.g., 'H2O 2' returns the index of the "second" 
        H2O molecule in the system.
        
        returns: int. index of the specified molecule in the "molecule axis" of `arr`.
        '''

        name = key.split()[0]
        idx = int(key.split()[1])-1

        return self.mlclidx[name][idx]
    
    def particle(self, key):
        '''given a key of the form '<molecule-name> <particle-name> <1-indexed idx>, returns the index of that particle in `arr`.
        
        key: string. key of the form '<molecule-name> <particle-name> <1-indexed idx>. e.g., 'H2O H 2' returns the index of the 
        "second" H atom in the H2O molecules of the system.
        '''

        info = key.split()

        mlcl = info[0]
        ptcl = info[1]
        idx = int(info[2])-1

        return self.ptclidx[mlcl + ' ' + ptcl][idx]

    def to_arr(self):
        '''return the current "active walkers" stored in `arr`.'''
        return self.arr[self.walkers_idx]
    
    def set_arr(self, new_arr, isUpdate=True):
        '''
        assign or update the internal ndarray `arr`. 

        `new_arr`: ndarray. shape=(cur_num_walkers || new_num_walkers, 3, nParticles). the new internal ndarray to update or assign `arr`.
        `isUpdate`: bool. if `True`, `cur_num_walkers` is taken in the shape above, and `new_arr` should essentially contain all walkers 
        previously in `arr`, but arbitrarily updated (e.g., `isUpdate` is `True` after the walkers have been modified in `diffusionFunction`).
        if `False`, `new_num_walkers` is taken in the shape above, and `new_arr` can be an arbitrary internal ndarray of walkers, except that the
        number of particles must match the number of particles set at initialization of this object (and order will be used to determine the identity
        of each particle).
        '''

        # if this is an update to the walkers, simply assign `new_arr` to the indices of the walkers already in `arr`
        if isUpdate:
            self.arr[self.walkers_idx] = new_arr

        # if this is more than just an update to the current walkers, use info in `new_arr` to essentially 
        # assign `new_arr` to `arr`
        else:
            self.nWalkers = new_arr.shape[0]
            print(f'self.nWalkers: {self.nWalkers}')
            self.walkers_idx = range(self.nWalkers)
            self.next_idx = self.nWalkers

            if self.nWalkers > self.cap:
                self.realloc(self.nWalkers*2)

            self.arr[self.walkers_idx, :, :] = new_arr
    
    def realloc(self, new_cap=None):
        '''
        reallocate the internal `arr` ndarray. 

        new_cap: int or None. the new capacity of the internal `arr` ndarray. no-op if `new_cap` is 
        less than the current number of walkers, in which case the reallocation would be destructive.
        if None, the new capacity is twice the current number of walkers.
        '''
        # set the new capacity to twice the number of walkers if `new_cap` was not specified
        if new_cap is None:
            new_cap = self.nWalkers*2

        # no-op if `new_cap` is less than the current number of walkers
        if new_cap < self.nWalkers:
            return
        
        # set the new capacity
        self.cap = new_cap
        
        # allocate the new internal ndarray
        if self.rigid_or_flexible == 'flexible':
            new_arr = np.zeros((self.cap, self.nMolecules, self.nParticles, 3))
        elif self.rigid_or_flexible == 'rigid':
            new_arr = np.zeros((self.cap, self.nMolecules, 6))

        # place the walkers in the newly allocated internal ndarray
        new_arr[:self.nWalkers] = self.arr[self.walkers_idx]
        self.arr = new_arr

        # update indices accordingly
        self.walkers_idx = list(range(self.nWalkers))
        self.next_idx = self.nWalkers

    def make(self, count, gen):
        '''
        add `count` new walkers, with positions generated by the function `gen`.
        count: int. number of walkers to create and add.
        gen: func(count) -> (count, nMolecules, nParticles, 3) ndarray. func that generates an initialized ndarray of a given shape.
        '''

        # reallocate if there is not enough space to accomodate the new walkers
        if self.next_idx + count > self.cap:
            self.realloc((self.nWalkers+count)*2)

        # generate the new walkers
        new_walkers = gen(count)

        # add new walkers to `arr`, and extend `walkers_idx` to index the newly added walkers
        self.arr[self.next_idx:self.next_idx+count] = new_walkers
        self.walkers_idx.extend(range(self.next_idx, self.next_idx+count))

        # increment internal count variables
        self.nWalkers += count
        self.next_idx += count

    def replicate(self, idx):
        '''
        replicate the walkers at the indices `idx`.

        idx: list of int. the indices of walkers in `arr` to replicate.
        '''

        # obtain the collection of walkers to replicate
        to_copy = self.arr[idx]

        # reallocate if the replication would exceed the current capacity
        if self.next_idx + len(idx) > self.cap:
            self.realloc()

        # add the replicated walkers to the end of the internal ndarray `arr`
        self.arr[self.next_idx:self.next_idx + len(idx)] = to_copy

        # update `walkers_idx` to include the indices of the newly replicated walkers
        self.walkers_idx.extend(range(self.next_idx, self.next_idx + len(idx)))

        # update the internal walker counts
        self.nWalkers += len(idx)
        self.next_idx += len(idx)
    
    def delete(self, idx):
        '''
        delete the walkers at the indices `idx`.

        idx: list of int. the indices of walkers in `arr` to delete.
        '''

        # remove the indices of walkers to delete from the list of active walker indices
        self.walkers_idx = list(filter(lambda i: i not in idx, self.walkers_idx))

        # update the internal walker counts
        self.nWalkers -= len(idx)

    def mass(self, ptcl):
        '''return the mass in g/mole of the particle `ptcl` (string)'''
        return self.particles2mass[ptcl]
    
    def atomic_mass(self, ptcl):
        '''return the mass in amu of the particle `ptcl` (string).'''
        return self.particles2atomic[ptcl]
    
    def compute_reduced_mass(self):
        # compute the reduced mass of the system
        product = 1
        sum = 0
        
        for ptcl in self.particles:
            product *= self.particles2atomic[ptcl]
            sum += self.particles2atomic[ptcl]

        return product/sum

    def reduced_mass(self):
        '''return the reduced mass in amu of the system.'''
        return self.rmass
    
    def update_vecs(self, arr=None):
        '''
        bonds[nMolecules (source)] = (nWalkers, nMolecules (destination), nParticles (source), nParticles (destination), 3)
        '''
        if arr is None:
            arr = self.to_arr()
        self.vecs = []

        for mlcl in self.molecules:
            for mlclidx in range(self.molecules2counts[mlcl]):
                self.vecs.append(np.zeros((self.nWalkers, self.nMolecules - mlclidx, self.nParticles, self.nParticles, 3)))

                for ptclidx in range(self.nParticles):
                    self.vecs[mlclidx][:, :, ptclidx, :, :] = arr[:, mlclidx:, :, :] - arr[:, mlclidx, ptclidx, :].reshape(self.nWalkers, 1, 1, 3)

    def update_dists(self):
        self.dists = []

        for mlclidx in range(self.nMolecules):
            self.dists.append(np.linalg.norm(self.vecs[mlclidx], axis=4))     

    def reset(self):
        self.vecs = None
        self.dists = None

        self.walkers_idx = []
        self.nWalkers = 0
        self.cap = 10000
        if self.rigid_or_flexible == 'flexible':
            self.arr = np.zeros((self.cap, self.nMolecules, self.nParticles, 3))
        elif self.rigid_or_flexible == 'rigid':
            self.arr = np.zeros((self.cap, self.nMolecules, 6))
    
class DMC:
    """DMC Simulation."""
    def __init__(self):
        self.time_step = None
        self.nTimeSteps = None
        self.nWalkers = None

        self.nTrials = None

        self.val_func = None
        self.diffuse_func = None
        self.gen_func = None

        self.walkers = None
        self.cur_walker_count = None

        self.seed = None

        self.ref_val = None

        self.data = dict()
        self.mean_data = dict()
        self.iter = None
        self.trial_iter = 0

        self.var_threshold = 1e-5

    def setParams(self, time_step, nTimeSteps, nWalkers, val_func, diffuse_func, gen_func, walkers, nTrials=1, seed=None):
        self.time_step = time_step
        self.nTimeSteps = nTimeSteps
        self.nTrials = nTrials
        self.nWalkers = nWalkers

        self.val_func = val_func
        self.diffuse_func = diffuse_func
        self.gen_func = gen_func

        self.walkers = walkers
        self.seed = seed

        self.data['reference values'] = np.zeros(self.nTimeSteps)
        self.data['walker populations'] = np.zeros(self.nTimeSteps+1)

        self.mean_data['reference values'] = np.zeros(self.nTrials)
        self.mean_data['walker populations'] = np.zeros(self.nTrials)

    def init(self):
        walkers.reset()

        self.walkers.make(self.nWalkers, self.gen_func)

        self.data['reference values'] = np.zeros(self.nTimeSteps)
        self.data['walker populations'] = np.zeros(self.nTimeSteps+1)

        self.cur_walker_count = self.walkers.nWalkers

        np.random.seed(self.seed)

    def reset(self):
        self.walkers.reset()

        for entry in self.data:
            self.data[entry] = np.zeros(self.nTimeSteps)

    def penalty(self):
        return -(self.cur_walker_count-self.nWalkers)/(self.time_step*self.nWalkers)

    def compute_ref_val(self):
        return np.mean(self.val_func(self.walkers)) + self.penalty()

    def diffuse(self):
        self.diffuse_func(self.walkers)

    def adjust_walker_population(self):        
        idx = np.array(self.walkers.walkers_idx)

        vals = self.val_func(self.walkers)

        greater_vals = vals[vals > self.ref_val]
        greater_idx = idx[vals > self.ref_val]  

        prob_delete = np.clip(np.exp(-(greater_vals-self.ref_val)*self.time_step, dtype=np.float128), 0, 1)

        delete_rand = np.random.uniform(0.0, 1.0, (greater_vals.size))
        delete_idx = greater_idx[prob_delete < delete_rand]

        lesser_vals = vals[vals < self.ref_val]
        lesser_idx = idx[vals < self.ref_val]

        prob_replicate = np.clip(np.exp(-(lesser_vals-self.ref_val)*self.time_step, dtype=np.float128) - 1.0, 0, 1)

        replicate_rand = np.random.uniform(0.0, 1.0, (lesser_vals.size))
        replicate_idx = lesser_idx[prob_replicate > replicate_rand]

        self.walkers.delete(delete_idx.tolist())
        self.walkers.replicate(replicate_idx.tolist())

    def find_convergence_point(self, data):
        for step in range(int(self.nTimeSteps/2), self.nTimeSteps, int(self.nTimeSteps/16)):
            if np.var(self.data[data][step:]) < self.var_threshold:
                return step
            
        return -1

    def simulation(self):
        self.data['walker populations'][0] = self.walkers.nWalkers

        for self.iter in range(self.nTimeSteps):
            print(self.iter)
            print(self.walkers.nWalkers)
            self.ref_val = self.compute_ref_val()
            self.diffuse()
            self.adjust_walker_population()

            self.data['reference values'][self.iter] = self.ref_val
            self.data['walker populations'][self.iter+1] = self.walkers.nWalkers

            self.cur_walker_count = self.walkers.nWalkers
                
    def run(self):
        while self.trial_iter < self.nTrials:
            try:
                self.init()
                self.simulation()
                
                for entry in self.data:
                    convergence_point = self.find_convergence_point('reference values')
                    self.mean_data[entry][self.trial_iter] = np.mean(self.data[entry][convergence_point:])

                self.trial_iter += 1

            except ZeroDivisionError:
                if self.seed is not None:
                    exit('stochastic processes associated with this seed leads to a divsion by zero.')
                else:
                    print('stochastic processes led to division by zero. now starting a new simulation.')
                    self.reset()
                    self.run()

    # def visualize_ref_vals(self):
    #     plt.plot(np.arange(self.ref_vals.size), self.ref_vals)
    #     plt.plot(np.arange(self.ref_vals.size), self.mean_ref_val()*np.ones((self.ref_vals.size)))
    #     plt.title('Reference Value vs. Time Step')
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Reference Value')
    #     plt.show()

    # def visualize_walker_population(self):
    #     plt.plot(np.arange(self.walker_counts.size), self.walker_counts)
    #     plt.plot(np.arange(self.walker_counts.size), self.mean_walker_population()*np.ones((self.walker_counts.size)))
    #     plt.title('Walker Count vs. Time Step')
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Walker Count')
    #     plt.show()

    def mean_ref_val(self):
        return np.mean(self.mean_data['reference values'])

    # def mean_walker_population(self):
    #     return np.mean(self.walker_counts)
    
    def register_data(self, name):
        self.data[name] = np.zeros(self.nTimeSteps)
        self.mean_data[name] = np.zeros(self.nTrials)

    def add_data(self, name, data):
        self.data[name][self.iter] = data     
    
###  UTILITY FUNCTIONS  ###

def bond_angles(bond1_vector, bond2_vector, bond1_len=None, bond2_len=None):

    if bond1_len is None:
        bond1_len = np.linalg.norm(bond1_vector, axis=1)
    if bond2_len is None:
        bond2_len = np.linalg.norm(bond2_vector, axis=1)

    dprods = np.sum(bond1_vector*bond2_vector, axis=1)
    angles = np.arccos(dprods/(bond1_len*bond2_len))

    return angles

def rotation_matrix(axis, rad):
    if axis=='x':
        return np.array([[1, 0, 0, 0], [0, np.cos(rad), -np.sin(rad), 0], [0, np.sin(rad), np.cos(rad), 0], [0, 0, 0, 1]])
    if axis=='y':
        return np.array([[np.cos(rad), 0, np.sin(rad), 0], [0, 1, 0, 0], [-np.sin(rad), 0, np.cos(rad), 0], [0, 0, 0, 1]])
    if axis=='z':
        return np.array([[np.cos(rad), -np.sin(rad), 0, 0], [np.sin(rad), np.cos(rad), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
def translation_matrix(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

def rigid_matrices(walkers):
    arr = walkers.to_arr()
    
    translations = np.zeros((walkers.nWalkers, walkers.nMolecules, 4, 4))
    translations[:, :, :, :] = np.eye(4)

    translations[:, :, :3, 3] = arr[:, :, :3]

    xrotations = np.zeros((walkers.nWalkers, walkers.nMolecules, 4, 4))
    xrad = arr[:, :, 3]
    xrotations[:, :, :, :] = np.eye(4)
    xrotations[:, :, 1, 1] = np.cos(xrad)
    xrotations[:, :, 1, 2] = -np.sin(xrad)
    xrotations[:, :, 2, 1] = np.sin(xrad)
    xrotations[:, :, 2, 2] = np.cos(xrad)

    yrotations = np.zeros((walkers.nWalkers, walkers.nMolecules, 4, 4))
    yrad = arr[:, :, 4]
    yrotations[:, :, :, :] = np.eye(4)
    yrotations[:, :, 0, 0] = np.cos(yrad)
    yrotations[:, :, 0, 2] = np.sin(yrad)
    yrotations[:, :, 2, 0] = -np.sin(yrad)
    yrotations[:, :, 2, 2] = np.cos(yrad)

    zrotations = np.zeros((walkers.nWalkers, walkers.nMolecules, 4, 4))
    zrad = arr[:, :, 5]
    zrotations[:, :, :, :] = np.eye(4)
    zrotations[:, :, 0, 0] = np.cos(zrad)
    zrotations[:, :, 0, 1] = -np.sin(zrad)
    zrotations[:, :, 1, 0] = np.sin(zrad)
    zrotations[:, :, 1, 1] = np.cos(zrad)

    return translations  @ yrotations @ xrotations @ zrotations
    
def rigid_arr(walkers, std_mlcl):
    matrices = rigid_matrices(walkers)

    return (matrices @ std_mlcl.T)[:, :].transpose((0, 1, 3, 2))

###  BEGIN SIMULATOR EDIT ZONE  ###

###  Simulation Parameters:  ###

# atu
timeStep = 10

# time steps of simulation
nTimeSteps = 2000

nTrials = 1

# initial population of walkers in the system
nWalkers = 10000

seed = None

###  Particles:  ###

# particle name -> mass g/mole
particles2mass = {'H': 1.007825, 'O': 15.99491461957, 'M': 0.0}

# molecule name (should be comprised of names of particles above, each possibly followed by a count) ->
#   count of molecule in system
molecules2counts = {'H2OM': 2}

###  Potential Energy Function:  ###
    
###  Constants:  ###

# hartrees*bohrs/e^2
kc = 332.1*(1.88973*4.184)/2625.5

# bohrs
rOH = 0.9572 * 1.88973

# radians
aHOH = 104.52 * np.pi/180.0 

# bohrs
rOM = 0.15 * 1.88973

# hartrees * bohrs^12
A = 600.0e3*(1.88973**12 * 4.184)/2625.5

# hartrees * bohrs^6
B = 610.0*(1.88973**6 * 4.184)/2625.5

origin_H2OM = np.zeros((4, 4))

# homogeneous coordinate
origin_H2OM[:, 3] = 1

# first hydrogen particle
origin_H2OM[0, :3] = np.array([rOH, 0, 0])

# second hydrogen particle
origin_H2OM[1] = rotation_matrix('z', aHOH) @ origin_H2OM[0]
origin_H2OM[1, :3] = rOH*(origin_H2OM[1, :3]/np.linalg.norm(origin_H2OM[1, :3]))

# oxygen particle is at the "origin"
origin_H2OM[2, :3] = origin_H2OM[2, :3]

# M "particle"
origin_H2OM[3] = rotation_matrix('z', aHOH/2) @ origin_H2OM[0]
origin_H2OM[3, :3] = rOM*(origin_H2OM[3, :3]/np.linalg.norm(origin_H2OM[3, :3]))

dist_threshold = 0

def coulombic(walkers):

    qO = 0.0
    qH = 0.52
    qM = -1.04

    col = np.zeros(walkers.nWalkers)
    for srcmlclidx in range(walkers.nMolecules-1):

        srcptclidxHH, destptclidxHH = np.meshgrid(walkers.ptclidx['H2OM H'], walkers.ptclidx['H2OM H'])
        srcptclidxHH = srcptclidxHH.flatten()
        destptclidxHH = destptclidxHH.flatten()
        
        dHH = walkers.dists[srcmlclidx][:, 1:, srcptclidxHH, destptclidxHH]
        dHH = dHH.reshape(walkers.nWalkers, int(dHH.size/walkers.nWalkers))

        dHH[dHH <= dist_threshold] = np.infty

        col += np.sum(kc*qH*qH/dHH, axis=1)

        # srcptclidxHO, destptclidxHO = np.meshgrid(walkers.ptclidx['H2OM H'], walkers.ptclidx['H2OM O'])
        # srcptclidxHO = srcptclidxHO.flatten()
        # destptclidxHO = destptclidxHO.flatten()

        # dHO = walkers.dists[srcmlclidx][:, 1:, srcptclidxHO, destptclidxHO]
        # dHO = dHO.reshape(walkers.nWalkers, int(dHO.size/walkers.nWalkers))

        # dHO[dHO <= dist_threshold] = np.infty

        # col += np.sum(qH*qO/dHO, axis=1)

        # srcptclidxOH, destptclidxOH = np.meshgrid(walkers.ptclidx['H2OM O'], walkers.ptclidx['H2OM H'])
        # srcptclidxOH = srcptclidxOH.flatten()
        # destptclidxOH = destptclidxOH.flatten()

        # dOH = walkers.dists[srcmlclidx][:, 1:, srcptclidxOH, destptclidxOH]
        # dOH = dOH.reshape(walkers.nWalkers, int(dOH.size/walkers.nWalkers))

        # dOH[dOH <= dist_threshold] = np.infty
        # col += np.sum(qO*qH/dOH, axis=1)

        # srcptclidxOO, destptclidxOO = np.meshgrid(walkers.ptclidx['H2OM O'], walkers.ptclidx['H2OM O'])
        # srcptclidxOO = srcptclidxOO.flatten()
        # destptclidxOO = destptclidxOO.flatten()

        # dOO = walkers.dists[srcmlclidx][:, 1:, srcptclidxOO, destptclidxOO]
        # dOO = dOO.reshape(walkers.nWalkers, int(dOO.size/walkers.nWalkers))

        # dOO[dOO <= dist_threshold] = np.infty
        # col += np.sum(qO*qO/dOO, axis=1)

        srcptclidxHM, destptclidxHM = np.meshgrid(walkers.ptclidx['H2OM H'], walkers.ptclidx['H2OM M'])
        srcptclidxHM = srcptclidxHM.flatten()
        destptclidxHM = destptclidxHM.flatten()
        
        dHM = walkers.dists[srcmlclidx][:, 1:, srcptclidxHM, destptclidxHM]
        dHM = dHM.reshape(walkers.nWalkers, int(dHM.size/walkers.nWalkers))

        dHM[dHM <= dist_threshold] = np.infty

        col += np.sum(kc*qH*qM/dHM, axis=1)

        srcptclidxMH, destptclidxMH = np.meshgrid(walkers.ptclidx['H2OM M'], walkers.ptclidx['H2OM H'])
        srcptclidxMH = srcptclidxMH.flatten()
        destptclidxMH = destptclidxMH.flatten()
        
        dMH = walkers.dists[srcmlclidx][:, 1:, srcptclidxMH, destptclidxMH]
        dMH = dMH.reshape(walkers.nWalkers, int(dMH.size/walkers.nWalkers))

        dMH[dMH <= dist_threshold] = np.infty

        col += np.sum(kc*qM*qH/dMH, axis=1)

    return col

def lennard_jones(walkers):
    lj = np.zeros(walkers.nWalkers)
    for srcmlclidx in range(walkers.nMolecules-1):

        srcptclidxOO, destptclidxOO = np.meshgrid(walkers.ptclidx['H2OM O'], walkers.ptclidx['H2OM O'])
        srcptclidxOO = srcptclidxOO.flatten()
        destptclidxOO = destptclidxOO.flatten()

        dOO = walkers.dists[srcmlclidx][:, 1:, srcptclidxOO, destptclidxOO]
        dOO = dOO.reshape(walkers.nWalkers, int(dOO.size/walkers.nWalkers))

        dOO[dOO <= dist_threshold] = np.infty

        lj += np.sum((A/dOO**12) - (B/dOO**6), axis=1)

    return lj

def inter_potential(walkers):
    return coulombic(walkers) + lennard_jones(walkers)

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

    walkers.update_vecs(rigid_arr(walkers, origin_H2OM)[:, :, :, :3])
    walkers.update_dists()
    pot = inter_potential(walkers)

    return pot

def diffusionFunction(walkers):
    '''diffuse (i.e., modify the position) of all the walkers (ideally in a vectorized way).
    walkers: Walkers object. see the walkers interface above.
    '''
    arr = walkers.to_arr()
    arr[:, :, :] += np.random.normal(0.0, np.sqrt(timeStep/(2*walkers.atomic_mass('H')+walkers.atomic_mass('O'))))

    walkers.set_arr(arr)

def posFunction(count):
    '''generate an ndarray representation of `count` walkers, with positions (i.e., the ndarray's data) initialized in some way.
    count: int. the number of walkers for which to generate positions.
    
    returns: ndarray. shape=(count, 3, nParticles). an ndarray representation of `count` walkers structured as described in 
    the walkers interface blurb above.
    '''
    pos = np.zeros((count, 2, 6))

    mlcl1 = np.random.normal(0, 0.5, (count, 1, 6))
    mlcl2 = np.random.normal(10, 0.5, (count, 1, 6))
    
    pos[:, 0] = mlcl1.reshape(count, 6)
    pos[:, 1] = mlcl2.reshape(count, 6)
    
    # pos = 5*np.random.rand(count, 2, 6)

    return pos

###  END SIMULATOR EDIT ZONE  ###

if __name__ == '__main__':
    walkers = Walkers(particles2mass, molecules2counts, 'rigid')

    sim = DMC()
    sim.setParams(timeStep, nTimeSteps, nWalkers, potentialEnergyFunction, diffusionFunction, posFunction, walkers, nTrials, seed)

    sim.register_data('HO bond 1')
    sim.register_data('HO bond 2')

    sim.run()

    print(f'simulation converges approx. to {sim.mean_ref_val()}')