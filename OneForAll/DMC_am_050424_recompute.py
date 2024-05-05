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

np.random.seed(1)

class Walkers:
    def __init__(self, particles2mass, molecules2counts, init_cap=10000):
        '''
        Walkers constructor.

        particles2mass: dictionary: string -> float. a dictionary that maps particle names to their mass in g/mole.
        molecules2counts: dictionary: string -> int. a dictionary that maps molecule names to their number of occurrences in the target system.
        init_cap: int. the initial capacity of the internal `arr` ndarray in walker count.
        '''
        
        # dictionary. string -> float. maps particle names to mass in g/mole.
        self.particles2mass = particles2mass

        # list of string. all names of particles in the system.
        self.particles = list(particles2mass.keys())

        # dictionary. string -> int. maps molecule names to the number of molecules in the system.
        self.molecules2counts = molecules2counts

        # list of string. all names of molecules in the system.
        self.molecules = list(molecules2counts.keys())

        # dictionary. string -> list of string. maps molecule names to the names of particle contained therein.
        self.molecules2particles = dict()

        # dictionary. int -> string. maps molecule index to name and 1-index within system.
        self.index2molecule = dict()

        # dictionary. string -> int. maps molecule names to its total number of particles.
        self.molecules2particlecounts = dict()

        # dictionary. string -> float. maps particle names to mass in amu.
        self.particles2atomic = dict()

        # dictionary. string -> int. maps '<molecule-name> <particle-name>' to an index within ndarray `arr`.
        self.ptclidx = dict()

        # dictionary. string -> int. maps [molecule name] to an index within ndarray `arr`.
        self.mlclidx = dict()

        # register each of the system's particles
        for ptcl in particles2mass:
            self.register_particle(ptcl, particles2mass[ptcl])

        self.vecs = None
        self.dists = None

        # compute the reduced mass of the system
        prod = 1
        sum = 0
        
        for ptcl in self.particles:
            prod *= self.particles2atomic[ptcl]
            sum += self.particles2atomic[ptcl]

        # float. the reduced mass of the system in amu.
        self.rmass = prod/sum

        # int. total number of molecules in the target system.
        self.nMolecules = 0

        # int. largest number of particles contained within a single molecule of the target system.
        self.nParticles = 0

        self.register_molecules()
        
        # list of int. indices of current active walkers within the internal ndarray `arr`.
        self.walkers_idx = []

        # int. number of current active walkers.
        self.nWalkers = 0

        # int. number of walkers that can currently be fit in the internal ndarray `arr`.
        self.cap = init_cap

        # int. the next index at which to place a new walker.
        self.next_idx = 0

        # ndarray. shape=(self.cap, self.nMolecules, self.nParticles, 3). the internal ndarray that contains a description of the walkers, i.e., 
        # several arrangements of Cartesian positions of the particles in the system.
        self.arr = np.zeros((self.cap, self.nMolecules, self.nParticles, 3))

    def register_particle(self, name, mass):
        '''
        insert the particle described by `name` and `mass` into the data structures of this object, and compute its atomic mass.

        name: string. name of the particle—its key within this object's dictionaries.
        mass: float. mass of the particle in g/mole.
        '''

        # compute the atomic mass of the particle, and create an entry for the particle in the amu dictionary 
        # of this object
        self.particles2atomic[name] = mass/(6.02213670000e23*9.10938970000e-28)

    def register_molecules(self):
        ''''''

        # the first index that should be associated to the current molecule
        m_idx = 0

        # the largest number of particles encountered within a single molecule
        max_ptcls = 0

        # for each molecule...
        for mlcl in self.molecules2counts:

            # add a key to the molecules2particles dictionary
            self.molecules2particles[mlcl] = []

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

                # add the particle name to the list of particles contained within the current molecule
                self.molecules2particles[mlcl].append(ptcl_name)

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

            for idx in range(m_idx, m_idx + self.molecules2counts[mlcl]):
                self.index2molecule[idx] = mlcl + ' ' + str(idx - m_idx + 1)

            # setup the start index for the next molecule
            m_idx += self.molecules2counts[mlcl]

            # map the molecule name to its total number of particles
            self.molecules2particlecounts[mlcl] = total_ptcls

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
    
    # def bond(self, key):
    #     return self.bdidx[key]

    def to_arr(self):
        '''return the current "active walkers" stored in `arr`.'''
        return self.arr[self.walkers_idx, :, :, :]
    
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
            self.arr[self.walkers_idx, :, :, :] = new_arr

        # if this is more than just an update to the current walkers, use info in `new_arr` to essentially 
        # assign `new_arr` to `arr`
        else:
            self.nWalkers = new_arr.shape[0]
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
        new_arr = np.zeros((self.cap, self.nMolecules, self.nParticles, 3))

        # place the walkers in the newly allocated internal ndarray
        new_arr[:self.nWalkers, :, :, :] = self.arr[self.walkers_idx, :, :, :]
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
        to_copy = self.arr[idx, :, :, :]

        # reallocate if the replication would exceed the current capacity
        if self.next_idx + len(idx) > self.cap:
            self.realloc()

        # add the replicated walkers to the end of the internal ndarray `arr`
        self.arr[self.next_idx:self.next_idx + len(idx), :, :, :] = to_copy

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

    def reduced_mass(self):
        '''return the reduced mass in amu of the system.'''
        return self.rmass
    
    def update_vecs(self):
        '''
        bonds[nMolecules (source)] = (nWalkers, nMolecules (destination), nParticles (source), nParticles (destination), 3)
        '''

        arr = self.to_arr()
        self.vecs = []

        for mlclidx in range(self.nMolecules):

            mlcl_name = self.index2molecule[mlclidx].split()[0]

            self.vecs.append(np.zeros((self.nWalkers, self.nMolecules - mlclidx, self.nParticles, self.nParticles, 3)))

            for ptclidx in range(self.molecules2particlecounts[mlcl_name]):
                self.vecs[mlclidx][:, :, ptclidx, :, :] = arr[:, mlclidx:, :, :] - arr[:, mlclidx, ptclidx, :].reshape(self.nWalkers, 1, 1, 3)

    def update_dists(self):
        self.dists = []

        for mlclidx in range(self.nMolecules):
            self.dists.append(np.linalg.norm(self.vecs[mlclidx], axis=4))     
    
class DMC:
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
        # pen = (1.0-(self.cur_walker_count/self.nWalkers))/(2.0*self.time_step)
        pen = -(self.cur_walker_count-self.nWalkers)/(self.time_step*self.nWalkers)

        print(f'penalty: {pen}')

        return pen


    def compute_ref_val(self):
        print('in compute_ref_val')
        r = np.mean(self.val_func(self.walkers)) + self.penalty()
        # r = np.mean(self.val_func(self.walkers)) + self.penalty()

        print(f'reference energy: {r}')
        return r

    def diffuse(self):
        # print(f'\nWALKERS:\n{self.walkers.to_arr()}\n')

        DF = pd.DataFrame(self.walkers.to_arr().reshape(int(self.walkers.to_arr().size/3), 3))
        DF.to_csv('before.csv')
        self.diffuse_func(self.walkers)
        # print(f'\nWALKERS:\n{self.walkers.to_arr()}\n')
        DF = pd.DataFrame(self.walkers.to_arr().reshape(int(self.walkers.to_arr().size/3), 3))
        DF.to_csv('after.csv')

    def adjust_walker_population(self):        
        idx = np.array(self.walkers.walkers_idx)

        print('in adjust_walker_population')
        vals = self.val_func(self.walkers)

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

    def equilibration(self):
        for i in range(self.nStepsEq):
            print(f'timestep: {i}')
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

def bond_vectors(walkers, mlcl, ptcl1, ptcl2, ptcl1_idx=0, ptcl2_idx=0):
    '''return the bond between `ptcl1` and `ptcl2` (global variables that were assigned indices into the third axis
    of the `walkers` ndarray) as a vector in R^3, origin at `ptcl1`.
    walkers: Walkers object. see the Walkers interface below.
    ptcl1: int. index into the third axis of the `arr` ndarray of `walkers`; that is, an index of a particle within
    the array. the tail of the vector will be placed at this particle's position.
    ptcl2: int. index into the third axis of the `arr` ndarray of `walkers`; that is, an index of a particle within 
    the array. the tip of the vector will be placed at this particle's position.
    
    returns: ndarray. shape=(cur_num_walkers, molecules2counts[mlcl], 3). an R^3 vector from `ptcl1` to `ptcl2` for each walker.
    '''
    arr = walkers.to_arr()

    return arr[:, walkers.mlclidx[mlcl], walkers.ptclidx[mlcl + ' ' + ptcl1][ptcl1_idx], :] - \
        arr[:, walkers.mlclidx[mlcl], walkers.ptclidx[mlcl + ' ' + ptcl2][ptcl2_idx], :]

def bond_angles(bond1_vector, bond2_vector, bond1_len=None, bond2_len=None):

    if bond1_len is None:
        bond1_len = np.linalg.norm(bond1_vector, axis=1)
    if bond2_len is None:
        bond2_len = np.linalg.norm(bond2_vector, axis=1)

    dprods = np.sum(bond1_vector*bond2_vector, axis=1)
    angles = np.arccos(dprods/(bond1_len*bond2_len))

    return angles

###  BEGIN SIMULATOR EDIT ZONE  ###

###  Simulation Parameters:  ###

# atu
timeStep = 10

# equilibration steps. number of steps over which to "relax" the system.
nTimeStepsEq = 1000  

# production steps. number of steps over which to compute an average reference value 
# and walker population after equilibration.
nTimeStepsProd = 1000 

# initial population of walkers in the system (i.e., # of walkers BEFORE equilibration)
nWalkers = 500 

###  Particles:  ###

# particle name -> mass g/mole
particles2mass = {'H': 1.007825, 'O': 15.99491461957}

# molecule name (should be comprised of names of particles above, each possibly followed by a count) ->
#   count of molecule in system
molecules2counts = {'H2O': 2}
nParticles = len(particles2mass)

###  Potential Energy Function:  ###
    
###  Constants:  ###

# alu
equilibriumLength = 1.0 / 0.529177

# radians
equilibriumAngle = 112.0 * np.pi/180.0

# aeu/alu^2
kl = 1059.162 * (0.529)**2 * (4.184/2625.5)

# aeu/rad^2
ka = 75.90*(4.184/2625.5)

def coulombic(walkers):

    qO = -0.84
    qH = 0.42

    col = np.zeros(walkers.nWalkers)
    for srcmlclidx in range(walkers.nMolecules-1):

        srcptclidxHH, destptclidxHH = np.meshgrid(walkers.ptclidx['H2O H'], walkers.ptclidx['H2O H'])
        srcptclidxHH = srcptclidxHH.flatten()
        destptclidxHH = destptclidxHH.flatten()

        dHH = walkers.dists[srcmlclidx][:, 1:, srcptclidxHH, destptclidxHH]
        dHH = dHH.reshape(walkers.nWalkers, int(dHH.size/walkers.nWalkers))

        dHH[dHH == 0] = np.infty

        col += np.sum(qH*qH/dHH, axis=1)

        srcptclidxHO, destptclidxHO = np.meshgrid(walkers.ptclidx['H2O H'], walkers.ptclidx['H2O O'])
        srcptclidxHO = srcptclidxHO.flatten()
        destptclidxHO = destptclidxHO.flatten()

        dHO = walkers.dists[srcmlclidx][:, 1:, srcptclidxHO, destptclidxHO]
        dHO = dHO.reshape(walkers.nWalkers, int(dHO.size/walkers.nWalkers))

        dHO[dHO == 0] = np.infty

        col += np.sum(qH*qO/dHO, axis=1)

        srcptclidxOH, destptclidxOH = np.meshgrid(walkers.ptclidx['H2O O'], walkers.ptclidx['H2O H'])
        srcptclidxOH = srcptclidxOH.flatten()
        destptclidxOH = destptclidxOH.flatten()

        dOH = walkers.dists[srcmlclidx][:, 1:, srcptclidxOH, destptclidxOH]
        dOH = dOH.reshape(walkers.nWalkers, int(dOH.size/walkers.nWalkers))

        dOH[dOH == 0] = np.infty
        col += np.sum(qO*qH/dOH, axis=1)

        srcptclidxOO, destptclidxOO = np.meshgrid(walkers.ptclidx['H2O O'], walkers.ptclidx['H2O O'])
        srcptclidxOO = srcptclidxOO.flatten()
        destptclidxOO = destptclidxOO.flatten()

        dOO = walkers.dists[srcmlclidx][:, 1:, srcptclidxOO, destptclidxOO]
        dOO = dOO.reshape(walkers.nWalkers, int(dOO.size/walkers.nWalkers))

        dOO[dOO == 0] = np.infty
        col += np.sum(qO*qO/dOO, axis=1)

    return col

def lennard_jones(walkers):
    epsilon = 0.1554252*(4.184/2625.5)
    sigma = 3.165492/0.529177

    lj = np.zeros(walkers.nWalkers)
    for srcmlclidx in range(walkers.nMolecules-1):

        srcptclidxOO, destptclidxOO = np.meshgrid(walkers.ptclidx['H2O O'], walkers.ptclidx['H2O O'])
        srcptclidxOO = srcptclidxOO.flatten()
        destptclidxOO = destptclidxOO.flatten()

        dOO = walkers.dists[srcmlclidx][:, 1:, srcptclidxOO, destptclidxOO]
        dOO = dOO.reshape(walkers.nWalkers, int(dOO.size/walkers.nWalkers))

        dOO[dOO == 0] = np.infty

        lj += np.sum(4*epsilon*((sigma/dOO)**12 - (sigma/dOO)**6), axis=1)

    return lj


def intra_potential(walkers):
    intraEnergy = np.zeros(walkers.nWalkers)

    for watermlcl in range(1, walkers.molecules2counts['H2O']+1):
        bd1 = walkers.get_vector('H2O ' + str(watermlcl), 'H2O ' + str(watermlcl), 'H2O O 1', 'H2O H 1')
        bd2 = walkers.get_vector('H2O ' + str(watermlcl), 'H2O ' + str(watermlcl), 'H2O O 1', 'H2O H 2')

        distances1 = np.linalg.norm(bd1, axis=1)
        distances2 = np.linalg.norm(bd2, axis=1)

        potLen1 = 1/2 * kl * (distances1 - equilibriumLength)**2
        potLen2 = 1/2 * kl * (distances2 - equilibriumLength)**2

        angles = bond_angles(bd1, bd2, distances1, distances2)

        potAngle = 1/2 * ka * (angles - equilibriumAngle)**2

        intraEnergy += potLen1 + potLen2 + potAngle

    return intraEnergy

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

    walkers.update_vecs()
    walkers.update_dists()
    pot = intra_potential(walkers) + inter_potential(walkers)

    print(f'potential energies: {pot}')
    print(f'walkers: {walkers.nWalkers}')

    return pot

def diffusionFunction(walkers):
    '''diffuse (i.e., modify the position) of all the walkers (ideally in a vectorized way).
    walkers: Walkers object. see the walkers interface above.
    '''
    arr = walkers.to_arr()

    H_shape = arr[:, walkers.mlclidx['H2O'], walkers.ptclidx['H2O H'], :].shape
    O_shape = arr[:, walkers.mlclidx['H2O'], walkers.ptclidx['H2O O'], :].shape

    H_arr = arr[:, walkers.mlclidx['H2O'], walkers.ptclidx['H2O H'], :]
    O_arr = arr[:, walkers.mlclidx['H2O'], walkers.ptclidx['H2O O'], :]

    H_arr += np.random.normal(0.0, np.sqrt(timeStep/walkers.atomic_mass('H')), H_shape)

    O_arr += np.random.normal(0.0, np.sqrt(timeStep/walkers.atomic_mass('O')), O_shape)
    walkers.update_vecs()
    walkers.update_dists()

    print(O_arr.shape)

    # while(np.any(walkers.dists[walkers.molecule('H2O 1')][:, ]))
    
    arr[:, walkers.mlclidx['H2O'], walkers.ptclidx['H2O O'], :] += np.random.normal(0.0, np.sqrt(timeStep/walkers.atomic_mass('O')), O_shape)

    walkers.set_arr(arr)

def posFunction(count):
    '''generate an ndarray representation of `count` walkers, with positions (i.e., the ndarray's data) initialized in some way.
    count: int. the number of walkers for which to generate positions.
    
    returns: ndarray. shape=(count, 3, nParticles). an ndarray representation of `count` walkers structured as described in 
    the walkers interface blurb above.
    '''
    pos = np.random.rand(count, 2, 3, 3)
    return pos

###  END SIMULATOR EDIT ZONE  ###

if __name__ == '__main__':
    walkers = Walkers(particles2mass, molecules2counts)

    sim = DMC()
    sim.setParams(timeStep, nTimeStepsEq, nTimeStepsProd, nWalkers, potentialEnergyFunction, diffusionFunction, posFunction, walkers)
    sim.run()
    sim.visualize_ref_vals()
    sim.visualize_walker_population()

    print(f'simulation converges approx. to {sim.mean_ref_val()}')