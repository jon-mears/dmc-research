'''
Godfred Awuku and Jonathan Mears

DMC_am_TIP4_dimer.py
CS446 -- DMC Research
Spring 2024
----------------------------------------------------------------------------------------------
Enter `DMC_am_TIP4_dimer_1.py` into the terminal to run DMC on the system of the water dimer
with the TIP4 model of water, and the first specified set of parameters. 

Parameters:
    timeStep:          10
    nTimeSteps:        10,000
    nTrials:           100
    nInitialWalkers:   1,000
    seed:              1
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil

class Walkers:
    def __init__(self, particles2mass, molecules2counts, rigid_or_flexible, std_mlcl=None, init_cap=10000):
        '''
        Walkers constructor.

        particles2mass: dictionary: string -> float. A dictionary that maps particle names to their mass in g/mole.
        molecules2counts: dictionary: string -> int. A dictionary that maps molecule names to their number of occurrences in the target system.
        rigid_or_flexible: string. Should take on the value 'rigid' or 'flexible'. Denotes whether the bonds in the molecular system under 
            investigation are rigid or flexible.
        init_cap: int. The initial capacity of the internal `arr` ndarray in walker count.
        std_mlcl: ndarray or None. shape=(nParticles, 4). The "standard" configuration + homogeneous coordinate of a molecule in a system 
            that consists of only one type of molecule, for use in `rigid_arr_to_std_arr`. Does not need to be set in a flexible system.
        '''

        # string. Should be either 'rigid' or 'flexible'. Denotes wheter the bonds in the molecular system under investigation are rigid
        # or flexible.
        self.rigid_or_flexible = rigid_or_flexible

        # ndarray. shape=(nParticles, 4). The "standard" configuration + homogeneous coordinate of a molecule in a system
        # that consists of only type of molecule, for use in `rigid_arr_to_std_arr`. Does not need to be set in a flexible system. 
        self.std_mlcl = std_mlcl

        # dictionary. string -> float. Maps particle names to mass in g/mole.
        self.particles2mass = particles2mass

        # list[string]. All names of particles in the system.
        self.particles = list(particles2mass.keys())

        # dictionary. string -> int. Maps molecule names to their number of occurrences in the system.
        self.molecules2counts = molecules2counts

        # list[string]. All names of molecules in the system.
        self.molecules = list(molecules2counts.keys())

        # int. Largest number of particles contained within a single molecule of the target system.
        self.nParticles = None

        # int. Total number of molecules in the target system.
        self.nMolecules = None

        # dictionary. string -> list[int]. Maps '<molecule-name> <particle-name>' to the appropriate indices within ndarray `arr`.
        self.ptclidx = dict()

        # dictionary. string -> list[int]. Maps '<molecule-name>' to the appropriate indices within ndarray `arr`.
        self.mlclidx = dict()

        # dictionary. string -> float. Maps particle names to mass in amu.
        self.particles2atomic = dict()

        # perform setup related to the system's particles
        self.setup_particles()

        # perform setup related to the system's molecules
        self.setup_molecules()

        # float. The reduced mass of the system in amu.
        self.rmass = self.compute_reduced_mass()

        # list[ndarray]. Stores the displacement vector between every pair of particles in the system.
        # interface: self.vecs[source molecule][walker(s), destination molecule, source particle, destination particle, cartesian coordinate(s)].
        # Only stores the displacement vectors from a molecule to molecules stored at larger (and equal) indices. Avoids a repetition of all displacement vectors.
        self.vecs = None

        # list[ndarray]. Stores the distance between every pair of particles in the system.
        # interface: self.dists[source molecule][walker(s), destination molecule, source particle, destination particle, cartesian coordinate(s)].
        # Only stores the distances from a molecule to molecules stored at larger (and equal) indices. Avoids a repetition of all distances.
        self.dists = None
        
        # list[int]. Indices of current active walkers within the internal ndarray `arr`.
        self.walkers_idx = []

        # int. Number of current active walkers.
        self.nWalkers = 0

        # int. Number of walkers that can currently be fit in the internal ndarray `arr`.
        self.cap = init_cap

        # int. The next index at which to place a new walker.
        self.next_idx = 0

        # ndarray. Stores the configurations of all the walkers of the system. Structure depends on whether the underlying 
        # system is rigid or flexible.
        if self.rigid_or_flexible == 'flexible':

            # interface: self.arr[walker(s), molecule(s), particle(s), cartesian coordinate(s)]
            self.arr = np.zeros((self.cap, self.nMolecules, self.nParticles, 3))
        elif self.rigid_or_flexible == 'rigid':

            # interface: self.arr[walker(s), molecule(s), 3-dim position vector && 3-dim rotation vector]
            self.arr = np.zeros((self.cap, self.nMolecules, 6))

    def setup_particles(self):
        '''Setup the particles of the system by computing the atomic mass (amu) of each.'''

        # compute the atomic mass of each particle, and create an entry for the particle in the amu dictionary 
        for ptcl in self.particles:
            self.particles2atomic[ptcl] = self.particles2mass[ptcl]/(6.02213670000e23*9.10938970000e-28)

    def setup_molecules(self):
        '''
        Setup several aspects of the molecules of the system: the indices of `mlclidx`, the indices of `ptclidx`, the value
        of `nMolecules`, and the value of `nParticles`.
        '''

        # initialized to count the total number of molecules of the system
        self.nMolecules = 0

        # the first index that should be associated to the current molecule in `mlclidx`
        m_idx = 0

        # the largest number of particles encountered within a single molecule
        max_ptcls = 0

        # for each molecule...
        for mlcl in self.molecules:
            # initialized to count the total number of particles within the current molecule
            total_ptcls = 0

            # the first index that should be associated to the current particle within the current molecule in `ptclidx`
            p_idx = 0

            # index within the molecule name — will point to the start of the current particle substring
            chi = 0

            # iterate through the particle substrings of the current molecule
            while chi != len(mlcl):

                # index within the molecule name — will point one past the end of the current particle substring
                chj = chi + 1

                # index within the molecule name — will point to the start of numeric data within the current particle substring, or
                # will be None-valued if the current particle substring does not contain numeric data
                chk = None 

                # iterate `chj` to one character past the end of the current particle substring
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
                self.ptclidx[f'{mlcl} {ptcl_name}'] = list(range(p_idx, p_idx+ptcl_count))

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

    def molecule(self, key):
        '''
        Given a key of the form '{molecule name} {1-indexed idx}', returns the index of that molecule in `arr`.
        
        key: string. Key of the form '{molecule name} {1-indexed idx}'. e.g., 'H2O 2' returns the index of the "second"
        H2O molecule in the system.
        
        return: int. Index of the specified molecule in the "molecule axis" of `arr`.
        '''

        # split the key into its constituent parts
        info = key.split()

        # extract the name of the molecule from the key
        name = info[0]

        # extract the index of the molecule from the key, and make it 0-indexed
        idx = int(info[1])-1

        # return the requested molecule's index within `arr`
        return self.mlclidx[name][idx]
    
    def particle(self, key):
        '''
        Given a key of the form '{molecule name} {particle name} {1-indexed idx}', returns the index of that particle in `arr`.
        
        key: string. Key of the form '{molecule name} {particle name} {1-indexed idx}'. e.g., 'H2O H 2' returns the index of the 
        "second" H atom in the H2O molecules of the system.

        return: int. Index of the specified particle in the "particle axis" of `arr`.
        '''

        # split the key into its constituent parts
        info = key.split()

        # extract the name of the molecule from the key
        mlcl = info[0]

        # extract the name of the particle from the key
        ptcl = info[1]

        # extract the index of the particle from the key, and make it 0-indexed
        idx = int(info[2])-1

        # return the requested particle's index within `arr`
        return self.ptclidx[f'{mlcl} {ptcl}'][idx]

    def to_arr(self):
        '''Return the current "active walkers" stored in `arr`.'''
        return self.arr[self.walkers_idx]
    
    def update_arr(self, updated_arr):
        '''
        Update the internal ndarray `arr`. That is, individually update each of the already active walkers of `arr`.
        
        updated_arr: ndarray. flexible: shape=(nActiveWalkers, nMolecules, nParticles, nCartesianCoordinates) / 
            rigid: shape=(nActiveWalkers, nMolecules, 2*nCartesianCoordinates). An array containing the updated 
            positions of all active walkers in the simulation.
        '''
        
        # update each active walker in the simulation based on `updated_arr`
        self.arr[self.walkers_idx] = updated_arr
    
    def set_arr(self, new_arr):
        '''
        Assign to the internal ndarray `arr`. 

        new_arr: ndarray. flexible: shape=(nWalkers, nMolecules, nParticles, nCartesianCoordinates) / 
            rigid: shape=(nWalkers, nMolecules, 2*nCartesianCoordinates). An array containing any number of
            completely new configurations of the target system.
        '''

        # the new number of walkers in the system is the number of walkers in `new_arr`
        self.nWalkers = new_arr.shape[0]

        # generate a new linear list of indices for the active walkers in the simulation
        self.walkers_idx = range(self.nWalkers)

        # next index to fill is the index after the one already filled
        self.next_idx = self.nWalkers

        # reallocate `arr` if it is not already large enough to contain all the walkers of `new_arr`
        if self.nWalkers > self.cap:
            self.realloc(2*self.nWalkers)

        # store the walkers of `new_arr` in `arr`
        self.arr[self.walkers_idx] = new_arr
    
    def realloc(self, new_cap=None):
        '''
        Reallocate `arr` so that it can contain at most `new_cap` walkers. 

        new_cap: int or None. The new capacity of `arr` in walker count. No-op if `new_cap` is 
        less than the current number of walkers, in which case the reallocation would be destructive.
        If None, `arr` is reallocated so that it can contain at most twice the current number of walkers.
        '''
        # set the new capacity to twice the current number of walkers if `new_cap` was not specified
        if new_cap is None:
            new_cap = 2*self.nWalkers

        # no-op if `new_cap` is less than the current number of walkers
        if new_cap < self.nWalkers:
            return
        
        # set the new capacity
        self.cap = new_cap
        
        # initialize the newly reallocated `arr`, structured appropriately whether the underlying model is 'flexible' or 'rigid'
        if self.rigid_or_flexible == 'flexible':
            new_arr = np.zeros((self.cap, self.nMolecules, self.nParticles, 3))
        elif self.rigid_or_flexible == 'rigid':
            new_arr = np.zeros((self.cap, self.nMolecules, 6))

        # place the walkers in the newly reallocated `arr`, and set it appropriately
        new_arr[:self.nWalkers] = self.arr[self.walkers_idx]
        self.arr = new_arr

        # update indices accordingly
        self.walkers_idx = list(range(self.nWalkers))
        self.next_idx = self.nWalkers

    def make(self, count, pos):
        '''
        Add `count` new walkers, with positions generated by the function `gen`.

        count: int. Number of walkers to create and add.
        pos: func(count) -> `ndarray`. flexible: shape=(count, nMolecules, nParticles, nCartesianCoordinates).
            rigid: shape=(count, nMolecules, 2*nCartesianCoordinates). Function that generates an initialized 
            ndarray of a shape appropriate for the molecular system under investigation.
        '''

        # reallocate if there is not enough space to accomodate the new walkers
        if self.next_idx + count > self.cap:
            self.realloc((self.nWalkers+count)*2)

        # generate the new walkers
        new_walkers = pos(count)

        # add new walkers to `arr`, and extend `walkers_idx` to index the newly added walkers
        self.arr[self.next_idx:self.next_idx+count] = new_walkers
        self.walkers_idx.extend(list(range(self.next_idx, self.next_idx+count)))

        # increment internal count variables
        self.nWalkers += count
        self.next_idx += count

    def replicate(self, idx):
        '''
        Replicate the walkers at the indices `idx`.

        idx: list[int]. The indices of walkers in `arr` to replicate.
        '''

        # obtain the collection of walkers to replicate
        to_copy = self.arr[idx]

        # reallocate if the replication would exceed the current capacity
        if self.next_idx + len(idx) > self.cap:
            self.realloc()

        # add the replicated walkers to the end of the internal ndarray `arr`
        self.arr[self.next_idx:self.next_idx + len(idx)] = to_copy

        # update `walkers_idx` to include the indices of the newly replicated walkers
        self.walkers_idx.extend(list(range(self.next_idx, self.next_idx + len(idx))))

        # update the internal walker counts
        self.nWalkers += len(idx)
        self.next_idx += len(idx)
    
    def delete(self, idx):
        '''
        Delete the walkers at the indices `idx`.

        idx: list[int]. The indices of walkers in `arr` to delete. Does not contain repeats.
        '''

        # remove the indices of walkers to delete from the list of active walker indices
        self.walkers_idx = list(filter(lambda i: i not in idx, self.walkers_idx))

        # update the internal walker counts
        self.nWalkers -= len(idx)

    def mass(self, ptcl):
        '''Return the mass in g/mole of the particle `ptcl` (string).'''
        return self.particles2mass[ptcl]
    
    def atomic_mass(self, ptcl):
        '''Return the mass in amu of the particle `ptcl` (string).'''
        return self.particles2atomic[ptcl]
    
    def compute_reduced_mass(self):
        '''Compute and return the reduced mass of the system in amu.
        
        return: float. The reduced mass of the system, in amu.
        '''

        # initialize the cumulative product and sum
        product = 1
        sum = 0
        
        # compute the cumulative sum and product of all the atomic masses of the 
        # particles in the system
        for ptcl in self.particles:
            product *= self.particles2atomic[ptcl]
            sum += self.particles2atomic[ptcl]

        # return the reduced mass of the system in amu
        return product/sum

    def reduced_mass(self):
        '''Return the reduced mass of the system in amu.'''
        return self.rmass
    
    def update_vecs(self):
        '''Compute and store the displacement vectors between all particles of the system in `self.vecs`.'''

        # set local `arr` either directly to the array of configurations of active walkers in the case of a flexible
        # model, or to the array of configurations of walkers transformed to world space in the case of a rigid model
        if self.rigid_or_flexible == 'flexible':
            arr = self.to_arr()
        elif self.rigid_or_flexible == 'rigid':
            arr = rigid_arr_to_std_arr(self.to_arr(), self.std_mlcl)

        # reset `self.vecs` to be empty
        self.vecs = []

        # for each molecule instance in the system...
        for mlclidx in range(self.nMolecules):

            # append an array for all the displacement vectors within this molecule and between this molecule and molecules of larger indices
            self.vecs.append(np.zeros((self.nWalkers, self.nMolecules - mlclidx, self.nParticles, self.nParticles, 3)))

            # for each possible particle instance in the system...
            for ptclidx in range(self.nParticles):
                # compute and store the displacement vectors within this molecule and between it and subsequent molecules
                self.vecs[mlclidx][:, :, ptclidx] = arr[:, mlclidx:] - arr[:, mlclidx, ptclidx].reshape(self.nWalkers, 1, 1, 3)

    def update_dists(self):
        '''
        Compute and store the distances between all particles of the system in `self.dists`. `update_vecs` should be called prior to
        `update_dists`.
        '''

        # reset `self.dists` to be empty
        self.dists = []

        # for each molecule instance in the system...
        for mlclidx in range(self.nMolecules):

            # append an array of distances within this molecule and to molecules of larger indices, 
            # as computed from the displacement vectors of `self.vecs`
            self.dists.append(np.linalg.norm(self.vecs[mlclidx], axis=4))     

    def reset(self):
        '''Reset the `Walkers` data structure.'''

        # reset `self.vecs` and `self.dists`
        self.vecs = None
        self.dists = None

        # delete all walkers
        self.walkers_idx = []
        self.nWalkers = 0
        self.cap = 10000

        # reinitialize `self.arr` based on whether the underlying model is 'flexible' or 'rigid'
        if self.rigid_or_flexible == 'flexible':
            self.arr = np.zeros((self.cap, self.nMolecules, self.nParticles, 3))
        elif self.rigid_or_flexible == 'rigid':
            self.arr = np.zeros((self.cap, self.nMolecules, 6))
    
class DMC:
    """DMC Simulation."""

    # store strings for bug-free access to the reference energy and walker population data
    reference_energies = 'reference_energies'
    walker_populations = 'walker_populations'

    def __init__(self):
        '''DMC constructor.'''

        # float. Number of atomic time units between each timestep of the simulation.
        self.time_step = None

        # int. Number of timesteps to run the simulation.
        self.nTimeSteps = None

        # int. Number of initial walkers in the simulation.
        self.nInitialWalkers = None

        # int. Number of trials over which to run the simulation.
        self.nTrials = None

        # func(DMC) -> ndarray.  shape=(DMC.walkers.nWalkers). Computes the potential energy for each
        # active walker in the simulation.
        self.energy_func = None

        # func(DMC) -> None. Diffuses (that is, updates the positions) of each walker in the simulation.
        self.diffuse_func = None

        # func(count) -> ndarray. flexible: shape=(count, nMolecules, nParticles, nCartesianCoordinates).
        # rigid: shape=(count, nMolecules, 2*nCartesianCoordinates). Function that generates an initialized 
        # ndarray of a shape appropriate for the molecular system under investigation.
        self.pos_func = None

        # Walkers object. Data structure for maintaining the walkers of the simulation.
        self.walkers = None

        # int or None. Seed used for randomizations of the simulation. None if randomizations are not to be seeded.
        self.seed = None

        # float. Reference energy value of the simulation at the current timestep.
        self.ref_val = None

        # int. Reference energy convergence timestep of the current trial of the simulation.
        self.convergence_point = None

        # dictionary. string -> ndarray. shape=(nTimeSteps). Space wherein a user can specify specialized data to
        # store per timestep.
        self.data = dict()

        # dictionary. string -> ndarray. shape=(nTrials). Uses the same keys as `self.data`. Stores the mean of each
        # specialized user-specified data variable over all timesteps post-(reference-energy-)convergence of each trial. 
        self.mean_data = dict()

        # dictionary. string -> boolean. Uses the same keys as `self.data`. Stores, for each specialized user-specified data 
        # variable, whether all associated data should be saved to disk.
        self.save_data = dict()

        # int. Current timestep of the current trial of the simulation.
        self.iter = None

        # int. Current trial of the simulation.
        self.trial_iter = 0

        # ndarray. shape=(nTrials). Stores the reference energy convergence timestep of the simulation per trial.
        self.convergence_points = None

        # create entries in the `self.data` dictionary for the reference energies and walker populations
        self.data[self.reference_energies] = None
        self.data[self.walker_populations] = None

    def setParams(self, time_step, nTimeSteps, nInitialWalkers, nTrials, energy_func, diffuse_func, pos_func, walkers, seed=None):
        '''
        Set the simulation parameters.
        
        time_step: float. Number of atomic time units between each timestep of the simulation.
        nTimeSteps: int. Number of timesteps to run the simulation.
        nInitialWalkers: Number of initial walkers in the simulation.
        nTrials: int. Number of trials over which to run the simulation.
        energy_func: func(DMC) -> ndarray. shape=(DMC.walkers.nWalkers). Computes the potential
            energy for each active walker in the simulation.
        diffuse_func: func(DMC) -> None. Diffuses (that is, updates the positions) of each walker in the simulation.
        pos_func: func(count) -> ndarray. flexible: shape=(count, nMolecules, nParticles, nCartesianCoordinates).
            rigid: shape=(count, nMolecules, 2*nCartesianCoordinates). Function that generates an initialized ndarray of 
            a shape appropriate for the molecular system under investigation.
        walkers: Walkers object. Data structure for maintaining the walkers of the simulation.
        seed: int or None. Seed used for randomizations of the simulation. None if randomizations are not to be seeded.    
        '''

        # set all simulation parameters
        self.time_step = time_step
        self.nTimeSteps = nTimeSteps
        self.nTrials = nTrials
        self.nInitialWalkers = nInitialWalkers

        self.energy_func = energy_func
        self.diffuse_func = diffuse_func
        self.pos_func = pos_func

        self.walkers = walkers
        self.seed = seed

        # setup data vector for the mean of the reference energies over each trial
        self.mean_data[self.reference_energies] = np.zeros(self.nTrials)

        # setup data vector for the mean of the walker populations over each trial
        self.mean_data[self.walker_populations] = np.zeros(self.nTrials)

        # setup the data vector to store the convergence timesteps for each trial
        self.convergence_points = np.zeros(self.nTrials)

    def init(self):
        '''Perform initialization prior to a trial of the simulation.'''

        # reset the walkers data structure
        walkers.reset()

        # randomly initialize the appropriate number of walkers
        self.walkers.make(self.nInitialWalkers, self.pos_func)

        # setup data vector for each of the specialized user-specified data variables
        for entry in self.data:
            self.data[entry] = np.zeros(self.nTimeSteps)

    def penalty(self):
        '''Compute and return the penalty term for computation of the current reference energy.'''
        return (1.0-(self.walkers.nWalkers/self.nInitialWalkers))/(2.0*self.time_step)

    def compute_ref_val(self):
        '''Compute and return the current reference value.'''
        return np.mean(self.energy_func(self)) + self.penalty()

    def diffuse(self):
        '''Apply the diffusion function to all walkers of the simulation.'''
        self.diffuse_func(self)

    def adjust_walker_population(self):   
        '''
        Adjust the walker populations semi-stochastically, favoring walkers that have a lower potential
        energy than the current reference energy.
        '''   

        # obtain the indices of active walkers within `self.walkers.arr`  
        idx = np.array(self.walkers.walkers_idx)

        # compute the potential energies of each walker
        vals = self.energy_func(self)

        # obtain the potential energies that are greater than the current reference energy
        greater_vals = vals[vals > self.ref_val]

        # obtain the indices of walkers that have a greater potential energy than the reference energy
        greater_idx = idx[vals > self.ref_val]  

        # compute the probability that each walker with a potential energy greater than the
        # reference energy is not deleted
        prob_delete = np.exp(-(greater_vals-self.ref_val)*self.time_step)

        # stochastically determine which of the walkers with a potential energy greater than the 
        # reference energy should be deleted
        delete_rand = np.random.uniform(0.0, 1.0, (greater_vals.size))
        delete_idx = greater_idx[prob_delete < delete_rand]

        # obtain the potential energies that are less than the current reference energy
        lesser_vals = vals[vals < self.ref_val]

        # obtain the indices of walkers that have a lesser potential energy than the reference energy
        lesser_idx = idx[vals < self.ref_val]

        # compute the probability that each walker with a potential energy less than the 
        # reference energy is replicated
        prob_replicate = np.exp(-(lesser_vals-self.ref_val)*self.time_step) - 1.0

        # stochastically determine which of the walkers with a potential energy less than the 
        # reference energy should be replicated
        replicate_rand = np.random.uniform(0.0, 1.0, (lesser_vals.size))
        replicate_idx = lesser_idx[prob_replicate > replicate_rand]

        # delete and replicate the walkers appropriately
        self.walkers.delete(delete_idx.tolist())
        self.walkers.replicate(replicate_idx.tolist())

    def compute_convergence_point(self, entry, range_threshold=1e-2):
        '''
        Compute the convergence point for variable `data` of the current simulation, with an allowance of `range_threshold` range
        past convergence.

        entry: string. Name of variable for which to compute convergence.
        range_threshold: float. Maximum range allowed to exist past convergence.
        '''
        # count backwards through the timesteps...
        for step in range(self.nTimeSteps-1, -1, -1):

            # if, in our backwards traversal, we have reached a timestep for which the subsequent range is not within the 
            # allowable threshold...
            if np.max(self.data[entry][step:]) - np.min(self.data[entry][step:]) >= range_threshold:

                # if this occurs less than 250 timesteps from the end of the simulation...
                if step + 250 > self.nTimeSteps-1:

                    # print that the given data variable does not seem to converge in this trial, and so return the final timestep
                    print(f'{entry} does not seem to converge in trial #{self.trial_iter}. Returning last step of the simulation.')
                    return self.nTimeSteps-1
                
                # if convergence occurs more than 250 timesteps before the end of the simulation...
                else:

                    # return 250 timesteps past the step at which the range exited the allowable range, 
                    # to allow the simulation to "smooth out"
                    return step+250        
                
        # simulation is always within allowable range, so return the first timestep
        return 0

    def run_trial(self):
        '''Run a trial of the simulation, assuming everything has already been initialized.'''

        # for each timestep of the trial...
        for self.iter in range(self.nTimeSteps):

            # compute the reference value
            self.ref_val = self.compute_ref_val()

            # diffuse the walkers
            self.diffuse()

            # adjust the walker population accordingly
            self.adjust_walker_population()

            # record the reference energy at this timestep
            self.data[self.reference_energies][self.iter] = self.ref_val

            # record the walker population at this timestep
            self.data[self.walker_populations][self.iter] = self.walkers.nWalkers
                
    def simulation(self):
        '''Run the simulation.'''

        # store the pathname of the folder used for simulation data
        path = f'{os.path.dirname(os.path.abspath(__file__))}/sim_data'

        # delete the data of past experiments and create a new folder for simulation data
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)

        # seed the randomizations of the simulation
        np.random.seed(self.seed)

        # iterate over the trials
        while self.trial_iter < self.nTrials:

            # catch divisions by zero that occur from poor initialization conditions
            try:

                # initialize the simulation for the upcoming trial
                self.init()

                # run a trial of the simulation
                self.run_trial()

                # find the reference energy convergence point of the current trial
                self.convergence_point = self.compute_convergence_point(self.reference_energies)

                # store the reference energy convergence point of the current trial in the array of 
                # reference energy convergence points
                self.convergence_points[self.trial_iter] = self.convergence_point
                
                # for each specialized user-specified data variable...
                for entry in self.data:

                    # store the mean of the data variable for this trail post-(reference-energy-)convergence
                    self.mean_data[entry][self.trial_iter] = np.mean(self.data[entry][self.convergence_point:])

                # iterate over all data variables
                for entry in self.save_data:

                    # if the current variable is to be saved to disk...
                    if self.save_data[entry]:
                        
                        # save the data to disk
                        df = pd.DataFrame(self.data[entry])
                        df.to_csv(f'sim_data/{entry}_{self.trial_iter}')

                # increment the trial counter
                self.trial_iter += 1

            # if a division error occurs due to poor intialization conditions...
            except ZeroDivisionError:

                # print a message to let the user know a new trial is being started, and reinitialize and
                # run the trial
                print('Stochastic processes led to division by zero. Now starting a new simulation.')
                self.init()
                self.run_trial()
    
    def register_data(self, name):
        '''
        Register specialized data to save over the simulation.

        name: string. Name of specialized data to register.
        '''

        # by default, do not save data to disk
        self.save_data[name] = False

        # add an entry to the dictionary of data variables over a single trial
        self.data[name] = None

        # initialize an entry for the mean values of the data variable post-(reference-energy-)convergence per trial
        self.mean_data[name] = np.zeros(self.nTrials)

    def add_data(self, name, val):
        '''
        Add `val` to the array of data for the data variable `name` for the current timestep.

        name: string. Name of the data variable.
        val: float. Data to be added under the supplied data variable for the current timestep.
        '''
        self.data[name][self.iter] = val

    def save(self, name):
        '''Ensure that data variable `name` is saved to disk.'''
        self.save_data[name] = True

    def load(self, name):
        '''
        Load all the data of the saved data variable `name` from disk, and return it.

        name: string. Name of the data variable for which to retrieve all data over all trials.

        return: ndarray. shape=(nTrials, nTimeSteps). All the data saved under `name` over all timesteps and trials. If 
            `name` was not a saved data variable, returns an array of all 0s.
        '''

        # obtain the path to the source
        path = os.path.dirname(os.path.abspath(__file__))

        # initialize the table of data to return
        d = np.zeros((self.nTrials, self.nTimeSteps))

        # if the data was saved... (otherwise, an array of all 0s is returned)
        if self.save_data[name]:

            # for each trial...
            for trial_iter in range(self.nTrials):

                # retrieve the data stored for that trial
                d[trial_iter] = np.genfromtxt(f'{path}/sim_data/{name}_{trial_iter}', delimiter=',')[1:, 1:].reshape(self.nTimeSteps)

        # return all the data
        return d
    
    def extract_post_convergence_flat(self, loaded_data):
        '''
        Extract all the data that occurs post-reference-energy-convergence for each trial in `loaded_data`, and flatten
        it to a single ndarray.

        loaded_data: ndarray. shape=(nTrials, nTimeSteps). Data over all trials of the simulation as loaded via `load`.

        return: ndarray. Single-dimensional and sized appropriately to contain each data point that occurs after the 
            reference-energy-convergence-point of its trial.
        '''

        # obtain the number of post-convergence steps of each trial
        equilibration_steps = self.nTimeSteps - self.convergence_points

        # initialize an array to contain all the post-convergence data points
        post_convergence = np.zeros(int(np.sum(equilibration_steps)))

        # initialize a variable to keep track of the start of data of a particular trial
        idx = 0

        # for each trial...
        for trial in range(sim.nTrials):

            # extract the data that occurs post-reference-energy convergence in the current trial
            post_convergence[int(idx):int(idx+equilibration_steps[trial])] = loaded_data[trial, int(self.convergence_points[trial]):]

            # increment `idx` to be able to assign to the 
            idx += equilibration_steps[trial]

        # return the extracted data
        return post_convergence
    
    def extract_post_convergence(self, loaded_data):
        '''
        Extract all the data that occurs post-reference-energy convergence for each trial in `loaded_data`, and store it
        in a list of ndarrays, one per trial, as convergence may (and likely will) occur at different timesteps every trial.

        loaded_data: ndarray. shape=(nTrials, nTimeSteps). Data over all trials of the simulation as loaded via `load`.

        return: list[ndarray]. Each is single-dimensional and sized appropriately to contain all the data points that occur
            post-reference-energy convergence for the given trial.
        '''

        # create a list to contain the post-reference-energy convergence data points of each trial
        post_convergence = []

        # for each trial...
        for trial in range(self.nTrials):

            # extract the data points that occur in the loaded data post-reference-energy convergence for the given trial
            post_convergence.append(loaded_data[trial, int(self.convergence_points[trial]):])

        # return the energies 
        return post_convergence
    
###  UTILITY FUNCTIONS  ###

def bond_angle(bond1_vector, bond2_vector, bond1_len=None, bond2_len=None):
    '''
    Compute the bond angle that occurs between `bond1_vector` and `bond2_vector`.

    bond1_vector: ndarray. shape=(nWalkers, nCartesianCoordinates). Vector that represents the first bond.
    bond2_vector: ndarray. shape=(nWalkers, nCartesianCoordinates). Vector that represents the second bond.
    bond1_len: ndarray. shape=(nWalkers). Length of the first bond, or None. If None, the length will be computed in this function.
    bond2_len: ndarray. shape=(nWalkers). Length of the second bond, or None. If None, the length will be computed in this function.

    return: ndarray. shape=(nWalkers). The computed bond angle.
    '''

    # compute those bond lengths that have not been pre-computed
    if bond1_len is None:
        bond1_len = np.linalg.norm(bond1_vector, axis=1)
    if bond2_len is None:
        bond2_len = np.linalg.norm(bond2_vector, axis=1)

    # compute the dot product of the bond vectors
    dprod = np.sum(bond1_vector*bond2_vector, axis=1)

    # use the dot product of the bond vectors to compute the angle between them, and return it
    angle = np.arccos(dprod/(bond1_len*bond2_len))
    return angle

def rotation_matrix(axis, rad):
    '''
    Construct and return a homogeneous rotation matrix of `rad` radians about `axis`.

    axis: string. 'x', 'y', or 'z'. The axis about which to rotate.
    rad: float. Number of radians for which to perform the rotation.

    return: ndarray. shape=(4, 4). The requested homogeneous rotation matrix.
    '''

    # construct and return the requested homogeneous rotation matrix
    if axis=='x':
        return np.array([[1, 0, 0, 0], [0, np.cos(rad), -np.sin(rad), 0], [0, np.sin(rad), np.cos(rad), 0], [0, 0, 0, 1]])
    if axis=='y':
        return np.array([[np.cos(rad), 0, np.sin(rad), 0], [0, 1, 0, 0], [-np.sin(rad), 0, np.cos(rad), 0], [0, 0, 0, 1]])
    if axis=='z':
        return np.array([[np.cos(rad), -np.sin(rad), 0, 0], [np.sin(rad), np.cos(rad), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
def translation_matrix(x, y, z):
    '''
    Construct and return a homogeneous rotation matrix to translate to three-dimensional position (x, y, z).

    x: float. X-coordinate of position to which to translate.
    y: float. Y-coordinate of position to which to translate.
    z: float. Z-coordinate of position to which to translate.

    return: ndarray. shape=(4, 4). The requested homogeneous translation matrix.
    '''

    # construct and return the requested homogeneous translation matrix
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    
def rigid_arr_to_std_arr_matrix(rigid_arr):
    '''
    Given an array `rigid_arr` of the form appropriate for a rigid molecular system, computes the array needed to transform
    the "standard molecule" of the system into the system described by `rigid_arr`.

    rigid_arr: ndarray. shape=(nWalkers, nMolecules, 2*nCartesianCoordinates). The array for which a matrix is desired to transform
        it into the explicit array representation of the system.

    return: ndarray. Appropriately shaped for the @-operation that occurs in `rigid_arr_to_std_arr`. The array that 
        can be applied to the "standard molecule" to create the standard array representation of the system presented in `rigid_arr`.
    '''

    # store the `nWalkers` and `nMolecules` implied by the shape of `rigid_arr`
    nWalkers = rigid_arr.shape[0]
    nMolecules = rigid_arr.shape[1]

    # construct the homogeneous translation matrix needed per walker
    translations = np.zeros((nWalkers, nMolecules, 4, 4))
    translations[:] = np.eye(4)
    translations[:, :, :3, 3] = rigid_arr[:, :, :3]

    # construct the homogeneous rotation matrix about the x-axis needed per walker
    xrotations = np.zeros((nWalkers, nMolecules, 4, 4))
    xrad = rigid_arr[:, :, 3]
    xrotations[:] = np.eye(4)
    xrotations[:, :, 1, 1] = np.cos(xrad)
    xrotations[:, :, 1, 2] = -np.sin(xrad)
    xrotations[:, :, 2, 1] = np.sin(xrad)
    xrotations[:, :, 2, 2] = np.cos(xrad)

    # construct the homogeneous rotation matrix about the y-axis needed per walker
    yrotations = np.zeros((nWalkers, nMolecules, 4, 4))
    yrad = rigid_arr[:, :, 4]
    yrotations[:] = np.eye(4)
    yrotations[:, :, 0, 0] = np.cos(yrad)
    yrotations[:, :, 0, 2] = np.sin(yrad)
    yrotations[:, :, 2, 0] = -np.sin(yrad)
    yrotations[:, :, 2, 2] = np.cos(yrad)

    # construct the homogeneous rotation matrix about the z-axis needed per walker
    zrotations = np.zeros((nWalkers, nMolecules, 4, 4))
    zrad = rigid_arr[:, :, 5]
    zrotations[:] = np.eye(4)
    zrotations[:, :, 0, 0] = np.cos(zrad)
    zrotations[:, :, 0, 1] = -np.sin(zrad)
    zrotations[:, :, 1, 0] = np.sin(zrad)
    zrotations[:, :, 1, 1] = np.cos(zrad)

    # compute the overall transformation matrix needed to compute the standard array 
    # representation of the molecular system
    return translations @ yrotations @ xrotations @ zrotations
    
def rigid_arr_to_std_arr(rigid_arr, std_mlcl):
    '''
    Convert `rigid_arr`, of the form appropriate for a rigid molecular system, into the corresponding form appropriate 
    for a flexible system (i.e., with all positions of all particles of each molecule explicitly stored), by transforming
    `std_mlcl` as specified in `rigid_arr`.

    rigid_arr: ndarray. shape=(nWalkers, nMolecules, 2*nCartesianCoordinates). Representation of the desired system that 
        stores, for each molecule, the position- and rotation-vector.
    std_mlcl: ndarray. shape=(nParticles, nCartesianCoordinates + 1HomogeneousCoordinate).  

    return: ndarray. shape=(nWalkers, nMolecules, nParticles, nCartesianCoordinates). The system specifed by `rigid_arr` and 
        `std_mlcl` in explicit position-per-particle form.
    '''

    # obtain the matrix needed to transform `std_mlcl` into the system specified by `rigid_arr`
    matrix = rigid_arr_to_std_arr_matrix(rigid_arr)

    # compute and return the explicit position-per-particle form of the system specified by `rigid_arr` and 
    # `std_mlcl` 
    return (matrix @ std_mlcl.T)[:].transpose((0, 1, 3, 2))[:, :, :, :3]

###  BEGIN SIMULATOR EDIT ZONE  ###

# float. Number of atomic time units that pass per timestep of simulation.
timeStep = 10

# int. Number of timesteps per trial of the simulation.
nTimeSteps = 10000

# int. Number of trials of the simulation.
nTrials = 100

# int. Initial population of walkers in the system.
nInitialWalkers = 1000

# int or None. Random seed of the simulation. If None, the randomizations of the system are not seeded.
seed = 1

# dictionary: string -> float. Maps particle names to their masses in g/mole.
particles2mass = {'H': 1.007825, 'O': 15.99491461957, 'M': 0.0}

# dictionary: string -> int. Maps molecule names to their number of occurrences in the relevant
# molecular system.
molecules2counts = {'H2OM': 2}

# bohr radii
rOH = 0.9572*1.88973 

# radians
aHOH = 104.52*np.pi/180.0

# bohr radii
rOM = 0.15*1.88973

# hartrees*(bohr radii)/e^2
kc = 332.1*(1.88973*4.184)/2625.5

# oxygen charge
qO = 0.0

# hydrogen charge
qH = 0.42

# M charge
qM = -1.04

# hartrees/molecule — constant useful in Lennard-Jones computation
epsilon = 0.64895/2625.5

# bohr radii — constant useful in Lennard-Jones computation
sigma = 3.1536*1.88973

# "standard" H2O molecule configuration + a homogeneous coordinate, for use in `rigid_arr_to_std_arr`
std_H2OM = np.zeros((4, 4))

# set the homogeneous coordinate
std_H2OM[:, 3] = 1

# set the first hydrogen particle
std_H2OM[0, :3] = np.array([rOH, 0, 0])

# set the second hydrogen particle
std_H2OM[1] = rotation_matrix('z', aHOH) @ std_H2OM[0]
std_H2OM[1, :3] = rOH*(std_H2OM[1, :3]/np.linalg.norm(std_H2OM[1, :3]))

# oxygen particle is at the "origin" — does not need to be set

# set the "M" particle
std_H2OM[3] = rotation_matrix('z', aHOH/2) @ std_H2OM[0]
std_H2OM[3, :3] = rOM*(std_H2OM[3, :3]/np.linalg.norm(std_H2OM[3, :3]))

def coulombic(dmc):
    '''
    Compute the energy due to coulombic interactions per walker.

    dmc: DMC object. Simulation for which to compute the energy due to coulombic interactions.

    return: ndarray. shape=(nWalkers). Energy due to coulombic interactions per walker.
    '''

    # obtain the data structure containing the walkers of the simulation
    walkers = dmc.walkers

    # initialize an array to contain the energy due to coulombic 
    # interactions per walker
    col = np.zeros(walkers.nWalkers)

    # iterate over each molecule except the last, as we will have already seen all
    # intermolecular interactions by then
    for srcmlclidx in range(walkers.nMolecules-1):

        # obtain arrays of indices so that every combination of H-to-H is obtained
        srcptclidxHH, destptclidxHH = np.meshgrid(walkers.ptclidx['H2OM H'], walkers.ptclidx['H2OM H'])
        srcptclidxHH = srcptclidxHH.flatten()
        destptclidxHH = destptclidxHH.flatten()
        
        # obtain all intermolecular distances between hydrogen atoms, and reshape 
        # to nicely list each per walker
        dHH = walkers.dists[srcmlclidx][:, 1:, srcptclidxHH, destptclidxHH]
        dHH = dHH.reshape(walkers.nWalkers, int(dHH.size/walkers.nWalkers))

        # for zero distances, set to infinity so that the coulombic term evaluates to zero
        dHH[dHH == 0] = np.infty

        # sum all the coulombic terms between hydrogen atoms within each walker, and add to total coulombic
        col += np.sum(qH*qH/dHH, axis=1)

        # obtain arrays of indices so that every combination of H-to-M is obtained
        srcptclidxHM, destptclidxHM = np.meshgrid(walkers.ptclidx['H2OM H'], walkers.ptclidx['H2OM M'])
        srcptclidxHM = srcptclidxHM.flatten()
        destptclidxHM = destptclidxHM.flatten()
        
        # obtain all intermolecular distances between hydrogen and M particles, and reshape
        # to nicely list each per walker
        dHM = walkers.dists[srcmlclidx][:, 1:, srcptclidxHM, destptclidxHM]
        dHM = dHM.reshape(walkers.nWalkers, int(dHM.size/walkers.nWalkers))

        # for zero distances, set to infinity so that the coulombic term evaluates to zero
        dHM[dHM == 0] = np.infty

        # sum all coulombic terms between hydrogen and M particles within each walker, and add to
        # total coulombic
        col += np.sum(qH*qM/dHM, axis=1)

        # obtain arrays of indices so that every combination of M-to-H is obtained
        srcptclidxMH, destptclidxMH = np.meshgrid(walkers.ptclidx['H2OM M'], walkers.ptclidx['H2OM H'])
        srcptclidxMH = srcptclidxMH.flatten()
        destptclidxMH = destptclidxMH.flatten()
        
        # obtain all intermolecular distances between M and hydrogen particles, and reshape
        # to nicely list each per walker
        dMH = walkers.dists[srcmlclidx][:, 1:, srcptclidxMH, destptclidxMH]
        dMH = dMH.reshape(walkers.nWalkers, int(dMH.size/walkers.nWalkers))

        # for zero distances, set to infinity so that the coulombic term evaluates to zero
        dMH[dMH == 0] = np.infty

        # sum all coulombic terms between M and hydrogen particles within each walker, and add to 
        # total coulombic
        col += np.sum(qM*qH/dMH, axis=1)

    # return the energy resulting from coulombic interactions per walker
    return col

def lennard_jones(dmc):
    '''
    Compute the energy due to Lennard-Jones interactions per walker.

    dmc: DMC object. Simulation for which to compute the energy due to Lennard-Jones interactions.

    return: ndarray. shape=(nWalkers). Energy due to Lennard-Jones interactions per walker.
    '''

    # obtain the data structure containing the walkers of the simulation
    walkers = dmc.walkers

    # initialize an array to contain the energy due to Lennard-Jones
    # interactions per walker
    lj = np.zeros(walkers.nWalkers)

    # iterate over each molecule except the last, as we will have already seen all
    # intermolecular interactions by then
    for srcmlclidx in range(walkers.nMolecules-1):

        # obtain arrays of indices so that every combination of O-to-O is obtained
        srcptclidxOO, destptclidxOO = np.meshgrid(walkers.ptclidx['H2OM O'], walkers.ptclidx['H2OM O'])
        srcptclidxOO = srcptclidxOO.flatten()
        destptclidxOO = destptclidxOO.flatten()

        # obtain all intermolecular distances between oxygen atoms, and reshape to
        # nicely list each per walker
        dOO = walkers.dists[srcmlclidx][:, 1:, srcptclidxOO, destptclidxOO]
        dOO = dOO.reshape(walkers.nWalkers, int(dOO.size/walkers.nWalkers))

        # for zero distances, set to infinity so that the Lennard-Jones term evaluates to zero
        dOO[dOO == 0] = np.infty

        # sum all the Lennard-Jones terms between oxygen atoms within each walker, and add to total
        # Lennard-Jones
        lj += np.sum(4*epsilon*((sigma/dOO)**12 - (sigma/dOO)**6), axis=1)

    # return the energy resulting from Lennard-Jones interactions per walker
    return lj

def inter_potential(dmc):
    '''Compute and return the energy due to intermolecular forces per walker.'''
    return coulombic(dmc) + lennard_jones(dmc)

def energy(dmc):
    '''
    Computes the potential energy of all walkers in the simulation. 

    dmc: DMC object. Simulation for which to compute the potential energy of all walkers.
    
    return: ndarray. shape=(nWalkers). Contains the potential energy of each walker in the simulation.
    '''

    # obtain the data structure containing the walkers of the simulation
    walkers = dmc.walkers

    # update all the inter-/intra-molecular displacement vectors and distances of the system
    walkers.update_vecs()
    walkers.update_dists()

    # compute and return the potential energy of each walker of the simulation
    return inter_potential(dmc)

def diffusion(dmc):
    '''
    Diffuse (i.e., modify the position) of all the walkers in the simulation.

    dmc: DMC object. Simulation for which to diffuse all walkers.
    '''

    # obtain the data structure containing the walkers of the simulation
    walkers = dmc.walkers

    # obtain the array that contains the active walkers of the simulation
    arr = walkers.to_arr()

    # add random values of a normal distribution to the position- and rotation-vectors of the configurations
    arr[:] += np.random.normal(0.0, np.sqrt(timeStep/(2*walkers.atomic_mass('H')+walkers.atomic_mass('O'))), arr.shape)

    # update the walkers of the simulation
    walkers.update_arr(arr)

def position(count):
    '''
    Initialize `count` walkers. 

    count: int. The number of walkers to initialize.
    
    return: ndarray. flexible: shape=(count, nMolecules, nParticles, nCartesianCoordinates). 
        rigid: shape=(count, nMolecules, 2*nCartesianCoordinates). `count` initialized walkers.
    '''
    # randomly initialize the position- and rotation-vectors of a rigid representation of the system
    pos = np.zeros((count, 2, 6))
    pos[:, 0] = np.random.normal(0, 0.5, (count, 6))
    pos[:, 1] = np.random.normal(10, 0.5, (count, 6))

    return pos

###  END SIMULATOR EDIT ZONE  ###

if __name__ == '__main__':
    # initialize the Walkers data structure
    walkers = Walkers(particles2mass, molecules2counts, 'rigid', std_H2OM) 

    # initialize the DMC simulation object
    sim = DMC()
    sim.setParams(timeStep, nTimeSteps, nInitialWalkers, nTrials, energy, diffusion, position, walkers, seed)

    # save all data to disk
    sim.save(sim.walker_populations)
    sim.save(sim.reference_energies)

    # run the simulation
    sim.simulation()

    # obtain a flattened ndarray containing all the post-convergence reference energies
    all_reference_energies = sim.extract_post_convergence_flat(sim.load(sim.reference_energies))

    # compute and print the mean of all post-convergence reference energies
    print(f'Simulation Converges Approximately to {np.mean(all_reference_energies)} Hartrees\n')

    ### PLOTTING ###
    
    # walker population over last trial after convergence
    plt.hist(sim.data[sim.walker_populations][sim.convergence_point:], 100)
    plt.title(f'Distribution of Walker Populations Over the Last Trial ({os.path.basename(__file__).split(".")[0]})')
    plt.xlabel('Population')
    plt.ylabel('Frequency')
    plt.axvline(nInitialWalkers, color='red', label=f'Initial Walker Population ({nInitialWalkers} Walkers)')
    plt.legend()
    plt.show()

    # obtain a flattened ndarray containing all the post-convergence walker populations
    all_walker_populations = sim.extract_post_convergence_flat(sim.load(sim.walker_populations))

    # walker population over all trials
    plt.hist(all_walker_populations, 100)
    plt.title(f'Distribution of Walker Populations Over {nTrials} Trials ({os.path.basename(__file__).split(".")[0]})')
    plt.xlabel('Population')
    plt.ylabel('Frequency')
    plt.axvline(nInitialWalkers, color='red', label=f'Initial Walker Population ({nInitialWalkers} Walkers)')
    plt.legend()
    plt.show()

    # reference energies over entirety of last trial
    plt.plot(sim.data[sim.reference_energies])
    plt.title(f'Reference Energies Over All of Last Trial ({os.path.basename(__file__).split(".")[0]})')
    plt.xlabel('Time Step')
    plt.ylabel('Reference Energy (Hartrees)')
    plt.axvline(sim.convergence_point, color='red', label=f'Convergence Point (Timestep #{sim.convergence_point})')
    plt.legend()
    plt.show()

    print(f'Standard Deviation Post-Convergence of Last Trial: {np.std(sim.data[sim.reference_energies][sim.convergence_point:])} Hartrees')

    # max and min reference energies of last trial post-convergence
    M = np.max(sim.data[sim.reference_energies][sim.convergence_point:])
    m = np.min(sim.data[sim.reference_energies][sim.convergence_point:])

    # reference energies over last trial post-convergence
    plt.plot(np.arange(sim.convergence_point, sim.nTimeSteps), sim.data[sim.reference_energies][sim.convergence_point:])
    plt.title(f'Reference Energies of Last Trial Post-Convergence ({os.path.basename(__file__).split(".")[0]})')
    plt.xlabel('Time Step')
    plt.ylabel('Reference Energy (Hartrees)')
    plt.axhline(sim.mean_data[sim.reference_energies][sim.nTrials-1], color='red', label=f'Mean Reference Energy ({sim.mean_data[sim.reference_energies][sim.nTrials-1]:.3f} Hartrees)')
    plt.axhline(M, color='black', linestyle='dashed', label=f'Max ({M:.3f} Hartrees)')
    plt.axhline(m, color='black', linestyle='dotted', label=f'Min ({m:.3f} Hartrees)')
    plt.legend()
    plt.show()

    # range of last trial
    r = M - m
    print(f'Range of Last Trial: {r} Hartrees')

    # obtain all reference energies post-convergence, not flattened
    all_reference_energies = sim.extract_post_convergence(sim.load(sim.reference_energies))

    # list to contain the post-convergence standard deviations of each trial
    stddevs = []

    # list to contain the post-convergence ranges of each trial
    rs = []
    
    # iterate over all trials
    for trial in range(nTrials):

        # compute and append the post-convergence standard deviation and range
        stddevs.append(np.std(all_reference_energies[trial]))
        rs.append(np.max(all_reference_energies[trial]) - np.min(all_reference_energies[trial]))

    # compute the mean post-convergence standard deviation over all trials
    mean_stddev = np.mean(stddevs)
    print(f'Mean Standard Deviation Post-Convergence Over All {nTrials} Trials: {mean_stddev} Hartrees')

    # compute the mean convergence point over all trials
    mean_convergence_point = np.mean(sim.convergence_points)
    print(f'Mean Convergence Point Over All {nTrials} Trials: {mean_convergence_point} Timesteps')

    # compute the mean post-convergence range over all trials
    mean_range = np.mean(rs)
    print(f'Mean Range Over All {nTrials} Trials: {mean_range} Hartrees')