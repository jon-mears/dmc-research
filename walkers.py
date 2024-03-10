"""walker.py: Pass."""

import numpy as np

class Walkers:
    '''assumes that all walkers have identical sets of particles with no variation.
       common data is contained in this class. data unique to an individual walker 
       is contained within the respective walker object.
    '''
    def __init__(self, init_cap=10000):
        self.arr = None

        self.particles = [] # particle names
        self.particles2mass = dict() # g/mole
        self.particles2atomic = dict() # amu
        
        self.walkers_idx = []
        self.nWalkers = 0
        self.cap = init_cap

        self.next_idx = 0

    def register_particle(self, name, mass):
        self.particles.append(name)
        self.particles2mass[name] = mass

        self.particles2atomic[name] = mass/(6.02213670000e23*9.10938970000e-28)

    def to_arr(self):
        if self.arr is None:
            return
        
        return self.arr[self.walkers_idx, :, :]
    
    def set_arr(self, new_arr):
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
        if self.arr is None:
            self.arr = np.zeros((self.cap, 3, len(self.particles)))

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

    def reduced_mass(self):
        prod = 1
        sum = 0
        
        for ptcl in self.particles:
            prod *= self.particles2atomic[ptcl]
            sum += self.particles2atomic[ptcl]

        return prod/sum