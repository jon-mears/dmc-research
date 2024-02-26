import numpy as np

class particle:
    def __init__(self, mass):
        self.pos = np.random.rand(1, 3)
        self.mass = mass