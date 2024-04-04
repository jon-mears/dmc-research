"""walker.py: Pass."""

import particle as p

class _walker:
    def __init__(self, particles):
        self.particles = dict()
        for particle in particles.keys():
            self.particles[particle] = p.particle(particles[particle])

def walker(particles):
    return lambda: _walker(particles)