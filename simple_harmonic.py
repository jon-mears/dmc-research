import numpy as np
import matplotlib.pyplot as plt


class Particle:
    def __init__(self, mass, position=None):
        self.mass = mass
        if position is None:
            self.position = np.random.normal(5.0, 0.1)
        else:
            self.position = position

class DMCSimulation:
    def __init__(self, n_walkers=1000, equilibrium_position=5.0, k=1.0, mass=1, dt=10.0, duration=100):
        self.equilibrium_position = equilibrium_position
        self.k = k
        self.dt = dt
        self.duration = duration
        self.n_walkers = n_walkers
        # Initialize particles with a specified mass and random positions
        self.particles = [Particle(mass) for _ in range(n_walkers)]
        # Tracking variables
        self.reference_energies = []
        self.walker_counts = []
        self.all_positions = []  # Collect positions for histogram analysis

    def potential_energy(self, x):
        return 0.5 * self.k * (x - self.equilibrium_position) ** 2

    def DMC_ALGORITHM(self):
        for _ in range(self.duration):
            new_particles = []
            potential_energies = []

            for particle in self.particles:
                # Simulate diffusion
                diffusion_step = np.random.normal(0, np.sqrt(self.dt / particle.mass))
                particle.position += diffusion_step
                pe = self.potential_energy(particle.position)
                potential_energies.append(pe)

            average_potential_energy = np.mean(potential_energies)

            current_population_size = len(self.particles)
            penalty = (1.0 - current_population_size / self.n_walkers) / (2.0 * self.dt)
            reference_energy = np.mean(potential_energies) + penalty
            self.reference_energies.append(reference_energy)
        


            for particle in self.particles:
                pe = self.potential_energy(particle.position)
                prob_delete = np.exp(-(pe - reference_energy) * self.dt)
                prob_replicate = np.exp(-(pe - reference_energy) * self.dt) - 1.0
                print(prob_delete, "    ",prob_replicate)
                rand_num = np.random.rand()

                if prob_delete >= rand_num:
                    new_particles.append(particle)
                    if prob_replicate > rand_num:
                        # Replicate particle with the same position and mass
                        new_particles.append(Particle(particle.mass, particle.position))
            
            self.particles = new_particles
            self.walker_counts.append(len(self.particles))
            self.all_positions.extend([p.position for p in self.particles])

    def analyze_simulation(self):

        plt.figure(figsize=(6, 4))
        plt.plot(self.reference_energies)
        plt.title('Reference Energy Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Reference Energy')
        plt.show()  

        
        plt.figure(figsize=(6, 4))
        plt.plot(self.walker_counts)
        plt.title('Walker Count Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Walkers')
        plt.show()  

        
        plt.figure(figsize=(6, 4))
        plt.hist(self.all_positions, bins=30, density=True)
        plt.title('Distribution of Walker Positions')
        plt.xlabel('Position')
        plt.ylabel('Density')
        plt.show() 




simulation = DMCSimulation()
simulation.DMC_ALGORITHM()
simulation.analyze_simulation()






