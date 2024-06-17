import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, bounds):
        self.position = np.random.rand(2) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        self.velocity = np.random.rand(2) * 2 - 1
        self.best_position = self.position.copy()
        self.best_value = np.inf

    def update(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        r1, r2 = np.random.rand(2), np.random.rand(2)
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity
        self.position += self.velocity

def objective_function(position):
    return np.sum(position**2)  # Simple quadratic function

def pso_optimize(bounds, num_particles, iterations):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = np.random.rand(2) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    global_best_value = np.inf

    for _ in range(iterations):
        for particle in particles:
            value = objective_function(particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = particle.position.copy()
            if value < global_best_value:
                global_best_value = value
                global_best_position = particle.position.copy()

        for particle in particles:
            particle.update(global_best_position)

        plt.clf()
        for particle in particles:
            plt.plot(particle.position[0], particle.position[1], 'bo')
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
        plt.pause(0.1)

    return global_best_position, global_best_value

# Define bounds for the search space
bounds = np.array([[-10, 10], [-10, 10]])
# Run PSO
best_position, best_value = pso_optimize(bounds, num_particles=30, iterations=100)
print('Best Position:', best_position)
print('Best Value:', best_value)



