import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Boid class
class Boid:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype='float64')
        self.velocity = np.array(velocity, dtype='float64')
        self.acceleration = np.zeros(2, dtype='float64')
        self.max_speed = 4
        self.max_force = 0.1

    def update(self):
        self.velocity += self.acceleration
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        self.position += self.velocity
        self.acceleration *= 0

    def apply_force(self, force):
        self.acceleration += force

    def edges(self, width, height):
        if self.position[0] > width:
            self.position[0] = 0
        elif self.position[0] < 0:
            self.position[0] = width
        if self.position[1] > height:
            self.position[1] = 0
        elif self.position[1] < 0:
            self.position[1] = height

    def flock(self, boids):
        separation = self.separate(boids)
        alignment = self.align(boids)
        cohesion = self.cohere(boids)
        self.apply_force(separation)
        self.apply_force(alignment)
        self.apply_force(cohesion)

    def separate(self, boids):
        desired_separation = 25
        steer = np.zeros(2, dtype='float64')
        total = 0
        for other in boids:
            distance = np.linalg.norm(self.position - other.position)
            if 0 < distance < desired_separation:
                diff = self.position - other.position
                diff /= distance  # Weight by distance
                steer += diff
                total += 1
        if total > 0:
            steer /= total
        if np.linalg.norm(steer) > 0:
            steer = steer / np.linalg.norm(steer) * self.max_speed - self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = steer / np.linalg.norm(steer) * self.max_force
        return steer

    def align(self, boids):
        neighbor_dist = 50
        steer = np.zeros(2, dtype='float64')
        total = 0
        for other in boids:
            if 0 < np.linalg.norm(self.position - other.position) < neighbor_dist:
                steer += other.velocity
                total += 1
        if total > 0:
            steer /= total
            steer = steer / np.linalg.norm(steer) * self.max_speed - self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = steer / np.linalg.norm(steer) * self.max_force
        return steer

    def cohere(self, boids):
        neighbor_dist = 50
        steer = np.zeros(2, dtype='float64')
        total = 0
        for other in boids:
            if 0 < np.linalg.norm(self.position - other.position) < neighbor_dist:
                steer += other.position
                total += 1
        if total > 0:
            steer /= total
            steer = steer - self.position
            if np.linalg.norm(steer) > 0:
                steer = steer / np.linalg.norm(steer) * self.max_speed - self.velocity
                if np.linalg.norm(steer) > self.max_force:
                    steer = steer / np.linalg.norm(steer) * self.max_force
        return steer

# Visualization setup
width, height = 800, 600
num_boids = 100
boids = [Boid(position=(np.random.rand()*width, np.random.rand()*height),
              velocity=(np.random.rand()*2-1, np.random.rand()*2-1)) for _ in range(num_boids)]

fig, ax = plt.subplots()
ax.set_xlim(0, width)
ax.set_ylim(0, height)
scat = ax.scatter([b.position[0] for b in boids], [b.position[1] for b in boids])

def update(frame):
    for boid in boids:
        boid.flock(boids)
        boid.update()
        boid.edges(width, height)
    scat.set_offsets([boid.position for boid in boids])
    return scat,

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)
plt.show()

