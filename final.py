import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
from coordinates import width, height, drone_coordinates


# Boid class
class Boid:
    def __init__(self, position, velocity, target):
        self.position = np.array(position, dtype="float64")
        self.velocity = np.array(velocity, dtype="float64")
        self.acceleration = np.zeros(2, dtype="float64")
        self.max_speed = 4
        self.max_force = 0.1
        self.target = np.array(target, dtype="float64")

    def update(self):
        if np.linalg.norm(self.position - self.target) > 2:
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
            self.position[0] = width
            self.velocity[0] *= -1
        elif self.position[0] < 0:
            self.position[0] = 0
            self.velocity[0] *= -1
        if self.position[1] > height:
            self.position[1] = height
            self.velocity[1] *= -1
        elif self.position[1] < 0:
            self.position[1] = 0
            self.velocity[1] *= -1

    def flock(self, boids, kdtree):
        neighbors = kdtree.query_ball_point(self.position, 50)
        neighbors_boids = [boids[i] for i in neighbors if boids[i] != self]

        separation = self.separate(neighbors_boids)
        alignment = self.align(neighbors_boids)
        cohesion = self.cohere(neighbors_boids)
        target_seek = self.seek(self.target)

        self.apply_force(separation)
        self.apply_force(alignment)
        self.apply_force(cohesion)
        self.apply_force(target_seek)

    def separate(self, boids):
        desired_separation = 25
        steer = np.zeros(2, dtype="float64")
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
            steer = (steer / np.linalg.norm(steer)) * self.max_speed - self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = steer / np.linalg.norm(steer) * self.max_force
        return steer

    def align(self, boids):
        neighbor_dist = 50
        steer = np.zeros(2, dtype="float64")
        total = 0
        for other in boids:
            if 0 < np.linalg.norm(self.position - other.position) < neighbor_dist:
                steer += other.velocity
                total += 1
        if total > 0:
            steer /= total
            steer = (steer / np.linalg.norm(steer)) * self.max_speed - self.velocity
            if np.linalg.norm(steer) > self.max_force:
                steer = steer / np.linalg.norm(steer) * self.max_force
        return steer

    def cohere(self, boids):
        neighbor_dist = 50
        steer = np.zeros(2, dtype="float64")
        total = 0
        for other in boids:
            if 0 < np.linalg.norm(self.position - other.position) < neighbor_dist:
                steer += other.position
                total += 1
        if total > 0:
            steer /= total
            steer = steer - self.position
            if np.linalg.norm(steer) > 0:
                steer = (steer / np.linalg.norm(steer)) * self.max_speed - self.velocity
                if np.linalg.norm(steer) > self.max_force:
                    steer = steer / np.linalg.norm(steer) * self.max_force
        return steer

    def seek(self, target):
        desired = target - self.position
        distance = np.linalg.norm(desired)
        if distance > 0:
            desired = (desired / distance) * self.max_speed
            steer = desired - self.velocity
            return steer
        return np.zeros(2, dtype="float64")


def create_grid_points(grid_size, num_points):
    x = np.linspace(0, grid_size[0], int(np.ceil(np.sqrt(num_points))))
    y = np.linspace(0, grid_size[1], int(np.ceil(np.sqrt(num_points))))
    xv, yv = np.meshgrid(x, y)
    points = np.vstack([xv.ravel(), yv.ravel()]).T
    return points[:num_points]


def assign_targets(boids, target_positions):
    cost_matrix = np.zeros((len(boids), len(target_positions)))
    for i, boid in enumerate(boids):
        for j, target in enumerate(target_positions):
            cost_matrix[i, j] = np.linalg.norm(boid.position - target)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i, j in zip(row_ind, col_ind):
        boids[i].target = target_positions[j]


grid_size = (width, height)  # Size of the grid
num_boids = len(drone_coordinates)  # Number of points
target_positions = drone_coordinates
initial_positions = create_grid_points(grid_size, num_boids)


# Visualization setup
def initialize_boids(num_boids):
    boids = []
    for i in range(num_boids):
        position = initial_positions[i]
        target = target_positions[i]
        velocity = target - position
        if np.linalg.norm(velocity) > 0:
            velocity = (
                velocity / np.linalg.norm(velocity)
            ) * 2  # Normalize and set speed to 2
        boid = Boid(position=position, velocity=velocity, target=target)
        boids.append(boid)
    return boids


boids = initialize_boids(num_boids)

fig, ax = plt.subplots()
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_facecolor("black")

scat = ax.scatter(
    [b.position[0] for b in boids], [b.position[1] for b in boids], c="white"
)


def update(frame):
    if frame % 10 == 0:  # Reassign targets every 10 frames
        assign_targets(boids, target_positions)

    positions = [boid.position for boid in boids]
    kdtree = KDTree(positions)

    for boid in boids:
        boid.flock(boids, kdtree)
        boid.update()
        boid.edges(width, height)

    scat.set_offsets([boid.position for boid in boids])
    return (scat,)


ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)
plt.show()
