import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist  # Import ROS message types
import cv2

# Particle Swarm Optimization (PSO) Code
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

    return global_best_position, global_best_value            

# Boid Class
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

# APF algorithm
def attractive_potential(current_position, goal_position, alpha=1.0):
    return alpha * (goal_position - current_position)

def repulsive_potential(current_position, obstacles, beta=100.0, rho_0=5.0):
    repulsive_force = np.zeros(2)
    for obstacle in obstacles:
        obstacle_position = np.array(obstacle)
        distance = np.linalg.norm(current_position - obstacle_position)
        if distance < rho_0:
            repulsive_force += beta * (1.0 / distance - 1.0 / rho_0) * (current_position - obstacle_position) / (distance**3)
    return repulsive_force

def artificial_potential_field(start, goal, obstacles, grid_size, alpha=1.0, beta=100.0, rho_0=5.0, gamma=0.1, max_iterations=1000):
    path = [np.array(start)]
    current_position = np.array(start)

    for _ in range(max_iterations):
        attractive_force = attractive_potential(current_position, np.array(goal), alpha)
        repulsive_force = repulsive_potential(current_position, obstacles, beta, rho_0)
        total_force = attractive_force + repulsive_force

        new_position = current_position + gamma * total_force
        new_position = np.clip(new_position, [0, 0], [grid_size[0]-1, grid_size[1]-1])

        path.append(new_position)
        current_position = new_position

        if np.linalg.norm(current_position - np.array(goal)) < 1:
            break

    return np.array(path)

class DroneSwarmNode(Node):

    def __init__(self):
        super().__init__('drone_swarm_node')
        self.drone_command_pub = self.create_publisher(PoseStamped, 'drone_command', 10)
        self.timer = self.create_timer(1.0, self.run)

    def run(self):
        # Pattern generation (if applicable)
        image_path = 'path_to_image.jpg'
        pattern_coordinates = self.generate_pattern(image_path)

        # Algorithm selection and execution
        algorithm = 'boids'  # Choose 'pso', 'boids', or 'apf'
        if algorithm == 'pso':
            bounds = np.array([[-10, 10], [-10, 10]])
            best_position, best_value = pso_optimize(bounds, num_particles=30, iterations=100)
            drone_positions = best_position
        elif algorithm == 'boids':
            width, height = 800, 600
            num_boids = 10  # Number of drones
            boids = [Boid(position=(np.random.rand()*width, np.random.rand()*height),
                          velocity=(np.random.rand()*2-1, np.random.rand()*2-1)) for _ in range(num_boids)]
            for _ in range(100):  # Run boids simulation for 100 iterations
                for boid in boids:
                    boid.flock(boids)
                    boid.update()
            drone_positions = np.array([boid.position for boid in boids])
        elif algorithm == 'apf':
            start = [0, 0]
            goal = [50, 50]
            obstacles = [[20, 20], [30, 30], [40, 40]]
            grid_size = [100, 100]
            path = artificial_potential_field(start, goal, obstacles, grid_size)
            drone_positions = path[-1]

        # Publish drone positions
        for position in drone_positions:
            msg = PoseStamped()
            msg.pose.position.x = float(position[0])
            msg.pose.position.y = float(position[1])
            msg.pose.position.z = 0.0  # Assuming 2D plane, set to a fixed value or use another variable if 3D
            self.drone_command_pub.publish(msg)
        
    def generate_pattern(self, image_path):
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Blob detection parameters (tune these as needed)
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 10000

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)

        # Extract coordinates of the keypoints
        coordinates = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        return coordinates

if __name__ == '__main__':
    rclpy.init()
    node = DroneSwarmNode()
    rclpy.spin(node)
    rclpy.shutdown()