import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the attractive force
def attractive_potential(current_position, goal_position, alpha=1.0):
    return alpha * (goal_position - current_position)

# Function to calculate the repulsive force
def repulsive_potential(current_position, obstacles, beta=100.0, rho_0=5.0):
    repulsive_force = np.zeros(2)
    for obstacle in obstacles:
        obstacle_position = np.array(obstacle)
        distance = np.linalg.norm(current_position - obstacle_position)
        if distance < rho_0:
            repulsive_force += beta * (1.0 / distance - 1.0 / rho_0) * (current_position - obstacle_position) / (distance**3)
    return repulsive_force

# Artificial Potential Field (APF) function
def artificial_potential_field(start, goal, obstacles, grid_size, alpha=1.0, beta=100.0, rho_0=5.0, gamma=0.1, max_iterations=1000):
    path = [np.array(start)]
    current_position = np.array(start)

    for _ in range(max_iterations):
        # Calculate forces
        attractive_force = attractive_potential(current_position, np.array(goal), alpha)
        repulsive_force = repulsive_potential(current_position, obstacles, beta, rho_0)

        # Total force
        total_force = attractive_force + repulsive_force

        # Update position
        new_position = current_position + gamma * total_force
        new_position = np.clip(new_position, [0, 0], [grid_size[0]-1, grid_size[1]-1])

        path.append(new_position)
        current_position = new_position

        # Check if goal is reached
        if np.linalg.norm(current_position - np.array(goal)) < 1:
            break

    return np.array(path)

# Example usage
grid_size = (50, 50)
start = (5, 5)
goal = (45, 45)
obstacles = [(20, i) for i in range(15, 35)] + [(30, i) for i in range(15, 35)]

# Run APF
path = artificial_potential_field(start, goal, obstacles, grid_size, 1,300,10)

# Visualization
plt.figure(figsize=(10, 10))
plt.grid(True)

# Plot obstacles
for obs in obstacles:
    plt.plot(obs[1], obs[0], 'ks')  # Obstacle positions

# Plot path
path_x = path[:, 1]
path_y = path[:, 0]
plt.plot(path_x, path_y, 'r-', linewidth=2)
plt.plot(start[1], start[0], 'go', markersize=10)  # Start point
plt.plot(goal[1], goal[0], 'bo', markersize=10)    # Goal point

plt.xlim(0, grid_size[1])
plt.ylim(0, grid_size[0])
plt.gca().invert_yaxis()  # Invert Y-axis to match array indexing
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Artificial Potential Field Path Planning')
plt.show()


