# Drone Positions

# This script processes an image to determine optimal drone positions for a drone show display.

# The algorithm first detects the edges within the image, subsequently dividing it into 10x10 pixel blocks.
# If a block contains more than 8 white pixels, the center of that block is designated as a drone position.
# To prevent drones from being positioned too closely to one another, adjacent blocks are marked to be excluded from further consideration.
# The number of required drone positions is printed to the terminal.

import cv2 as cv
import numpy as np

# Load the image
img = cv.imread('Images/logo175_.jpg')
cv.imshow('logo', img)

# Perform edge detection using the Canny method
edges = cv.Canny(img, 100, 200)

# Obtain the dimensions of the edge-detected image
height, width = edges.shape

# Define block size and initialize variables for drone coordinates and block validation
block_size = 10
drone_coordinates = []
check = np.ones((height // block_size + 1, width // block_size + 1))

# Iterate through the image in 10x10 pixel blocks
for x in range(0, height, block_size):
    for y in range(0, width, block_size):
        block_x, block_y = x // block_size, y // block_size
        if check[block_x, block_y]:
            # Check if the block contains more than 8 edge pixels
            block = edges[x:x+block_size, y:y+block_size]
            if np.sum(block == 255) > 8:
                # Determine the center of the block for drone placement
                center_x = x + block_size // 2
                center_y = y + block_size // 2
                drone_coordinates.append((center_y, center_x))
                # Mark adjacent blocks not to be checked to prevent close drone placement
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = block_x + dx, block_y + dy
                        if 0 <= nx < check.shape[0] and 0 <= ny < check.shape[1]:
                            check[nx, ny] = 0

print(f"Number of drone positions: {len(drone_coordinates)}")

# Visualize drone positions on the output image
output_img = np.zeros((height, width), np.uint8)
for coord in drone_coordinates:
    cv.circle(output_img, coord, block_size // 2, 255, -1)  # Mark the drone positions

cv.imshow('Drone Positions', output_img)
cv.waitKey(0)
cv.destroyAllWindows()