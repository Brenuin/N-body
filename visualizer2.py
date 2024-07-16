import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Load data from the simulation output file
data = np.loadtxt('simulation_data.txt')

# Print the shape of the loaded data
print("Shape of loaded data:", data.shape)

# Number of point masses (should match NUMPLANETS + NUMSTARS in your C++ code)
NUMTRIALS = 9000 # Make sure this matches the total number of particles (NUMPLANETS + NUMSTARS)

# Calculate time_steps based on data size
total_elements = data.size
time_steps = total_elements // (NUMTRIALS * 4)  # 3 for position, 1 for mass

# Print the calculated number of time steps
print("Calculated time steps:", time_steps)

# Reshape the data to (time_steps, NUMTRIALS, 4)
data = data.reshape(time_steps, NUMTRIALS, 4)

# Extract positions and masses
positions = data[:, :, :3]
masses = data[:, :, 3]

# Print the shape of positions and masses to verify correct reshaping
print("Shape of positions:", positions.shape)
print("Shape of masses:", masses.shape)

# Define fixed axis limits based on initial galaxy setup
# For example, if galaxies are positioned at (0,0,0) and (2.0e10,0,0) with radius 5.0e9
galaxy_center1 = np.array([0.0, 0.0, 0.0])
galaxy_center2 = np.array([2.0e13, 0.0, 0.0])
galaxy_radius = 40.0e14
padding = 1.0e12

# Calculate the fixed axis limits
min_x = min(galaxy_center1[0] - galaxy_radius, galaxy_center2[0] - galaxy_radius) - padding
max_x = max(galaxy_center1[0] + galaxy_radius, galaxy_center2[0] + galaxy_radius) + padding
min_y = min(galaxy_center1[1] - galaxy_radius, galaxy_center2[1] - galaxy_radius) - padding
max_y = max(galaxy_center1[1] + galaxy_radius, galaxy_center2[1] + galaxy_radius) + padding
min_z = min(galaxy_center1[2] - galaxy_radius, galaxy_center2[2] - galaxy_radius) - padding
max_z = max(galaxy_center1[2] + galaxy_radius, galaxy_center2[2] + galaxy_radius) + padding

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the scatter plot
initial_sizes = np.log1p(masses[0] / 1e25) * 15
scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], s=initial_sizes)

# Set fixed axis limits based on the calculated ranges
ax.set_xlim([min_x, max_x])
ax.set_ylim([min_y, max_y])
ax.set_zlim([min_z, max_z])

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Update function for animation
def update_plot(frame_number, positions, masses, scat):
    print(f"Updating frame {frame_number}")
    scat._offsets3d = (positions[frame_number, :, 0], positions[frame_number, :, 1], positions[frame_number, :, 2])
    new_sizes = np.log1p(masses[frame_number] / 1e33) * 15
    scat.set_sizes(new_sizes)
    return scat,

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, frames=range(0, time_steps, 10), fargs=(positions, masses, scat), interval=10, repeat=False)

# Display the animation
plt.show()
