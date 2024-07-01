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

# Number of point masses (should match NUMTRIALS + 1 for the sun in your C++ code)
NUMTRIALS = 152

# Calculate time_steps based on data size
total_elements = data.size
time_steps = total_elements // (NUMTRIALS * 4)  # 3 for position, 1 for mass

# Reshape the data to (time_steps, NUMTRIALS, 4)
data = data.reshape(time_steps, NUMTRIALS, 4)

# Extract positions and masses
positions = data[:, :, :3]
masses = data[:, :, 3]

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the scatter plot
initial_sizes = np.log1p(masses[0] / 1e24) * 10
scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], s=initial_sizes)

# Set axis limits (adjusted to fit the larger range of positions)
ax.set_xlim([-2.5e11, 2.5e12])
ax.set_ylim([-2.5e11, 2.5e12])
ax.set_zlim([-2.5e11, 2.5e12])
# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Update function for animation
def update_plot(frame_number, positions, masses, scat):
    print(f"Updating frame {frame_number}")
    scat._offsets3d = (positions[frame_number, :, 0], positions[frame_number, :, 1], positions[frame_number, :, 2])
    new_sizes = np.log1p(masses[frame_number] / 1e24) * 30
    scat.set_sizes(new_sizes)
    return scat,

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, frames=range(0, time_steps, 1), fargs=(positions, masses, scat), interval=10, repeat=False)

# Display the animation
plt.show()
