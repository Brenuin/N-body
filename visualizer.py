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

# Number of point masses (should match NUMTRIALS in your C++ code)
NUMTRIALS = 30

# Calculate time_steps based on data size
total_elements = data.size
time_steps = total_elements // (NUMTRIALS * 3)

# Reshape the data to (time_steps, NUMTRIALS, 3)
data = data.reshape(time_steps, NUMTRIALS, 3)

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the scatter plot
scat = ax.scatter(data[0, :, 0], data[0, :, 1], data[0, :, 2])

# Set axis limits (adjusted to fit the larger range of positions)
ax.set_xlim([0, 1e7])
ax.set_ylim([0, 1e7])
ax.set_zlim([0, 1e7])

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Update function for animation
def update_plot(frame_number, data, scat):
    print(f"Updating frame {frame_number}")
    scat._offsets3d = (data[frame_number, :, 0], data[frame_number, :, 1], data[frame_number, :, 2])
    return scat,

# Create the animation
ani = animation.FuncAnimation(fig, update_plot, frames=range(time_steps), fargs=(data, scat), interval=50, repeat=False)

# Display the animation
plt.show()
