import numpy as np

# Define the number of point masses, including the black hole
NUMTRIALS = 21

# Load data from the simulation output file
data = np.loadtxt('simulation_data.txt')

# Print the shape of the data
print("Shape of the data array:", data.shape)

# Calculate the total number of elements
num_elements = data.size
print("Total number of elements:", num_elements)

# Calculate the possible number of time steps
possible_time_steps = num_elements // (NUMTRIALS * 3)
print("Possible number of time steps:", possible_time_steps)

# Verifying the reshaped dimensions
try:
    reshaped_data = data.reshape(possible_time_steps, NUMTRIALS, 3)
    print("Reshaped data dimensions:", reshaped_data.shape)
except ValueError as e:
    print("Error in reshaping the data:", e)
