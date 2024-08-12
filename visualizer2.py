import os
import sys
import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy.app import Timer
import matplotlib.pyplot as plt

# Default values
DEFAULT_NUM_BODIES = 44400
DEFAULT_GALAXY_RADIUS = 150.0e11

def load_data(file_path='nbody.out', num_bodies=DEFAULT_NUM_BODIES):
    """Load and reshape the simulation data."""
    print(f"Checking if {file_path} exists in {os.getcwd()}")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None

    with open(file_path, 'r') as file:
        lines = file.readlines()

    if not lines:
        print("Error: No data found in the file.")
        return None

    data = []
    for line in lines:
        values = list(map(float, line.split()))
        if len(values) == 4:
            data.append(values)
        else:
            print(f"Skipping invalid line: {line.strip()}")

    data = np.array(data, dtype='float64')
    total_elements = data.size
    time_steps = total_elements // (num_bodies * 4)
    data = data.reshape((time_steps, num_bodies, 4))

    print(f"Number of bodies: {num_bodies}")
    
    return data

def scale_masses(masses):
    """Scale the masses logarithmically for visualization."""
    return np.log1p(masses / 1e33) * 5

def calculate_axis_limits(galaxy_radius):
    """Calculate the axis limits based on galaxy configuration."""
    galaxy_center1 = np.array([0.0, 0.0, 0.0])
    galaxy_center2 = np.array([6*galaxy_radius, 0.0, 0.0])
    padding = 1.0e12

    min_x = min(galaxy_center1[0] - galaxy_radius, galaxy_center2[0] - galaxy_radius) - padding
    max_x = max(galaxy_center1[0] + galaxy_radius, galaxy_center2[0] + galaxy_radius) + padding
    min_y = min(galaxy_center1[1] - galaxy_radius, galaxy_center2[1] - galaxy_radius) - padding
    max_y = max(galaxy_center1[1] + galaxy_radius, galaxy_center2[1] + galaxy_radius) + padding
    min_z = min(galaxy_center1[2] - galaxy_radius, galaxy_center2[2] - galaxy_radius) - padding
    max_z = max(galaxy_center1[2] + galaxy_radius, galaxy_center2[2] + galaxy_radius) + padding

    return (min_x, max_x), (min_y, max_y), (min_z, max_z)

def setup_canvas(axis_limits):
    """Setup the Vispy canvas and view."""
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.set_range(*axis_limits)

    scatter = visuals.Markers()
    view.add(scatter)
    
    return scatter, view

def update_plot(frame_number, positions, masses, scatter):
    """Update the plot for each frame with enhanced colors, sizes, and detail."""
    sizes = scale_masses(masses[frame_number])
    min_size = 12  # Further increase minimum size for visibility
    sizes = np.clip(sizes, min_size, None)

    # yellow little masses
    cmap = plt.get_cmap('plasma')
    normed_masses = (masses[frame_number] - masses[frame_number].min()) / (masses[frame_number].max() - masses[frame_number].min())

    # yellow hue
    colors = cmap(normed_masses)[:, :3]
    
    # Make smaller masses more yellow
    yellow_shift = np.array([1.0, 1.0, 0.5])  # RGB values for yellowish color
    yellow_floor = 0.3  # How much to shift towards yellow for small masses
    colors = (1 - normed_masses[:, np.newaxis]) * yellow_shift * yellow_floor + colors * normed_masses[:, np.newaxis]

    brightness_floor = 0.6  
    colors = np.clip(colors + brightness_floor, 0, 1)

    #opacity
    alpha_channel = 1.0
    colors = np.concatenate([colors, np.full((colors.shape[0], 1), alpha_channel)], axis=1)

    scatter.set_data(positions[frame_number], size=sizes, edge_color=None, face_color=colors)

def main():
    """Main function to run the visualization."""
    num_bodies = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NUM_BODIES
    galaxy_radius = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GALAXY_RADIUS

    data = load_data(num_bodies=num_bodies)
    if data is None:
        print("Simulation data file not found. Exiting...")
        return

    positions = data[:, :, :3].astype(np.float32)
    masses = data[:, :, 3].astype(np.float32)

    axis_limits = calculate_axis_limits(galaxy_radius)
    scatter, view = setup_canvas(axis_limits)

    timer = Timer(interval=0.1, iterations=positions.shape[0])

    def on_timer(event):
        frame_number = int(event.iteration)
        if frame_number < positions.shape[0]:
            update_plot(frame_number, positions, masses, scatter)
        else:
            timer.stop()

    def on_key_press(event):
        if event.key == 'R':  
            print("Replay triggered")
            timer.start()

    canvas = scatter.parent.canvas
    canvas.events.key_press.connect(on_key_press)

    timer.connect(on_timer)
    timer.start()

    vispy.app.run()

if __name__ == "__main__":
    main()
