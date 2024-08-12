import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy import app
from ipywidgets import interact, FloatSlider, IntSlider

# Function to generate spiral position
def generateSpiralPositions(num_points, center, radius, height, numArms, b):
    positions = []
    for _ in range(num_points):
        theta = np.random.uniform(0, 4 * np.pi)  # More turns for tighter arms
        armOffset = np.random.uniform(-np.pi / (8 * numArms), np.pi / (8 * numArms))  # Allow some spread

        # Radial distance r with some randomness to avoid uniformity
        r = radius * np.sqrt(np.random.uniform(0, 1))  # r is now evenly distributed but randomized

        # Introduce some variation in b to avoid uniform spiral arms
        b_varied = b * np.random.uniform(0.8, 1.2)  # Slightly randomize the winding factor

        # Calculate x and y positions with a bit of randomness in tightness
        x = r * np.cos(theta + armOffset) * np.exp(b_varied * theta / (2 * np.pi))
        y = r * np.sin(theta + armOffset) * np.exp(b_varied * theta / (2 * np.pi))

        # Normalize the spiral to ensure it stays within the radius
        distance = np.sqrt(x**2 + y**2)
        if distance > radius:
            x = (x / distance) * radius
            y = (y / distance) * radius

        # More randomized perturbation to avoid regular wheel-like appearance
        perturbation = 0.04 * radius * np.random.uniform(-1, 1) * np.sin(numArms * theta + armOffset)

        x += perturbation
        y += perturbation
        z = np.random.uniform(-height, height)  # Add thickness to the galaxy

        positions.append([center[0] + x, center[1] + y, center[2] + z])
    
    return np.array(positions)

# Initialize the Vispy canvas
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'  # Set up a 3D camera

# Create the scatter plot
scatter = visuals.Markers()
view.add(scatter)

# Default parameter values
num_points = 10000
center = [0, 0, 0]
radius = 150.0
height = 20.0
numArms = 3
b = 0.3

# Generate initial points
positions = generateSpiralPositions(num_points, center, radius, height, numArms, b)
scatter.set_data(positions, face_color=(1, 1, 1, 0.8), size=2)

# Function to update the galaxy based on slider changes
def update_galaxy(numArms, b, radius, height):
    positions = generateSpiralPositions(num_points, center, radius, height, numArms, b)
    scatter.set_data(positions, face_color=(1, 1, 1, 0.8), size=2)
    canvas.update()

# Use ipywidgets for sliders
interact(update_galaxy,
         numArms=IntSlider(min=1, max=10, step=1, value=numArms),
         b=FloatSlider(min=0.1, max=3.0, step=0.1, value=b),
         radius=FloatSlider(min=50.0, max=300.0, step=10.0, value=radius),
         height=FloatSlider(min=1.0, max=50.0, step=1.0, value=height)
        );

# Run the application
app.run()

