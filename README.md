# N-body Simulation

![N-body Simulation](Untitled%20video%20-%20Made%20with%20Clipchamp.gif)

## Overview

This project implements an N-body simulation using CUDA parallel programming to efficiently compute the gravitational interactions between a large number of bodies (e.g., stars, planets, etc.). The simulation leverages the massive parallel processing power of modern GPUs to handle the computationally intensive calculations required by the N-body problem.

## The N-body Problem

The N-body problem involves predicting the individual motions of a group of celestial objects interacting with each other gravitationally. The challenge arises from the fact that every body exerts a force on every other body, leading to a complex system of differential equations that need to be solved to simulate the system's evolution over time.

In a direct N-body simulation, the force on each body is computed by summing the contributions from all other bodies. This requires `O(N^2)` calculations for each time step, making it computationally expensive as the number of bodies increases.

## CUDA Parallel Programming

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It allows developers to use the power of NVIDIA GPUs for general-purpose computing, enabling significant performance gains for parallelizable problems like the N-body simulation.

### Key Advantages of CUDA:
- **Massive Parallelism**: CUDA allows thousands of threads to run simultaneously, making it ideal for tasks that can be broken down into smaller, independent calculations.
- **Speed**: By offloading the intensive computations to the GPU, CUDA programs can achieve speedups of several orders of magnitude compared to their CPU counterparts.
- **Scalability**: CUDA programs can be scaled to work with larger datasets and more complex simulations, limited only by the hardware.

## Simulation Details

This implementation uses CUDA to compute the gravitational forces and update the positions and velocities of bodies in the simulation. The results are visualized to provide insights into the dynamic behavior of the system.

## Visualization

The simulation results can be visualized using the included visualization tools. Below is an example GIF demonstrating the simulation:

![N-body Simulation](Untitled%20video%20-%20Made%20with%20Clipchamp.gif)

## Getting Started

To get started with the simulation, clone this repository and follow the setup instructions in the provided documentation. Ensure that you have a CUDA-capable GPU and the necessary CUDA toolkit installed.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
