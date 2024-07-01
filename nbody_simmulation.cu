#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include <cstdio>   // For fprintf() and stderr
#include <fstream>  // For visualizer

// Define constants
const double G = 6.67430e-11; // Gravitational constant
const int NUMTRIALS = 30;
const double dt = 0.4; // Time step in seconds
const int steps = 1000; // Number of steps to simulate

// Define a structure to represent a point mass
struct PointMass {
    double mass;
    double position[3];       // 3D position: [x, y, z]
    double velocity[3];       // 3D velocity: [vx, vy, vz]
    double force[3];          // 3D force: [fx, fy, fz]
    double angularVelocity[3];// 3D angular velocity: [wx, wy, wz]
    double torque[3];         // 3D torque: [tx, ty, tz]
    double momentOfInertia;   // Moment of inertia
};

__device__ double computeDistance(const PointMass &a, const PointMass &b) {
    double dx = b.position[0] - a.position[0];
    double dy = b.position[1] - a.position[1];
    double dz = b.position[2] - a.position[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__global__ void updateForcesAndTorquesKernel(PointMass *d_masses, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_masses[i].force[0] = d_masses[i].force[1] = d_masses[i].force[2] = 0.0;
        d_masses[i].torque[0] = d_masses[i].torque[1] = d_masses[i].torque[2] = 0.0;

        for (int j = 0; j < n; ++j) {
            if (i != j) {
                double dx = d_masses[j].position[0] - d_masses[i].position[0];
                double dy = d_masses[j].position[1] - d_masses[i].position[1];
                double dz = d_masses[j].position[2] - d_masses[i].position[2];
                double distance = sqrt(dx * dx + dy * dy + dz * dz);
                if (distance == 0) continue; // Avoid division by zero

                double magnitude = G * d_masses[i].mass * d_masses[j].mass / (distance * distance);
                double force[3];
                force[0] = magnitude * dx / distance;
                force[1] = magnitude * dy / distance;
                force[2] = magnitude * dz / distance;

                atomicAdd(&d_masses[i].force[0], force[0]);
                atomicAdd(&d_masses[i].force[1], force[1]);
                atomicAdd(&d_masses[i].force[2], force[2]);

                double torque[3];
                torque[0] = d_masses[i].position[1] * force[2] - d_masses[i].position[2] * force[1];
                torque[1] = d_masses[i].position[2] * force[0] - d_masses[i].position[0] * force[2];
                torque[2] = d_masses[i].position[0] * force[1] - d_masses[i].position[1] * force[0];

                atomicAdd(&d_masses[i].torque[0], torque[0]);
                atomicAdd(&d_masses[i].torque[1], torque[1]);
                atomicAdd(&d_masses[i].torque[2], torque[2]);
            }
        }
    }
}

__global__ void updateKinematicsKernel(PointMass *d_masses, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_masses[i].velocity[0] += d_masses[i].force[0] / d_masses[i].mass * dt;
        d_masses[i].velocity[1] += d_masses[i].force[1] / d_masses[i].mass * dt;
        d_masses[i].velocity[2] += d_masses[i].force[2] / d_masses[i].mass * dt;

        d_masses[i].position[0] += d_masses[i].velocity[0] * dt;
        d_masses[i].position[1] += d_masses[i].velocity[1] * dt;
        d_masses[i].position[2] += d_masses[i].velocity[2] * dt;

        d_masses[i].angularVelocity[0] += d_masses[i].torque[0] / d_masses[i].momentOfInertia * dt;
        d_masses[i].angularVelocity[1] += d_masses[i].torque[1] / d_masses[i].momentOfInertia * dt;
        d_masses[i].angularVelocity[2] += d_masses[i].torque[2] / d_masses[i].momentOfInertia * dt;
    }
}

void setOrbitalVelocity(PointMass &mass, const PointMass &sun) {
    // Calculate the distance between the mass and the sun
    double dx = mass.position[0] - sun.position[0];
    double dy = mass.position[1] - sun.position[1];
    double dz = mass.position[2] - sun.position[2];
    double distance = sqrt(dx * dx + dy * dy + dz * dz);
    
    // Calculate the orbital velocity magnitude
    double orbitalVelocity = sqrt(G * sun.mass / distance);
    
    // Randomly choose a direction for the orbital velocity
    double angle = static_cast<double>(rand()) / RAND_MAX * 2 * M_PI;
    double sinAngle = sin(angle);
    double cosAngle = cos(angle);
    
    // Set the tangential velocity components
    if (dx != 0 || dy != 0) {
        mass.velocity[0] = -dy / distance * orbitalVelocity * cosAngle;
        mass.velocity[1] = dx / distance * orbitalVelocity * sinAngle;
        mass.velocity[2] = 0.0; // Initially set z-component to zero
    } else {
        // If the position vector is along the z-axis, adjust the velocity accordingly
        mass.velocity[0] = orbitalVelocity * cosAngle;
        mass.velocity[1] = orbitalVelocity * sinAngle;
        mass.velocity[2] = 0.0;
    }

    // Adjust the velocity to ensure it is tangential in 3D space
    double normalComponent = (mass.velocity[0] * dx + mass.velocity[1] * dy) / (dx * dx + dy * dy);
    mass.velocity[0] -= normalComponent * dx;
    mass.velocity[1] -= normalComponent * dy;
    mass.velocity[2] = orbitalVelocity * sinAngle;
}

int main() {
    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(NULL)));

    // Define the point masses
    std::vector<PointMass> masses;

    // Add a big sun mass at the center
    double sunMass = 1.989e30; // Approximate mass of the sun in kg
    PointMass sun = {sunMass, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 1.0e40};
    masses.push_back(sun);

    for (int i = 0; i < NUMTRIALS; ++i) {
        PointMass newMass;
        newMass.position[0] = static_cast<double>(std::rand()) / RAND_MAX * 5e7;
        newMass.position[1] = static_cast<double>(std::rand()) / RAND_MAX * 5e7;
        newMass.position[2] = static_cast<double>(std::rand()) / RAND_MAX * 5e7;

        newMass.velocity[0] = 0.0;
        newMass.velocity[1] = 0.0;
        newMass.velocity[2] = 0.0;

        newMass.force[0] = 0.0;
        newMass.force[1] = 0.0;
        newMass.force[2] = 0.0;

        newMass.angularVelocity[0] = static_cast<double>(std::rand()) / RAND_MAX * 1e5;
        newMass.angularVelocity[1] = static_cast<double>(std::rand()) / RAND_MAX * 1e5;
        newMass.angularVelocity[2] = static_cast<double>(std::rand()) / RAND_MAX * 1e5;

        newMass.torque[0] = 0.0;
        newMass.torque[1] = 0.0;
        newMass.torque[2] = 0.0;

        newMass.momentOfInertia = static_cast<double>(std::rand()) / RAND_MAX * 1e10;

        newMass.mass = static_cast<double>(std::rand()) / RAND_MAX * 1e25;

        setOrbitalVelocity(newMass, sun);

        masses.push_back(newMass);
    }

    // Allocate memory on the GPU
    PointMass *d_masses;
    cudaMalloc((void**)&d_masses, masses.size() * sizeof(PointMass));
    cudaMemcpy(d_masses, masses.data(), masses.size() * sizeof(PointMass), cudaMemcpyHostToDevice);

    int blockSize = 256; // Number of threads per block
    int numBlocks = (NUMTRIALS + 1 + blockSize - 1) / blockSize; // +1 for the sun mass

    // Open a file to write the simulation data
    std::ofstream outFile("simulation_data.txt");

    // Main simulation loop
    for (int step = 0; step < steps; ++step) {
        updateForcesAndTorquesKernel<<<numBlocks, blockSize>>>(d_masses, masses.size());
        cudaDeviceSynchronize();

        updateKinematicsKernel<<<numBlocks, blockSize>>>(d_masses, masses.size(), dt);
        cudaDeviceSynchronize();

        // Copy data back to host to write positions to file
        cudaMemcpy(masses.data(), d_masses, masses.size() * sizeof(PointMass), cudaMemcpyDeviceToHost);

        printf("Step %d:\n", step);

        // Write the positions and masses of all masses at the current time step
        for (size_t i = 0; i < masses.size(); ++i) {
            printf("Mass %zu: Position (%.2e, %.2e, %.2e)\n", i, masses[i].position[0], masses[i].position[1], masses[i].position[2]);
            outFile << masses[i].position[0] << " " << masses[i].position[1] << " " << masses[i].position[2] << " " << masses[i].mass << " ";
        }
        outFile << "\n"; // End of time step
    }

    outFile.close();

    // Free GPU memory
    cudaFree(d_masses);

    return 0;
}
