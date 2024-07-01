#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include <cstdio>   // For fprintf() and stderr
#include <fstream>  //Visualizer

#ifndef NUMT
#define NUMT -1
#endif

#ifndef NUMTRIALS
#define NUMTRIALS 100
#endif


// Define constants
const double G = 6.67430e-11; // Gravitational constant

// Define a structure to represent a point mass
struct PointMass {
    double mass;
    std::vector<double> position; // 3D position: [x, y, z]
    std::vector<double> velocity; // 3D velocity: [vx, vy, vz]
    std::vector<double> force;    // 3D force: [fx, fy, fz]

    PointMass(double m, const std::vector<double>& p, const std::vector<double>& v, const std::vector<double>& f)
        : mass(m), position(p), velocity(v), force(f) {}
};

// Function to compute the distance between two point masses
double computeDistance(const PointMass &a, const PointMass &b) {
    double dx = b.position[0] - a.position[0];
    double dy = b.position[1] - a.position[1];
    double dz = b.position[2] - a.position[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Function to compute gravitational force between two point masses
std::vector<double> computeForce(const PointMass &a, const PointMass &b) {
    std::vector<double> force(3, 0.0);
    double distance = computeDistance(a, b);
    if (distance == 0) return force; // Avoid division by zero

    double magnitude = G * a.mass * b.mass / (distance * distance);

    force[0] = magnitude * (b.position[0] - a.position[0]) / distance;
    force[1] = magnitude * (b.position[1] - a.position[1]) / distance;
    force[2] = magnitude * (b.position[2] - a.position[2]) / distance;

    return force;
}

// Function to update forces on each point mass
void updateForces(std::vector<PointMass> &masses) {
    int n = masses.size();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        masses[i].force.assign(3, 0.0); // Reset force
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                std::vector<double> force = computeForce(masses[i], masses[j]);
                masses[i].force[0] += force[0];
                masses[i].force[1] += force[1];
                masses[i].force[2] += force[2];
            }
        }
    }
}

// Function to update positions and velocities of point masses
void updatePositionsAndVelocities(std::vector<PointMass> &masses, double dt) {
    int n = masses.size();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        masses[i].velocity[0] += masses[i].force[0] / masses[i].mass * dt;
        masses[i].velocity[1] += masses[i].force[1] / masses[i].mass * dt;
        masses[i].velocity[2] += masses[i].force[2] / masses[i].mass * dt;

        masses[i].position[0] += masses[i].velocity[0] * dt;
        masses[i].position[1] += masses[i].velocity[1] * dt;
        masses[i].position[2] += masses[i].velocity[2] * dt;
    }
}

int main() {

     int numThreads = NUMT;
    if (numThreads == -1) {
        numThreads = omp_get_max_threads();
    }
    omp_set_num_threads(numThreads);
    double time0 = omp_get_wtime();

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(NULL))); // Changed nullptr to NULL

    // Define the point masses
    std::vector<PointMass> masses;
    
    for (int i = 0; i < NUMTRIALS; ++i) {
    std::vector<double> position(3);
    position[0] = static_cast<double>(std::rand()) / RAND_MAX * 1e7;
    position[1] = static_cast<double>(std::rand()) / RAND_MAX * 1e7;
    position[2] = static_cast<double>(std::rand()) / RAND_MAX * 1e7;

    std::vector<double> velocity(3, 0.0); // 0.0 for vx, vy, vz
    std::vector<double> force(3, 0.0);    // 0.0 for fx, fy, fz

    // Increase the mass to make the interaction more noticeable
    double mass = static_cast<double>(std::rand()) / RAND_MAX * 1e30;
    masses.push_back(PointMass(mass, position, velocity, force));
}


    // Open a file to write the simulation data
    std::ofstream outFile("simulation_data.txt");

    // Simulation parameters
    double dt = .01; // Time step in seconds
    int steps = 100; // Number of steps to simulate

    // Main simulation loop
    for (int step = 0; step < steps; ++step) {
        updateForces(masses);
        updatePositionsAndVelocities(masses, dt);
        printf("Step %d:\n", step);

        // Write the positions of all masses at the current time step
        for (size_t i = 0; i < masses.size(); ++i) {
            printf("Mass %zu: Position (%.2e, %.2e, %.2e)\n", i, masses[i].position[0], masses[i].position[1], masses[i].position[2]);
            outFile << masses[i].position[0] << " " << masses[i].position[1] << " " << masses[i].position[2] << " ";
        }
        outFile << "\n"; // End of time step
    }

    outFile.close();

    double time1 = omp_get_wtime();
    double megaTrialsPerSecond = (double)NUMTRIALS / (time1 - time0) / 1000000.;
    fprintf(stderr, "%2d , %8d , %6.2lf\n", NUMT, NUMTRIALS, megaTrialsPerSecond);

    return 0;
}
