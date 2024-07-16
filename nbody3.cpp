#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()
#include <cstdio>   // For fprintf() and stderr
#include <fstream>  // For visualizer
#include <iomanip>

#ifndef NUMT
#define NUMT -1   // Number of threads to be used in the for loop
#endif

#ifndef NUMPLANETS
#define NUMPLANETS 150
#endif

#ifndef NUMSTARS
#define NUMSTARS 150
#endif

// Define constants
const double G = 6.67430e-11; // Gravitational constant

// Define a structure to represent a point mass
struct PointMass {
    double mass;
    std::vector<double> position;       // 3D position: [x, y, z]
    std::vector<double> velocity;       // 3D velocity: [vx, vy, vz]
    std::vector<double> force;          // 3D force: [fx, fy, fz]
    std::vector<double> angularVelocity;// 3D angular velocity: [wx, wy, wz]
    std::vector<double> torque;         // 3D torque: [tx, ty, tz]
    double momentOfInertia;             // Moment of inertia

    PointMass(double m, const std::vector<double>& p, const std::vector<double>& v, const std::vector<double>& f, const std::vector<double>& w, const std::vector<double>& t, double I)
        : mass(m), position(p), velocity(v), force(f), angularVelocity(w), torque(t), momentOfInertia(I) {}
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

// Function to compute torque on a point mass
std::vector<double> computeTorque(const PointMass &a, const std::vector<double> &force) {
    std::vector<double> torque(3, 0.0);
    torque[0] = a.position[1] * force[2] - a.position[2] * force[1];
    torque[1] = a.position[2] * force[0] - a.position[0] * force[2];
    torque[2] = a.position[0] * force[1] - a.position[1] * force[0];
    return torque;
}

// Function to update forces and torques on each point mass
void updateForcesAndTorques(std::vector<PointMass> &masses) {
    int n = masses.size();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        masses[i].force.assign(3, 0.0); // Reset force
        masses[i].torque.assign(3, 0.0); // Reset torque
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                std::vector<double> force = computeForce(masses[i], masses[j]);
                masses[i].force[0] += force[0];
                masses[i].force[1] += force[1];
                masses[i].force[2] += force[2];

                std::vector<double> torque = computeTorque(masses[i], force);
                masses[i].torque[0] += torque[0];
                masses[i].torque[1] += torque[1];
                masses[i].torque[2] += torque[2];
            }
        }
    }
}

// Function to update positions, velocities, and angular velocities of point masses
void updateKinematics(std::vector<PointMass> &masses, double dt) {
    int n = masses.size();
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        masses[i].velocity[0] += masses[i].force[0] / masses[i].mass * dt;
        masses[i].velocity[1] += masses[i].force[1] / masses[i].mass * dt;
        masses[i].velocity[2] += masses[i].force[2] / masses[i].mass * dt;

        masses[i].position[0] += masses[i].velocity[0] * dt;
        masses[i].position[1] += masses[i].velocity[1] * dt;
        masses[i].position[2] += masses[i].velocity[2] * dt;

        masses[i].angularVelocity[0] += masses[i].torque[0] / masses[i].momentOfInertia * dt;
        masses[i].angularVelocity[1] += masses[i].torque[1] / masses[i].momentOfInertia * dt;
        masses[i].angularVelocity[2] += masses[i].torque[2] / masses[i].momentOfInertia * dt;
    }
}

void displayLoadingBar(int step, int steps) {
    int barWidth = 70;
    float progress = (float)step / steps;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

double randomInRange(double min, double max) {
    return min + static_cast<double>(std::rand()) / RAND_MAX * (max - min);
}

void generate_galaxy(std::vector<PointMass>& masses, int numPlanets, int numStars, const std::vector<double>& center, double radius, double mass_min, double mass_max) {
    int totalBodies = numPlanets + numStars;
    double totalMass = 0.0;
    double mass;

    // Generate planets
    for (int i = 0; i < numPlanets; ++i) {
        std::vector<double> position(3);
        double distance = randomInRange(0, radius);
        double angle = randomInRange(0, 2 * M_PI);
        position[0] = center[0] + distance * std::cos(angle);
        position[1] = center[1] + distance * std::sin(angle);
        position[2] = center[2] + randomInRange(-radius, radius);

        std::vector<double> velocity(3, 0.0);
        std::vector<double> force(3, 0.0);
        std::vector<double> angularVelocity(3, 0.0);
        std::vector<double> torque(3, 0.0);
        double momentOfInertia = randomInRange(1e10, 1e20);

        mass = randomInRange(100*mass_min, 100*mass_max);
        totalMass += mass;

        PointMass newBody(mass, position, velocity, force, angularVelocity, torque, momentOfInertia);
        masses.push_back(newBody);
    }

    // Generate stars
    for (int i = 0; i < numStars; ++i) {
        std::vector<double> position(3);
        double distance = randomInRange(0, radius);
        double angle = randomInRange(0, 2 * M_PI);
        position[0] = center[0] + distance * std::cos(angle);
        position[1] = center[1] + distance * std::sin(angle);
        position[2] = center[2] + randomInRange(-radius, radius);

        std::vector<double> velocity(3, 0.0);
        std::vector<double> force(3, 0.0);
        std::vector<double> angularVelocity(3, 0.0);
        std::vector<double> torque(3, 0.0);
        double momentOfInertia = randomInRange(1e10, 1e20);

        mass = randomInRange(mass_min, mass_max);
        totalMass += mass;

        PointMass newBody(mass, position, velocity, force, angularVelocity, torque, momentOfInertia);
        masses.push_back(newBody);
    }

    double averageMass = totalMass / totalBodies;

    // Set orbital velocities for stable orbits
    for (auto& mass : masses) {
        double dx = mass.position[0] - center[0];
        double dy = mass.position[1] - center[1];
        double dz = mass.position[2] - center[2];
        double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        
        // Calculate the orbital velocity magnitude
        double orbitalVelocity = std::sqrt(G * averageMass / distance);
        
        // Randomly choose a direction for the orbital velocity
        double angle = randomInRange(0, 2 * M_PI);
        double sinAngle = std::sin(angle);
        double cosAngle = std::cos(angle);
        
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
}

int main() {
    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    srand(static_cast<unsigned int>(time(NULL)));

    double randsize_min = 1.0e24, randsize_max = 1.0e26;
    double galaxyRadius = 2.0e12;

    std::vector<PointMass> masses;
    double galaxyCenter[3] = {0.0, 0.0, 0.0};
    generate_galaxy(masses, NUMPLANETS, NUMSTARS, galaxyCenter, galaxyRadius, randsize_min, randsize_max);

    PointMass *d_masses;
    cudaMalloc(&d_masses, masses.size() * sizeof(PointMass));
    cudaMemcpy(d_masses, masses.data(), masses.size() * sizeof(PointMass), cudaMemcpyHostToDevice);

    double dt = 1.0;
    int steps = 1000;
    int numThreads = 256;
    int numBlocks = (masses.size() + numThreads - 1) / numThreads;

    // Start timing
    cudaEventRecord(start);

    for (int step = 0; step < steps; ++step) {
        updateForcesAndTorques<<<numBlocks, numThreads>>>(d_masses, masses.size());
        updateKinematics<<<numBlocks, numThreads>>>(d_masses, masses.size(), dt);

        // Display loading bar
        displayLoadingBar(step, steps);
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(masses.data(), d_masses, masses.size() * sizeof(PointMass), cudaMemcpyDeviceToHost);
    cudaFree(d_masses);

    std::ofstream outFile("simulation_data.txt");

    for (const auto& mass : masses) {
        outFile << mass.position[0] << " " << mass.position[1] << " " << mass.position[2] << " " << mass.mass << " ";
    }
    outFile << "\n";

    outFile.close();

    double megaTrialsPerSecond = (double)(NUMPLANETS * NUMPLANETS * steps) / (milliseconds / 1000.0) / 1000000.0;
    fprintf(stderr, "NUMPLANETS: %8d, Performance: %6.2lf MegaTrials/Second\n", NUMPLANETS, megaTrialsPerSecond);

    std::ofstream perfFile("performance.csv", std::ios_base::app);
    if (perfFile.is_open()) {
        perfFile << NUMPLANETS << "," << megaTrialsPerSecond << "\n";
        perfFile.close();
    } else {
        std::cerr << "Unable to open performance.csv for writing\n";
    }

    // Complete the loading bar
    displayLoadingBar(steps, steps);
    std::cout << std::endl;

    return 0;
}
