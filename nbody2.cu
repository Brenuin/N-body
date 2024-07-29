#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef NUMPLANETS
#define NUMPLANETS 6000
#endif

#ifndef NUMSTARS
#define NUMSTARS 500
#endif

#ifndef NUMBLACKHOLES
#define NUMBLACKHOLES 1
#endif

#define G 6.67430e-11
#define softening 1e-9

struct Particle {
    double mass;
    double position[3];
    double velocity[3];
    double force[3];
};

struct Node {
    bool isInternal;
    double mass;
    double centerOfMass[3];
    double boundary[6]; // xmin, xmax, ymin, ymax, zmin, zmax
    Particle* particle; // Null if this is an internal node
    Node* children[8];  // Null if this is an external node

    Node() : isInternal(false), mass(0), particle(nullptr) {
        for (int i = 0; i < 8; ++i) children[i] = nullptr;
    }
};

// Random number generator in range [min, max]
double randomInRange(double min, double max) {
    return min + static_cast<double>(rand()) / RAND_MAX * (max - min);
}

// Generate a position within a spiral galaxy
void generateSpiralPosition(double* position, const double* center, double radius, double height, int numArms) {
    double theta = randomInRange(0, 2 * M_PI); // Random angle
    double armOffset = randomInRange(0, 2 * M_PI / numArms); // Offset to create multiple arms
    double r = radius * sqrt(randomInRange(0, 1)); // Higher probability near center

    // Logarithmic spiral parameters
    double a = 1.0;
    double b = 0.3; // Controls the tightness of the spiral

    double x = r * cos(theta + armOffset) * exp(b * theta);
    double y = r * sin(theta + armOffset) * exp(b * theta);

    // Add a perturbation to simulate spiral arms
    double perturbation = 0.1 * radius * sin(numArms * theta);

    position[0] = center[0] + x + perturbation;
    position[1] = center[1] + y + perturbation;
    position[2] = center[2] + randomInRange(-height, height); // Flatten in Z dimension
}

void insertParticle(Node* node, Particle* p) {
    // If the node is empty (leaf), insert the particle here
    if (!node->isInternal && node->particle == nullptr) {
        node->particle = p;
        return;
    }

    // If the node is not a leaf, update the center of mass and total mass
    if (!node->isInternal) {
        // Convert this node to an internal node and re-insert the existing particle
        node->isInternal = true;
        Particle* existingParticle = node->particle;
        node->particle = nullptr;
        insertParticle(node, existingParticle);
    }

    // Update the center of mass and total mass
    node->mass += p->mass;
    node->centerOfMass[0] = (node->centerOfMass[0] * (node->mass - p->mass) + p->position[0] * p->mass) / node->mass;
    node->centerOfMass[1] = (node->centerOfMass[1] * (node->mass - p->mass) + p->position[1] * p->mass) / node->mass;
    node->centerOfMass[2] = (node->centerOfMass[2] * (node->mass - p->mass) + p->position[2] * p->mass) / node->mass;

    // Determine the appropriate child node
    int octant = 0;
    if (p->position[0] > (node->boundary[0] + node->boundary[1]) / 2) octant += 1;
    if (p->position[1] > (node->boundary[2] + node->boundary[3]) / 2) octant += 2;
    if (p->position[2] > (node->boundary[4] + node->boundary[5]) / 2) octant += 4;

    // Create the child node if it doesn't exist
    if (node->children[octant] == nullptr) {
        node->children[octant] = new Node();
        // Set the boundary for the child node
        for (int i = 0; i < 6; ++i) node->children[octant]->boundary[i] = node->boundary[i];
        if (octant & 1) {
            node->children[octant]->boundary[0] = (node->boundary[0] + node->boundary[1]) / 2;
        } else {
            node->children[octant]->boundary[1] = (node->boundary[0] + node->boundary[1]) / 2;
        }
        if (octant & 2) {
            node->children[octant]->boundary[2] = (node->boundary[2] + node->boundary[3]) / 2;
        } else {
            node->children[octant]->boundary[3] = (node->boundary[2] + node->boundary[3]) / 2;
        }
        if (octant & 4) {
            node->children[octant]->boundary[4] = (node->boundary[4] + node->boundary[5]) / 2;
        } else {
            node->children[octant]->boundary[5] = (node->boundary[4] + node->boundary[5]) / 2;
        }
    }

    // Recursively insert the particle into the appropriate child node
    insertParticle(node->children[octant], p);
}

void calculateForce(Node* node, Particle* p, double theta) {
    if (node == nullptr || (node->particle == p && !node->isInternal)) return;

    double dx = node->centerOfMass[0] - p->position[0];
    double dy = node->centerOfMass[1] - p->position[1];
    double dz = node->centerOfMass[2] - p->position[2];
    double distance = sqrt(dx * dx + dy * dy + dz * dz);

    if (node->isInternal) {
        double s = node->boundary[1] - node->boundary[0];
        if (s / distance < theta) {
            // Treat this node as a single particle
            double forceMagnitude = G * node->mass * p->mass / (distance * distance + softening * softening);
            p->force[0] += forceMagnitude * dx / distance;
            p->force[1] += forceMagnitude * dy / distance;
            p->force[2] += forceMagnitude * dz / distance;
        } else {
            // Recursively calculate forces from child nodes
            for (int i = 0; i < 8; ++i) {
                calculateForce(node->children[i], p, theta);
            }
        }
    } else {
        // Calculate force from the single particle in this external node
        if (node->particle != p) {
            double forceMagnitude = G * node->particle->mass * p->mass / (distance * distance + softening * softening);
            p->force[0] += forceMagnitude * dx / distance;
            p->force[1] += forceMagnitude * dy / distance;
            p->force[2] += forceMagnitude * dz / distance;
        }
    }
}

__global__ void updateKinematics(Particle* particles, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    particles[i].velocity[0] += particles[i].force[0] / particles[i].mass * dt;
    particles[i].velocity[1] += particles[i].force[1] / particles[i].mass * dt;
    particles[i].velocity[2] += particles[i].force[2] / particles[i].mass * dt;

    particles[i].position[0] += particles[i].velocity[0] * dt;
    particles[i].position[1] += particles[i].velocity[1] * dt;
    particles[i].position[2] += particles[i].velocity[2] * dt;
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

void generate_galaxy(std::vector<Particle>& particles, int numPlanets, int numStars, int numBlackHoles, const double* center, double radius, double mass_min, double mass_max, int numArms, const double* initialVelocity = nullptr) {
    int totalBodies = numPlanets + numStars + numBlackHoles;
    double totalMass = 0.0;
    double mass;

    // Generate planets
    for (int i = 0; i < numPlanets; ++i) {
        double position[3];
        generateSpiralPosition(position, center, radius, radius * 0.1, numArms);

        double velocity[3] = {0.0, 0.0, 0.0};
        if (initialVelocity != nullptr) {
            velocity[0] = initialVelocity[0];
            velocity[1] = initialVelocity[1];
            velocity[2] = initialVelocity[2];
        }

        mass = randomInRange(100 * mass_min, 100 * mass_max);
        totalMass += mass;

        particles.push_back(Particle{mass, {position[0], position[1], position[2]}, {velocity[0], velocity[1], velocity[2]}, {0.0, 0.0, 0.0}});
    }

    // Generate stars
    for (int i = 0; i < numStars; ++i) {
        double position[3];
        generateSpiralPosition(position, center, radius, radius * 0.1, numArms);

        double velocity[3] = {0.0, 0.0, 0.0};
        if (initialVelocity != nullptr) {
            velocity[0] = initialVelocity[0];
            velocity[1] = initialVelocity[1];
            velocity[2] = initialVelocity[2];
        }

        mass = randomInRange(mass_min, mass_max);
        totalMass += mass;

        particles.push_back(Particle{mass, {position[0], position[1], position[2]}, {velocity[0], velocity[1], velocity[2]}, {0.0, 0.0, 0.0}});
    }

    // Generate black holes
    for (int i = 0; i < numBlackHoles; ++i) {
        double position[3] = {center[0], center[1], center[2]}; // Place black hole at the center

        double velocity[3] = {0.0, 0.0, 0.0};
        if (initialVelocity != nullptr) {
            velocity[0] = initialVelocity[0];
            velocity[1] = initialVelocity[1];
            velocity[2] = initialVelocity[2];
        }

        mass = randomInRange(1000000 * mass_min, 15000000 * mass_max);
        totalMass += mass;

        particles.push_back(Particle{mass, {position[0], position[1], position[2]}, {velocity[0], velocity[1], velocity[2]}, {0.0, 0.0, 0.0}});
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    srand(static_cast<unsigned int>(time(NULL)));

    double randsize_min = 1.0e23, randsize_max = 1.0e30;
    double galaxyRadius = 25.0e14;
    int numArms = 5;

    std::vector<Particle> particles;
    double galaxyCenter1[3] = {0.0, 0.0, 0.0};
    double galaxyCenter2[3] = {120.0e14, 0.0, 0.0};

    double initialVelocity1[3] = {100000.0, 0.0, 0.0};
    double initialVelocity2[3] = {-100000.0, 0.0, 0.0};

    generate_galaxy(particles, NUMPLANETS, NUMSTARS, NUMBLACKHOLES, galaxyCenter1, galaxyRadius, randsize_min, randsize_max, numArms, initialVelocity1);
    generate_galaxy(particles, NUMPLANETS, NUMSTARS, NUMBLACKHOLES, galaxyCenter2, galaxyRadius, randsize_min, randsize_max, numArms, initialVelocity2);

    // Initialize CUDA memory for particles
    Particle* d_particles;
    cudaMalloc(&d_particles, particles.size() * sizeof(Particle));
    cudaMemcpy(d_particles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

    double dt = 4000000; // Time step
    int steps = 1000;
    int numThreads = 256;
    int numBlocks = (particles.size() + numThreads - 1) / numThreads;

    // Open the output file once
    std::ofstream outFile("simulation_data.txt");

    // Start timing
    cudaEventRecord(start);

    for (int step = 0; step < steps; ++step) {
        // Build the tree for each step
        Node* root = new Node();
        root->boundary[0] = -2.0 * galaxyRadius;
        root->boundary[1] = 2.0 * galaxyRadius;
        root->boundary[2] = -2.0 * galaxyRadius;
        root->boundary[3] = 2.0 * galaxyRadius;
        root->boundary[4] = -2.0 * galaxyRadius;
        root->boundary[5] = 2.0 * galaxyRadius;

        for (Particle& p : particles) {
            p.force[0] = p.force[1] = p.force[2] = 0.0; // Reset forces
            insertParticle(root, &p);
        }

        // Calculate forces using Barnes-Hut
        double theta = 0.5;
        for (Particle& p : particles) {
            calculateForce(root, &p, theta);
        }

        // Copy updated particles back to device
        cudaMemcpy(d_particles, particles.data(), particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

        // Update kinematics using CUDA
        updateKinematics<<<numBlocks, numThreads>>>(d_particles, particles.size(), dt);

        // Copy data back to host to write to file
        cudaMemcpy(particles.data(), d_particles, particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

        // Write the positions of all particles at the current time step
        for (const auto& p : particles) {
            outFile << p.position[0] << " " << p.position[1] << " " << p.position[2] << " " << p.mass << " ";
        }
        outFile << "\n";

        // Display loading bar
        displayLoadingBar(step, steps);

        // Debugging: Print step number
        std::cout << "Completed step " << step + 1 << " of " << steps << std::endl;

        // Clean up tree nodes
        // (Add recursive function to delete all nodes in the tree)
    }

    // Close the output file
    outFile.close();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_particles);

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
