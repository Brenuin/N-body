#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef TIME
#define TIME 300
#endif

#ifndef GALAXYR
#define GALAXYR 35.0e16 // Updated to the current galaxy radius
#endif

#ifndef NUMPLANETS
#define NUMPLANETS 10
#endif


#ifndef NUMSTARS
#define NUMSTARS 30
#endif

#ifndef NUMSMALLBLACKHOLES
#define NUMSMALLBLACKHOLES 2
#endif

#ifndef NUMBLACKHOLES
#define NUMBLACKHOLES 1
#endif

#ifndef THREADS
#define THREADS 256
#endif


#define G 6.67430e-11

struct PointMass {
    double mass;
    double position[3];
    double velocity[3];
    double force[3];
    double angularVelocity[3];
    double torque[3];
    double momentOfInertia;
};

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

__device__ double computeDistance(const PointMass &a, const PointMass &b) {
    double dx = b.position[0] - a.position[0];
    double dy = b.position[1] - a.position[1];
    double dz = b.position[2] - a.position[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__device__ void computeForce(const PointMass &a, const PointMass &b, double *force) {
    double distance = computeDistance(a, b);
    if (distance == 0) return;

    double magnitude = G * a.mass * b.mass / (distance * distance);

    force[0] = magnitude * (b.position[0] - a.position[0]) / distance;
    force[1] = magnitude * (b.position[1] - a.position[1]) / distance;
    force[2] = magnitude * (b.position[2] - a.position[2]) / distance;
}

__device__ void computeTorque(const PointMass &a, const double *force, double *torque) {
    torque[0] = a.position[1] * force[2] - a.position[2] * force[1];
    torque[1] = a.position[2] * force[0] - a.position[0] * force[2];
    torque[2] = a.position[0] * force[1] - a.position[1] * force[0];
}

__global__ void updateForcesAndTorques(PointMass *masses, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    masses[i].force[0] = masses[i].force[1] = masses[i].force[2] = 0.0;
    masses[i].torque[0] = masses[i].torque[1] = masses[i].torque[2] = 0.0;

    for (int j = 0; j < n; ++j) {
        if (i != j) {
            double force[3];
            computeForce(masses[i], masses[j], force);
            masses[i].force[0] += force[0];
            masses[i].force[1] += force[1];
            masses[i].force[2] += force[2];

            double torque[3];
            computeTorque(masses[i], force, torque);
            masses[i].torque[0] += torque[0];
            masses[i].torque[1] += torque[1];
            masses[i].torque[2] += torque[2];
        }
    }
}

__global__ void updateKinematics(PointMass *masses, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

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


double randomInRange(double min, double max) {
    return min + static_cast<double>(rand()) / RAND_MAX * (max - min);
}

void generateSpiralPosition(double* position, const double* center, double radius, double height, int numArms = 3, double base_b = 0.3) {
    
    int arm = rand() % numArms;  //random arm select
    double base_theta = arm * (2 * M_PI / numArms);  
    double armOffset = randomInRange(-M_PI / (8 * numArms), M_PI / (8 * numArms));  
    double theta = randomInRange(0, 4 * M_PI) + base_theta + armOffset;  
    double r = radius * sqrt(randomInRange(0, 1));  
    double b = base_b * randomInRange(0.9, 1.1);  
    double x = r * cos(theta) * exp(b * theta / (2 * M_PI));
    double y = r * sin(theta) * exp(b * theta / (2 * M_PI));

    double perturbation = 0.02 * radius * randomInRange(-1, 1);  


    position[0] = center[0] + x + perturbation;
    position[1] = center[1] + y + perturbation;
    position[2] = center[2] + randomInRange(-height, height);  //height
}



void generate_galaxy(std::vector<PointMass>& masses, int numPlanets, int numStars, int numSmallBlackHoles, int numBlackHoles, double* center, double radius, double mass_min, double mass_max, int numArms, const double* initialVelocity = nullptr) {
    int startIdx = masses.size();
    int totalBodies = numPlanets + numStars + numSmallBlackHoles + numBlackHoles; // Include black holes in the count
    double totalMass = 0.0;
    double mass;

    auto randomInRange = [](double min, double max) {
        return min + static_cast<double>(rand()) / RAND_MAX * (max - min);
    };

    // Generate planets
    for (int i = 0; i < numPlanets; ++i) {
        double position[3];
        generateSpiralPosition(position, center, radius, radius * 0.1, 3, 2.0);

        double velocity[3] = {0.0, 0.0, 0.0};
        double momentOfInertia = randomInRange(1e10, 1e20);

        mass = randomInRange(mass_min, mass_max);
        totalMass += mass;

        masses.push_back(PointMass{mass, {position[0], position[1], position[2]}, {velocity[0], velocity[1], velocity[2]}, {}, {}, {}, momentOfInertia});
    }

    // Generate stars
    for (int i = 0; i < numStars; ++i) {
        double position[3];
        generateSpiralPosition(position, center, radius, radius * 0.1, 3, 2.0);

        double velocity[3] = {0.0, 0.0, 0.0};
        double momentOfInertia = randomInRange(1e10, 1e20);

        mass = randomInRange(100 * mass_min, 100 * mass_max);
        totalMass += mass;

        masses.push_back(PointMass{mass, {position[0], position[1], position[2]}, {velocity[0], velocity[1], velocity[2]}, {}, {}, {}, momentOfInertia});
    }

    // Generate small black holes
    for (int i = 0; i < numSmallBlackHoles; ++i) {
        double position[3];
        generateSpiralPosition(position, center, radius, radius * 0.1, 3, 2.0);

        double velocity[3] = {0.0, 0.0, 0.0};
        double momentOfInertia = randomInRange(1e20, 1e30);

        mass = randomInRange(1000 * mass_min, 15000 * mass_max);
        totalMass += mass;

        masses.push_back(PointMass{mass, {position[0], position[1], position[2]}, {velocity[0], velocity[1], velocity[2]}, {}, {}, {}, momentOfInertia});
    }

    // Calculate the center of mass for all bodies including black holes
    double centerOfMass[3] = {0.0, 0.0, 0.0};
    double totalMassForCenter = 0.0;

    for (int i = startIdx; i < startIdx + totalBodies; ++i) {
        centerOfMass[0] += masses[i].position[0] * masses[i].mass;
        centerOfMass[1] += masses[i].position[1] * masses[i].mass;
        centerOfMass[2] += masses[i].position[2] * masses[i].mass;
        totalMassForCenter += masses[i].mass;
    }

    centerOfMass[0] /= totalMassForCenter;
    centerOfMass[1] /= totalMassForCenter;
    centerOfMass[2] /= totalMassForCenter;

    double totalMassIncludingBH = totalMassForCenter;

    // Generate large black holes closer to the center and apply initial velocity properly
    for (int i = 0; i < numBlackHoles; ++i) {
        double position[3];
        double distance = randomInRange(0, 0.5 * radius);  // Closer to the center
        double angle = randomInRange(0, 2 * M_PI);
        position[0] = center[0] + distance * cos(angle);
        position[1] = center[1] + distance * sin(angle);
        position[2] = center[2] + randomInRange(-0.05 * radius, 0.05 * radius);  // Less spread in z

        double velocity[3] = {0.0, 0.0, 0.0};
        double momentOfInertia = randomInRange(1e20, 1e30);

        mass = randomInRange(100000 * mass_min, 1500000 * mass_max);
        totalMassIncludingBH += mass;

        masses.push_back(PointMass{mass, {position[0], position[1], position[2]}, {velocity[0], velocity[1], velocity[2]}, {}, {}, {}, momentOfInertia});
    }

    // Calculate and apply orbital velocities including the initial velocity for all bodies
    for (int i = startIdx; i < masses.size(); ++i) {
        auto& mass = masses[i];
        double dx = mass.position[0] - centerOfMass[0];
        double dy = mass.position[1] - centerOfMass[1];
        double dz = mass.position[2] - centerOfMass[2];
        double distance = sqrt(dx * dx + dy * dy + dz * dz);

        double orbitalVelocity = sqrt(G * totalMassIncludingBH / distance);

        double angle = randomInRange(0, 2 * M_PI);
        double sinAngle = sin(angle);
        double cosAngle = cos(angle);

        if (dx != 0 || dy != 0) {
            mass.velocity[0] = -dy / distance * orbitalVelocity * cosAngle;
            mass.velocity[1] = dx / distance * orbitalVelocity * sinAngle;
            mass.velocity[2] = 0.0;
        } else {
            mass.velocity[0] = orbitalVelocity * cosAngle;
            mass.velocity[1] = orbitalVelocity * sinAngle;
            mass.velocity[2] = 0.0;
        }

        double normalComponent = (mass.velocity[0] * dx + mass.velocity[1] * dy) / (dx * dx + dy * dy);
        mass.velocity[0] -= normalComponent * dx;
        mass.velocity[1] -= normalComponent * dy;
        mass.velocity[2] = orbitalVelocity * sinAngle;

        // Apply the initial velocity for all bodies
        if (initialVelocity) {
            mass.velocity[0] += initialVelocity[0];
            mass.velocity[1] += initialVelocity[1];
            mass.velocity[2] += initialVelocity[2];
        }
    }
}


int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    srand(static_cast<unsigned int>(time(NULL)));

    double randsize_min = 1.0e23, randsize_max = 1.0e30;
    double galaxyRadius = GALAXYR;
    int numArms = 4;

    std::vector<PointMass> masses;
    double galaxyCenter1[3] = {0.0, 0.0, 0.0};
    double galaxyCenter2[3] = {6 * galaxyRadius, 0.0, 0.0};

    double dt = .5;
    int steps = TIME;
    double collision_time = dt * steps / 2;
    double initialVelocityMagnitude = .4* galaxyRadius / collision_time;

    double initialVelocity1[3] = {initialVelocityMagnitude , 0.0, 0.0};
    double initialVelocity2[3] = {-initialVelocityMagnitude , 0.0, 0.0};

    generate_galaxy(masses, NUMPLANETS, NUMSTARS, NUMBLACKHOLES, NUMSMALLBLACKHOLES, galaxyCenter1, galaxyRadius, randsize_min, randsize_max, numArms, initialVelocity1);
    generate_galaxy(masses, NUMPLANETS, NUMSTARS, NUMBLACKHOLES, NUMSMALLBLACKHOLES, galaxyCenter2, galaxyRadius, randsize_min, randsize_max, numArms, initialVelocity2);

    PointMass* d_masses;
    cudaMalloc(&d_masses, masses.size() * sizeof(PointMass));
    cudaMemcpy(d_masses, masses.data(), masses.size() * sizeof(PointMass), cudaMemcpyHostToDevice);

    int numThreads = THREADS;
    int numBlocks = (masses.size() + numThreads - 1) / numThreads;

    std::ofstream outFile("nbody.out");

    cudaEventRecord(start);

    for (int step = 0; step < steps; ++step) {
        updateForcesAndTorques<<<numBlocks, numThreads>>>(d_masses, masses.size());
        updateKinematics<<<numBlocks, numThreads>>>(d_masses, masses.size(), dt);

        cudaMemcpy(masses.data(), d_masses, masses.size() * sizeof(PointMass), cudaMemcpyDeviceToHost);

        for (const auto& mass : masses) {
            outFile << mass.position[0] << " " << mass.position[1] << " " << mass.position[2] << " " << mass.mass << "\n";
        }
        
        displayLoadingBar(step, steps);
        std::cout << "Completed step " << step + 1 << " of " << steps << std::endl;
    }

    outFile.close();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_masses);

    double megaTrialsPerSecond = (double)(NUMPLANETS * NUMPLANETS * steps) / (milliseconds / 1000.0) / 1000000.0;
    fprintf(stderr, "NUMPLANETS: %8d, THREADS: %4d, Performance: %6.2lf MegaTrials/Second\n", NUMPLANETS, THREADS, megaTrialsPerSecond);

    std::ofstream perfFile("performance.csv", std::ios_base::app);
    if (perfFile.is_open()) {
        perfFile << NUMPLANETS << "," << THREADS << "," << megaTrialsPerSecond << "\n";
        perfFile.close();
    } else {
        std::cerr << "Unable to open performance.csv for writing\n";
    }

    displayLoadingBar(steps, steps);
    std::cout << std::endl;

    return 0;
}