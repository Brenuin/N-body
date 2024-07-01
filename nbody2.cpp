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
#define NUMT -1   //Number of threads to be used in the for loop
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

void setOrbitalVelocity(PointMass &mass, const PointMass &sun) {
    // Calculate the distance between the mass and the sun
    double dx = mass.position[0] - sun.position[0];
    double dy = mass.position[1] - sun.position[1];
    double dz = mass.position[2] - sun.position[2];
    double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    
    // Calculate the orbital velocity magnitude
    double orbitalVelocity = std::sqrt(G * sun.mass / distance);
    
    // Randomly choose a direction for the orbital velocity
    double angle = static_cast<double>(rand()) / RAND_MAX * 2 * M_PI;
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
void generate_celestial_body(std::vector<PointMass>& masses, int count, double mass_min, double mass_max, double distance_min, double distance_max, const PointMass& reference_body) {
    for (int i = 0; i < count; ++i) {
        std::vector<double> position(3);
        position[0] = randomInRange(distance_min, distance_max);
        position[1] = randomInRange(distance_min, distance_max);
        position[2] = randomInRange(distance_min, distance_max);

        std::vector<double> velocity(3, 0.0); // 0.0 for vx, vy, vz
        std::vector<double> force(3, 0.0);    // 0.0 for fx, fy, fz
        std::vector<double> angularVelocity(3);
        angularVelocity[0] = static_cast<double>(std::rand()) / RAND_MAX * 1e5;
        angularVelocity[1] = static_cast<double>(std::rand()) / RAND_MAX * 1e5;
        angularVelocity[2] = static_cast<double>(std::rand()) / RAND_MAX * 1e5;

        std::vector<double> torque(3, 0.0);  // 0.0 for tx, ty, tz
        double momentOfInertia = static_cast<double>(std::rand()) / RAND_MAX * 1e10;

        double mass = randomInRange(mass_min, mass_max);
        PointMass newBody(mass, position, velocity, force, angularVelocity, torque, momentOfInertia);

        setOrbitalVelocity(newBody, reference_body);

        masses.push_back(newBody);
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
    std::srand(static_cast<unsigned int>(std::time(NULL)));

    // Define control variables as ranges
    double randsizesun_min = 1.0e24, randsizesun_max = 2.0e25;
    double randsizeplanet_min = 1.0e25, randsizeplanet_max = 1.0e26;
    double randdistancesun_min = 2.0e9, randdistancesun_max = 2.0e12;
    double randdistancesplanet_min = 1.0e7, randdistancesplanet_max = 2.0e12;
    double randdistancesstar_min = 1.0e7, randdistancesstar_max = 2.0e12;
    double randsizestar_min = 1.0e24, randsizestar_max = 2.0e35;

    // Define the point masses
    std::vector<PointMass> masses;

    double sunMass = randomInRange(randsizesun_min, randsizesun_max); // Random mass of the sun
    double distance = randomInRange(randdistancesun_min, randdistancesun_max); // Random distance between the two suns
    double velocity = std::sqrt(G * sunMass * 0.7 * (distance / 2)); // Orbital velocity for each sun

    std::vector<double> sun1Position = {-distance / 2, 0.0, 0.0};
    std::vector<double> sun1Velocity = {0.0, velocity, 0.0};
    std::vector<double> sun2Position = {distance / 2, 0.0, 0.0};
    std::vector<double> sun2Velocity = {0.0, -velocity, 0.0};

    std::vector<double> sunForce(3, 0.0);
    std::vector<double> sunAngularVelocity(3, 0.0);
    std::vector<double> sunTorque(3, 0.0);
    double sunMomentOfInertia = 1.0e40; // Arbitrary large moment of inertia

    PointMass sun1(sunMass, sun1Position, sun1Velocity, sunForce, sunAngularVelocity, sunTorque, sunMomentOfInertia);
    masses.push_back(sun1);

    PointMass sun2(sunMass, sun2Position, sun2Velocity, sunForce, sunAngularVelocity, sunTorque, sunMomentOfInertia);
    masses.push_back(sun2);

    generate_celestial_body(masses, NUMPLANETS, randsizeplanet_min, randsizeplanet_max, randdistancesplanet_min, randdistancesplanet_max, sun1);
    generate_celestial_body(masses, NUMSTARS, randsizestar_min, randsizestar_max, randdistancesstar_min, randdistancesstar_max, sun1);

    // Open a file to write the simulation data
    std::ofstream outFile("simulation_data.txt");

    // Simulation parameters
    double dt = 0.01; // Time step in seconds
    int steps = 500; // Number of steps to simulate

    // Main simulation loop
    for (int step = 0; step < steps; ++step) {
        updateForcesAndTorques(masses);
        updateKinematics(masses, dt);
        printf("Step %d:\n", step);
        displayLoadingBar(step, steps);
        // Write the positions of all masses at the current time step
        for (size_t i = 0; i < masses.size(); ++i) {
            outFile << masses[i].position[0] << " " << masses[i].position[1] << " " << masses[i].position[2] << " " << masses[i].mass << " ";
        }
        outFile << "\n"; // End of time step
    }

    outFile.close();
    
    displayLoadingBar(steps, steps);
    std::cout << std::endl;

    double time1 = omp_get_wtime();
    double megaTrialsPerSecond = (double)(NUMPLANETS * NUMPLANETS * steps) / (time1 - time0) / 1000000.;
    fprintf(stderr, "NUMT: %2d, NUMPLANETS: %8d, Performance: %6.2lf MegaTrials/Second\n", NUMT, NUMPLANETS, megaTrialsPerSecond);

    std::ofstream perfFile("performance.csv", std::ios_base::app);
    if (perfFile.is_open()) {
        perfFile << NUMT << "," << NUMPLANETS << "," << megaTrialsPerSecond << "\n";
        perfFile.close();
    } else {
        std::cerr << "Unable to open performance.csv for writing\n";
    }

    return 0;
}
