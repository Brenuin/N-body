#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>

// Define constants
const double G = 6.67430e-3; // Scaled-down gravitational constant
const double BLACK_HOLE_MASS = 1e3; // Scaled-down mass of the black hole
const double MAX_DISTANCE = 10.0; // Scaled-down maximum distance from the black hole for other bodies
const double MIN_DISTANCE = 1.0; // Scaled-down minimum distance to avoid extremely high velocities
const double SOFTENING = 1e-5; // Small softening factor to avoid division by zero

struct PointMass {
    double mass;
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> force;
    double momentOfInertia;

    PointMass(double m, const std::vector<double>& p, const std::vector<double>& v, const std::vector<double>& f, double I)
        : mass(m), position(p), velocity(v), force(f), momentOfInertia(I) {}
};

std::vector<double> calculateOrbitalVelocity(double mass, double distance) {
    double speed = std::sqrt(G * BLACK_HOLE_MASS / (distance + SOFTENING)) * 0.1; // Further reduced speed for visibility
    return {0, speed, 0}; // Assuming initial orbit in the x-y plane
}

int main() {
    std::srand(static_cast<unsigned int>(std::time(NULL)));

    std::ofstream outFile("initial_state.txt");

    // Create the black hole at the center
    std::vector<double> blackHolePosition = {0.0, 0.0, 0.0};
    std::vector<double> blackHoleVelocity = {0.0, 0.0, 0.0};
    std::vector<double> blackHoleForce = {0.0, 0.0, 0.0};
    double blackHoleMomentOfInertia = 0.0; // Not needed for black hole in this simple model

    outFile << BLACK_HOLE_MASS << " " << blackHolePosition[0] << " " << blackHolePosition[1] << " " << blackHolePosition[2] << " "
            << blackHoleVelocity[0] << " " << blackHoleVelocity[1] << " " << blackHoleVelocity[2] << " "
            << blackHoleForce[0] << " " << blackHoleForce[1] << " " << blackHoleForce[2] << " " << blackHoleMomentOfInertia << "\n";

    // Generate random bodies
    const int NUM_BODIES = 20;
    for (int i = 0; i < NUM_BODIES; ++i) {
        double distance = MIN_DISTANCE + static_cast<double>(std::rand()) / RAND_MAX * (MAX_DISTANCE - MIN_DISTANCE);
        double angle = static_cast<double>(std::rand()) / RAND_MAX * 2 * M_PI;

        std::vector<double> position = {distance * std::cos(angle), distance * std::sin(angle), 0.0};
        std::vector<double> velocity = calculateOrbitalVelocity(BLACK_HOLE_MASS, distance);
        std::vector<double> force(3, 0.0);

        double mass = 10.0 + static_cast<double>(std::rand()) / RAND_MAX * 90.0; // Scaled-down mass
        double radius = 1.0; // Scaled-down radius
        double momentOfInertia = (2.0 / 5.0) * mass * radius * radius;

        outFile << mass << " " << position[0] << " " << position[1] << " " << position[2] << " "
                << velocity[0] << " " << velocity[1] << " " << velocity[2] << " "
                << force[0] << " " << force[1] << " " << force[2] << " " << momentOfInertia << "\n";
    }

    outFile.close();
    return 0;
}
