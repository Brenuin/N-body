#include <iostream>
#include <omp.h>

int main() {
    int maxThreads = omp_get_max_threads();
    std::cout << "Maximum number of threads available: " << maxThreads << std::endl;
    return 0;
}
