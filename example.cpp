#define BACKPROP_ENGINE_IMPLEMENTATION
#include "backprop_engine.h"
#include <bits/stdc++.h>

int main() {
    // Two stack variables
    var x(2.0), y(3.0);

    // Build an expression: z = (x * y) + sin(y)
    var z = (x * y) + y.sin();

    z.backward();

    std::cout << "After backward:\n";
    std::cout << "dz/dx= " << x.grad() << "\n";
    std::cout << "dz/dy= " << y.grad() << "\n\n";

    z.draw_graph();

    // Clean up dynamic objects
    free_all_vars();
    return 0;
}