#include "backprop_engine.h"
#include <iostream>

#define vdd std::vector<std::vector<double>>

int main() {

    var x(vdd{{1.0, 2.0}, {3.0, 4.0}});
    var y(vdd{{0.5, 0.5}, {0.5, 0.5}});
    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << y << std::endl;

    // Test derivatives
    std::cout << "\n=== Testing Derivatives ===\n";

    // Test matmul derivative
    std::cout << "\n--- Testing matmul derivative ---\n";
    var x1(vdd{{1.0, 2.0}, {3.0, 4.0}});
    var y1(vdd{{0.5, 0.5}, {0.5, 0.5}});
    var z1 = x1.matmul(y1);
    z1.backward();
    std::cout << "dx1:\n" << x1.grad() << "\n";
    std::cout << "dy1:\n" << y1.grad() << "\n";
    z1.zero_grad();

    // Test hadamard derivative
    std::cout << "\n--- Testing hadamard derivative ---\n"; 
    var x2(vdd{{1.0, 2.0}, {3.0, 4.0}});
    var y2(vdd{{0.5, 0.5}, {0.5, 0.5}});
    var z2 = x2.hadamard(y2);
    z2.backward();
    std::cout << "dx2:\n" << x2.grad() << "\n";
    std::cout << "dy2:\n" << y2.grad() << "\n";
    z2.zero_grad();

    // Test add derivative
    std::cout << "\n--- Testing add derivative ---\n";
    var x3(vdd{{1.0, 2.0}, {3.0, 4.0}});
    var y3(vdd{{0.5, 0.5}, {0.5, 0.5}});
    var z3 = x3 + y3;
    z3.backward();
    std::cout << "dx3:\n" << x3.grad() << "\n";
    std::cout << "dy3:\n" << y3.grad() << "\n";
    z3.zero_grad();

    // Test subtract derivative
    std::cout << "\n--- Testing subtract derivative ---\n";
    var x4(vdd{{1.0, 2.0}, {3.0, 4.0}});
    var y4(vdd{{0.5, 0.5}, {0.5, 0.5}});
    var z4 = x4 - y4;
    z4.backward();
    std::cout << "dx4:\n" << x4.grad() << "\n";
    std::cout << "dy4:\n" << y4.grad() << "\n";
    z4.zero_grad();

    // Test scale derivative
    std::cout << "\n--- Testing scale derivative ---\n";
    var x5(vdd{{1.0, 2.0}, {3.0, 4.0}});
    var z5 = x5 * 2.0;
    z5.backward();
    std::cout << "dx5:\n" << x5.grad() << "\n";
    z5.zero_grad();

    // Test transpose derivative
    std::cout << "\n--- Testing transpose derivative ---\n";
    var x6(vdd{{1.0, 2.0}, {3.0, 4.0}});
    var z6 = x6.transpose();
    z6.backward();
    std::cout << "dx6:\n" << x6.grad() << "\n";
    z6.zero_grad();

    // Test shift derivative
    std::cout << "\n--- Testing shift derivative ---\n";
    var x7(vdd{{1.0, 2.0}, {3.0, 4.0}});
    var z7 = x7 + 1.0;
    z7.backward();
    std::cout << "dx7:\n" << x7.grad() << "\n";
    z7.zero_grad();

    // Test ReLU derivative
    std::cout << "\n--- Testing ReLU derivative ---\n";
    var x8(vdd{{-1.0, 2.0}, {3.0, -4.0}});
    var z8 = x8.relu();
    z8.backward();
    std::cout << "dx8:\n" << x8.grad() << "\n";
    z8.zero_grad();

    // Test sigmoid derivative
    std::cout << "\n--- Testing sigmoid derivative ---\n";
    var x9(vdd{{1.0, 2.0}, {3.0, 4.0}});
    var z9 = x9.sigmoid();
    z9.backward();
    std::cout << "dx9:\n" << x9.grad() << "\n";
    z9.zero_grad();

    // Test tanh derivative
    std::cout << "\n--- Testing tanh derivative ---\n";
    var x10(vdd{{1.0, 2.0}, {3.0, 4.0}});
    var z10 = x10.tanh();
    z10.backward();
    std::cout << "dx10:\n" << x10.grad() << "\n";
    z10.zero_grad();

    return 0;
}