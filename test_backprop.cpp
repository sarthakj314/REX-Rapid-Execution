#include "backprop_engine.h"
#include <iostream>

int main() {
    // Test 1: Basic MLM Operations
    std::cout << "\n=== Test 1: Basic MLM Operations ===\n";
    
    // Create two 2x2 matrices
    std::vector<std::vector<double>> data1 = {{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::vector<double>> data2 = {{0.5, 0.5}, {0.5, 0.5}};
    
    MLM m1(data1);
    MLM m2(data2);
    
    std::cout << "Matrix 1:\n" << m1 << "\n";
    std::cout << "Matrix 2:\n" << m2 << "\n";
    
    // Test matrix operations
    std::cout << "Matrix multiplication:\n" << m1.matmul(m2) << "\n";
    std::cout << "Matrix addition:\n" << m1.add(m2) << "\n";
    std::cout << "Matrix hadamard product:\n" << m1.hadamard(m2) << "\n";
    std::cout << "Matrix transpose:\n" << m1.transpose() << "\n";
    std::cout << "Matrix scalar multiplication:\n" << m1.scale(2.0) << "\n";
    std::cout << "Matrix scalar addition:\n" << m1.shift(1.0) << "\n";

    // Test element-wise operations
    auto relu = [](double x) -> double { return x > 0 ? x : 0; };
    auto sigmoid = [](double x) { return 1.0 / (1.0 + exp(-x)); };
    auto tanh = [](double x) { return ::tanh(x); };

    std::cout << "ReLU:\n" << m1.apply(relu) << "\n";
    std::cout << "Sigmoid:\n" << m1.apply(sigmoid) << "\n";
    std::cout << "Tanh:\n" << m1.apply(tanh) << "\n";

    // Test static constructors
    std::cout << "2x2 zeros:\n" << MLM::zeros(2, 2) << "\n";
    std::cout << "2x2 ones:\n" << MLM::ones(2, 2) << "\n";
    std::cout << "2x2 identity:\n" << MLM::identity(2) << "\n";

    // Test utility functions
    std::cout << "Matrix 1 sum: " << m1.sum() << "\n";
    std::cout << "Matrix 1 mean: " << m1.mean() << "\n";
    std::cout << "Matrix 1 max: " << m1.max() << "\n";
    std::cout << "Matrix 1 min: " << m1.min() << "\n";

    // Test more complex computational graphs
    std::cout << "\n=== Test 1.5: Complex MLM Operations ===\n";
    
    // Create 3x3 matrices for more complex operations
    MLM m3({{1,2,3}, {4,5,6}, {7,8,9}});
    MLM m4({{9,8,7}, {6,5,4}, {3,2,1}});

    std::cout << "Matrix 3:\n" << m3 << "\n";
    std::cout << "Matrix 4:\n" << m4 << "\n";

    // Test chained operations
    MLM result = m3.matmul(m4).transpose().scale(0.5);
    std::cout << "Chained operations (m3 * m4)^T * 0.5:\n" << result << "\n";

    // Test element-wise operations with different activation functions
    MLM activated = result.apply(sigmoid);
    std::cout << "After sigmoid:\n" << activated << "\n";

    // Test matrix operations with different shapes
    MLM tall({{1,2}, {3,4}, {5,6}});  // 3x2
    MLM wide({{1,2,3}, {4,5,6}});     // 2x3

    std::cout << "Tall matrix:\n" << tall << "\n";
    std::cout << "Wide matrix:\n" << wide << "\n";
    std::cout << "Matrix multiplication (tall * wide):\n" << tall.matmul(wide) << "\n";

    // Test 2: Automatic Differentiation
    std::cout << "\n=== Test 2: Automatic Differentiation ===\n";
    
    // Create computational graph: f = (x * W + b).relu()
    var x(MLM({{1.0, 2.0}}), true);  // 1x2 input, is_input = true
    var W(MLM({{1.0, 0.0}, {0.0, 1.0}}));  // 2x2 weights 
    var b(MLM({{0.1, 0.1}}));  // 1x2 bias
    
    // Forward pass
    var h = x.matmul(W);
    var h_b = h + b;
    var out = h_b.relu();
    
    std::cout << "Forward pass result:\n" << out << "\n";
    
    // Backward pass
    out.backward();
    
    std::cout << "Gradients:\n";
    std::cout << "dx:\n" << x.grad() << "\n";
    std::cout << "dW:\n" << W.grad() << "\n";
    std::cout << "db:\n" << b.grad() << "\n";

    // Test 3: Optimization Step
    std::cout << "\n=== Test 3: Optimization Step ===\n";
    
    // Take a gradient step
    double learning_rate = 0.1;
    out.step(learning_rate);
    
    std::cout << "Updated W after gradient step:\n" << W.value() << "\n";
    
    // Zero out gradients for next iteration
    out.zero_grad();
    
    // Test 4: Visualization
    std::cout << "\n=== Test 4: Computational Graph ===\n";
    out.draw_graph();

    return 0;
} 