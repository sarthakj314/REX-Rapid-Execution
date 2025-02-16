# Backpropagation Engine

A modern C++ implementation of automatic differentiation and neural network backpropagation. This library provides a clean, intuitive interface for building and training computational graphs with automatic gradient computation.

## Features

### Multi-Linear Map (MLM) Class
- Matrix operations (multiplication, addition, transpose, inverse)
- Element-wise operations (hadamard product, activation functions)
- Static constructors (zeros, ones, identity matrices)
- Statistical functions (sum, mean, max/min)

### Variable (var) Class
- Automatic differentiation with backpropagation
- Dynamic computational graph construction
- Gradient computation and accumulation
- Visualization of computational graphs

## Getting Started

### Prerequisites
- C++17 or later
- A C++ compiler (g++ or clang++)

### Building the Project
```bash
# Clone the repository
git clone https://github.com/sarthakj314/backprop-engine.git
cd backprop-engine

# Build the linear regression example
g++ -std=c++17 backprop_engine.cpp linear_regression.cpp -o linear_regression

# Build the tests
g++ -std=c++17 backprop_engine.cpp test_backprop.cpp -o test_backprop
```

### Running Examples

#### Linear Regression
```bash
# Run the linear regression example
./linear_regression
```

The example:
1. Generates synthetic data points from the plane y = 0.75x₁ + 3x₂ + 1
2. Trains a linear regression model to find these coefficients
3. Prints training progress and final results

Example output:
```
Training linear regression...
Epoch 200/2000
    Loss: 0.0123
    W1: 0.7489
    W2: 2.9876
    b: 0.9923
...
Final parameters:
W1: 0.7501 (true: 0.75)
W2: 2.9998 (true: 3.00)
b: 1.0002 (true: 1.00)
```

#### Running Tests
```bash
# Run the test suite
./test_backprop
```

### API Examples

#### Creating Variables
```cpp
// Create variables from data
var x({{1.0, 2.0}, {3.0, 4.0}});  // 2x2 matrix
var y = var::zeros(2, 1);          // 2x1 zero matrix
var z = var::ones(1, 2);           // 1x2 ones matrix

// Set properties
x.set_name("x");                   // Name for visualization
x.freeze();                        // Freeze gradients
```

#### Matrix Operations
```cpp
// Basic operations
var a = x.matmul(y);              // Matrix multiplication
var b = x + y;                    // Addition
var c = x - y;                    // Subtraction
var d = x.hadamard(y);            // Element-wise multiplication
var e = x.transpose();            // Transpose
var f = x.inv();                  // Matrix inverse

// Activation functions
var g = x.relu();                 // ReLU activation
var h = x.sigmoid();              // Sigmoid activation
var i = x.tanh();                 // Tanh activation
```

#### Training
```cpp
// Forward pass
var pred = x.matmul(W) + b;
var loss = mse_loss(pred, y);

// Backward pass
loss.zero_grad();                 // Zero all gradients
loss.backward();                  // Compute gradients
loss.step(learning_rate);         // Update parameters

// Visualization
loss.draw_graph();                // Visualize computational graph
```

## Implementation Details

### Automatic Differentiation
The engine implements reverse-mode automatic differentiation (backpropagation) by:
1. Building a computational graph during forward pass
2. Topologically sorting nodes for gradient computation
3. Computing gradients backwards through the graph
4. Accumulating gradients at each node

### Memory Management
- Smart pointer usage for safe memory management
- Automatic cleanup of computational graph
- Prevention of memory leaks in cyclic graphs

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
