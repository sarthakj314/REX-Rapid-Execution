# Backpropagation Engine

A modern C++ implementation of automatic differentiation and neural network backpropagation. This library provides a clean, intuitive interface for building and training computational graphs with automatic gradient computation.

## Features

### Multi-Linear Map (MLM) Class
- **Basic Operations**
  - Matrix multiplication (`matmul`)
  - Element-wise multiplication (`hadamard`)
  - Addition and subtraction
  - Transpose
  - Scalar multiplication and addition

- **Element-wise Functions**
  - ReLU
  - Sigmoid
  - Tanh
  - Custom function application

- **Static Constructors**
  - Zeros matrix
  - Ones matrix
  - Identity matrix

- **Statistical Functions**
  - Sum
  - Mean
  - Max/Min
  - Median
  - Mode
  - Variance
  - Standard deviation

### Variable (var) Class
- **Automatic Differentiation**
  - Forward pass computation
  - Backward pass gradient computation
  - Gradient accumulation
  - Topological sorting for correct gradient flow

- **Neural Network Operations**
  - Matrix multiplication
  - Bias addition
  - Activation functions (ReLU, Sigmoid, Tanh)
  - Loss computation

- **Optimization**
  - Gradient descent step
  - Gradient zeroing
  - Learning rate control

- **Visualization**
  - Pretty-printed computational graph
  - Roman numeral node labeling
  - Color-coded output
  - Parent-child relationship display

## Usage Example

```cpp
// Create input, weights, and bias
var x(MLM({{1.0, 2.0}}), true);  // 1x2 input
var W(MLM({{1.0, 0.0}, {0.0, 1.0}}));  // 2x2 weights
var b(MLM({{0.1, 0.1}}));  // 1x2 bias

// Forward pass: f = (x * W + b).relu()
var h = x.matmul(W);
var h_b = h + b;
var out = h_b.relu();

// Backward pass
out.backward();

// Optimization step
double learning_rate = 0.1;
out.step(learning_rate);

// Zero gradients for next iteration
out.zero_grad();

// Visualize computational graph
out.draw_graph();
```

## Installation

### Requirements
- C++17 or later
- Standard library only (no external dependencies)

### Building
```bash
g++ -std=c++17 backprop_engine.cpp test_backprop.cpp -o test_backprop
```

## API Reference

### MLM Class
```cpp
class MLM {
    // Constructors
    MLM();
    MLM(size_t rows, size_t cols);
    MLM(const std::vector<std::vector<double>>& data);
    
    // Operations
    MLM matmul(const MLM& other) const;
    MLM hadamard(const MLM& other) const;
    MLM add(const MLM& other) const;
    MLM subtract(const MLM& other) const;
    MLM transpose() const;
    
    // Scalar operations
    MLM scale(double scalar) const;
    MLM shift(double scalar) const;
    
    // Statistical functions
    double sum() const;
    double mean() const;
    double variance() const;
    // ... and more
};
```

### var Class
```cpp
class var {
    // Constructors
    var(const MLM& val, bool is_input = false);
    var(const MLM& val, std::string op, std::vector<var*> p, bool is_input = false);
    
    // Operations
    var& matmul(var& rhs);
    var& operator+(var& rhs);
    var& operator-(var& rhs);
    var& operator*(var& rhs);
    
    // Activation functions
    var& relu();
    var& sigmoid();
    var& tanh();
    
    // Training
    void backward();
    void step(double learning_rate);
    void zero_grad();
};
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
