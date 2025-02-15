#include "backprop_engine.h"
#include <random>
#include <iostream>

#define vdd std::vector<std::vector<double>>

// Generate synthetic data: y = 2x1 + 3x2 + 1 + noise
std::tuple<var, var> generate_data(int n_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 0.1);
    
    vdd x_data(n_samples, std::vector<double>(2));
    vdd y_data(n_samples, std::vector<double>(1));
    
    for (int i = 0; i < n_samples; i++) {
        x_data[i][0] = i * 0.1;  // x1 values from 0 to 0.1*n_samples
        x_data[i][1] = i * 0.05; // x2 values from 0 to 0.05*n_samples
        y_data[i][0] = 2 * x_data[i][0] + 3 * x_data[i][1] + 1 + noise(gen);  // y = 2x1 + 3x2 + 1 + noise
    }

    var x(x_data); x.freeze(); x.set_input(true);
    var y(y_data); y.freeze(); 
    return {x, y};
}

// Mean squared error loss
var& mse_loss(var& pred, var& target) {
    var& diff = pred - target;
    var& sqr = diff.hadamard(diff);
    var& loss = sqr.mean();
    return loss;
}

int main() {
    // Set debug mode for each class separately
    MLM::DEBUG_MODE = true;  // Debug matrix operations
    var::DEBUG_MODE = true;  // Debug backprop operations
    
    // Generate synthetic data
    int n_samples = 10;
    auto [x, y] = generate_data(n_samples);
    
    // Create model parameters
    var W = var::zeros(2, 1);
    var b = var::zeros(1, 1);
    var b_broadcast = var::ones(n_samples, 1);
    b_broadcast.freeze();
    
    // Training hyperparameters
    double learning_rate = 0.01;
    int epochs = 100;
    
    std::cout << "Training linear regression...\n";
    std::cout << "True parameters: W = [2.0, 3.0]^T, b = 1.0\n\n";
    
    // Training loop
    var loss(MLM::zeros(1, 1));  // Declare outside loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        var pred = x.matmul(W) + b_broadcast.matmul(b);

        var diff = pred - y;
        loss = diff;

        loss.draw_graph();
        
        // Backward pass
        loss.zero_grad();
        loss.backward();

        // Update parameters
        loss.step(learning_rate);
        
        // Print progress every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << ", Loss: " << loss.value().at(0, 0)
                      << ", W: " << W.value()
                      << ", b: " << b.value().at(0, 0) << "\n";
        }
    }
    
    // Print final parameters
    std::cout << "\nFinal parameters:\n";
    std::cout << "W1: " << W.value().at(0, 0) << " (true: 2.0)\n";
    std::cout << "W2: " << W.value().at(1, 0) << " (true: 3.0)\n";
    std::cout << "b: " << b.value().at(0, 0) << " (true: 1.0)\n";
    
    // Visualize computational graph
    std::cout << "\nComputational graph for final iteration:\n";
    loss.draw_graph();
    
    return 0;
}